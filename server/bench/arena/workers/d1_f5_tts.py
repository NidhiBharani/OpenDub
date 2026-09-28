"""D1/D3/D4 worker: F5-TTS zero-shot cloning via the ``f5-tts`` package (server venv has 1.1.22).

params:
  model        F5 architecture config: F5TTS_v1_Base (stock en/zh) | F5TTS_Small (F5-Hindi) …
  ckpt_file    custom checkpoint (local path or hf://org/repo/file), e.g.
               hf://SPRINGLab/F5-Hindi-24KHz/model_2500000.safetensors
  vocab_file   vocab.txt the checkpoint was trained with
  languages    languages the checkpoint speaks (the stock one: [en, zh]); others → Unsupported
  nfe_step (32), cfg_strength (2), sway_sampling_coef (-1), speed (1.0), target_rms (0.1)
  use_target_duration  honour inputs.target_s via F5's ``fix_duration`` (D3); default true
  ref_text_mode        given (default) | asr (let F5 transcribe the reference) |
                       translit (spell the reference transcript in the target script with an
                       OpenAI-compatible LLM: translit_base_url, translit_model — what the app's
                       tts.f5_tts provider does for single-language checkpoints)

The app provider (app/providers/tts/f5_tts.py) adds Whisper take verification and speed retries;
here one call = one take, and best-of-N is the separate ``d3_best_of_n`` method candidate.
Output: 24 kHz mono wav.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve

SCRIPT_NAMES = {"hi": "Hindi (Devanagari script)", "ja": "Japanese katakana",
                "en": "English (Latin script)"}


def _resolve(path: str) -> str:
    if path.startswith("hf://"):
        from cached_path import cached_path

        return str(cached_path(path))
    return path


def _audio_loader_fallback() -> None:
    """Same torchcodec → soundfile fallback as the app provider (torchaudio.load may fail)."""
    import tempfile

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    with tempfile.NamedTemporaryFile(suffix=".wav") as probe:
        sf.write(probe.name, np.zeros(1600, dtype="float32"), 16000)
        try:
            torchaudio.load(probe.name)
            return
        except Exception as exc:  # noqa: BLE001 - version-dependent error types
            print(f"torchaudio.load unusable ({exc!r}); using soundfile", file=sys.stderr)

    def _sf_load(path, *_a, **_k):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _sf_load


def load(params: dict, lang: str):
    import torch
    from f5_tts.api import F5TTS

    dv.require_lang(lang, params.get("languages", ["en", "zh"]), "this F5 checkpoint")
    _audio_loader_fallback()
    device = params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = F5TTS(model=params.get("model", "F5TTS_v1_Base"),
                  ckpt_file=_resolve(params["ckpt_file"]) if params.get("ckpt_file") else "",
                  vocab_file=_resolve(params["vocab_file"]) if params.get("vocab_file") else "",
                  device=device)
    return {"model": model, "params": params, "lang": lang, "translit": {}}


def _transliterate(state: dict, text: str) -> str:
    p = state["params"]
    url, model = p.get("translit_base_url"), p.get("translit_model")
    if not (url and model and text):
        return text
    key = (text, state["lang"])
    if key in state["translit"]:
        return state["translit"][key]
    script = SCRIPT_NAMES.get(state["lang"], f"the native script of {state['lang']}")
    prompt = (f"Write the following sentence phonetically in {script}, the way a native reader "
              "would spell these exact foreign words. Do NOT translate; every word must be in "
              f"that script. Reply with the transliteration only.\n\n{text}")
    body = json.dumps({"model": model, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    out = (out.splitlines() or [text])[0].replace("‍", "").replace("‌", "")
    state["translit"][key] = out
    return out


def run(state: dict, item: dict, out: Path) -> dict:
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text

    p = state["params"]
    text = dv.text_of(item)
    ref, ref_text = dv.ref_of(item)
    mode = p.get("ref_text_mode", "given")
    if mode == "asr" or not ref_text:
        ref_text = ""  # F5 runs its internal Whisper on the reference
    elif mode == "translit":
        ref_text = _transliterate(state, ref_text)
    ref_file, ref_text = preprocess_ref_audio_text(ref, ref_text, show_info=lambda *_: None)

    fix_duration = None
    tgt = dv.target_s(item)
    if tgt and p.get("use_target_duration", True):
        fix_duration = dv.duration_s(ref_file) + tgt  # F5's fix_duration includes the prompt
    seed = dv.item_seed(item, p)
    wav = dv.wav_path(out)
    hash_seed = os.environ.get("PYTHONHASHSEED")  # f5 seed_everything clobbers it (see provider)
    try:
        state["model"].infer(ref_file=ref_file, ref_text=ref_text, gen_text=text,
                             file_wave=str(wav), seed=seed, show_info=lambda *_: None,
                             nfe_step=int(p.get("nfe_step", 32)),
                             cfg_strength=float(p.get("cfg_strength", 2.0)),
                             sway_sampling_coef=float(p.get("sway_sampling_coef", -1.0)),
                             speed=float(p.get("speed", 1.0)),
                             target_rms=float(p.get("target_rms", 0.1)),
                             fix_duration=fix_duration)
    finally:
        if hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed
    if not wav.exists():
        raise Unsupported("F5 produced no audio (empty text after vocab mapping?)")
    return dv.audio_payload(wav, seed=seed, ref_text_used=ref_text,
                            fix_duration=fix_duration)


def describe(state: dict) -> dict:
    from importlib.metadata import version

    p = state["params"]
    return {"model": p.get("model", "F5TTS_v1_Base"), "ckpt_file": p.get("ckpt_file"),
            "f5_tts": version("f5-tts")}


if __name__ == "__main__":
    serve(load, run, describe)
