"""D1/D2/D5 worker: Qwen3-TTS via the official ``qwen-tts`` package (0.1.1, pins
transformers 4.57.3). Verified against github.com/QwenLM/Qwen3-TTS README (2026-09-28).

params:
  model    Qwen/Qwen3-TTS-12Hz-1.7B-Base | -0.6B-Base (clone) | -1.7B-VoiceDesign (design) |
           -1.7B-CustomVoice / -0.6B-CustomVoice (preset speakers)
  revision HF commit
  mode     clone | design | custom
  dtype    bfloat16; attn: sdpa (no FA3 on sm_120; FA2 optional)
  speakers (custom) {lang: {female: Ono_Anna, male: …}} or one name — preset timbres: Vivian,
           Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
  instruct (design/custom) voice description; "{gender}" and "{emotion}" are filled from the item
  emotion_instruct  (clone) not supported by Base: ignored

Languages (card): zh, en, ja, ko, de, fr, ru, pt, es, it — no Hindi (→ Unsupported).
generate_* return (wavs, sr); 24 kHz.
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve

SUPPORTED = ("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it")


def load(params: dict, lang: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    dv.require_lang(lang, SUPPORTED, "Qwen3-TTS")
    model = Qwen3TTSModel.from_pretrained(
        params.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"), revision=params.get("revision"),
        device_map=params.get("device", "cuda:0"),
        dtype=getattr(torch, params.get("dtype", "bfloat16")),
        attn_implementation=params.get("attn", "sdpa"))
    return {"model": model, "params": params, "lang": lang, "language": dv.LANG_NAMES[lang]}


def _instruct(p: dict, item: dict) -> str:
    tmpl = p.get("instruct", "")
    return tmpl.format(gender=dv.gender_of(item) or "female",
                       emotion=dv.emotion_of(item) or "neutral") if tmpl else ""


def run(state: dict, item: dict, out: Path) -> dict:
    p, model = state["params"], state["model"]
    text = dv.text_of(item)
    seed = dv.item_seed(item, p)
    dv.seed_all(seed)
    mode = p.get("mode", "clone")
    gen = dict(p.get("generate_kwargs") or {})
    if mode == "clone":
        ref, ref_text = dv.ref_of(item)
        if not ref_text:
            raise Unsupported("Qwen3-TTS Base cloning needs the reference transcript")
        wavs, sr = model.generate_voice_clone(text=text, language=state["language"],
                                              ref_audio=ref, ref_text=ref_text, **gen)
    elif mode == "design":
        wavs, sr = model.generate_voice_design(text=text, language=state["language"],
                                               instruct=_instruct(p, item) or
                                               "A clear, natural {gender} voice.", **gen)
    elif mode == "custom":
        speaker = dv.pick_voice(p.get("speakers", "Vivian"), state["lang"], item)
        wavs, sr = model.generate_custom_voice(text=text, language=state["language"],
                                               speaker=speaker, instruct=_instruct(p, item),
                                               **gen)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    wav = dv.write_wav(dv.wav_path(out), wavs[0] if isinstance(wavs, list) else wavs, sr)
    return dv.audio_payload(wav, seed=seed, mode=mode)


def describe(state: dict) -> dict:
    from importlib.metadata import version

    p = state["params"]
    return {"model": p.get("model"), "revision": p.get("revision"), "mode": p.get("mode"),
            "qwen_tts": version("qwen-tts")}


if __name__ == "__main__":
    serve(load, run, describe)
