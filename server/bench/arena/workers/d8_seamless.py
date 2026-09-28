"""D8 worker (evaluate-only comparator): Meta Seamless direct speech-to-speech translation.

mode m4t_v2 (default): facebook/seamless-m4t-v2-large (CC-BY-NC-4.0) through transformers
  ``SeamlessM4Tv2Model.generate(**processor(audios=…, sampling_rate=16000), tgt_lang=…,
  return_intermediate_token_ids=True)`` → ``.waveform`` (16 kHz) and ``.sequences`` (text).
  Speech output covers eng, hin and jpn among its 36 targets; source speech ~100 languages.
mode expressive: SeamlessExpressive (gated, research licence; seamless_communication package,
  targets eng/fra/deu/ita/spa/cmn only) — no ja/hi pair, so registered disabled.

params: model, revision, speaker_id (m4t vocoder speaker, default 0), tgt_lang override.
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve

ISO3 = {"en": "eng", "hi": "hin", "ja": "jpn", "zh": "cmn", "ko": "kor", "de": "deu",
        "fr": "fra", "es": "spa", "it": "ita", "pt": "por", "ru": "rus", "ta": "tam",
        "te": "tel", "bn": "ben"}
SPEECH_TGTS = {"arb", "ben", "cat", "ces", "cmn", "cym", "dan", "deu", "eng", "est", "fin", "fra",
               "hin", "ind", "ita", "jpn", "kor", "mlt", "nld", "pes", "pol", "por", "ron", "rus",
               "slk", "spa", "swe", "swh", "tel", "tgl", "tha", "tur", "ukr", "urd", "uzn", "vie"}


def load(params: dict, lang: str):
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2Model

    if params.get("mode", "m4t_v2") != "m4t_v2":
        raise RuntimeError("SeamlessExpressive needs the seamless_communication stack (disabled)")
    tgt = params.get("tgt_lang") or ISO3.get(lang)
    if tgt not in SPEECH_TGTS:
        raise RuntimeError(f"SeamlessM4T v2 has no speech output for {lang!r}")
    name = params.get("model", "facebook/seamless-m4t-v2-large")
    proc = AutoProcessor.from_pretrained(name, revision=params.get("revision"))
    model = SeamlessM4Tv2Model.from_pretrained(name, revision=params.get("revision"),
                                               torch_dtype=torch.float16).to("cuda").eval()
    return {"proc": proc, "model": model, "tgt": tgt, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    import librosa
    import torch

    src = (item.get("inputs") or {}).get("audio")
    if not src:
        raise Unsupported("S2ST item has no inputs.audio")
    audio, _ = librosa.load(src, sr=16000)
    inputs = state["proc"](audios=audio, sampling_rate=16000, return_tensors="pt").to("cuda")
    inputs["input_features"] = inputs["input_features"].half()
    with torch.no_grad():
        res = state["model"].generate(**inputs, tgt_lang=state["tgt"],
                                      speaker_id=int(state["params"].get("speaker_id", 0)),
                                      return_intermediate_token_ids=True)
    text = state["proc"].decode(res.sequences[0].tolist(), skip_special_tokens=True)
    wav = dv.write_wav(dv.wav_path(out), res.waveform[0], 16000)
    return {**dv.audio_payload(wav), "text": text, "src_lang": dv.src_lang(item)}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model", "facebook/seamless-m4t-v2-large"),
            "tgt_lang": state["tgt"]}


if __name__ == "__main__":
    serve(load, run, describe)
