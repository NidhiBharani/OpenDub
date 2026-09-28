"""D1/D3/D4/D5 worker: OmniVoice (k2-fsa, 0.6B, 600+ languages; code Apache-2.0, weights
CC-BY-NC per the research artifacts) via ``omnivoice`` (0.2.1). README API (2026-09-28):
``OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)`` and
``model.generate(text=, ref_audio=, ref_text=, num_step=32, speed=, duration=, instruct=)`` →
list of np.ndarray at 24 kHz.

params:
  model, revision, dtype (float16), num_step (32)
  mode       clone | design (instruct=<attributes>, no reference)
  instruct   (design) e.g. "{gender}, clear voice" ({gender} from item meta)
  use_target_duration  pass ``duration=inputs.target_s`` (fixed output length) for D3
  pass_language        also pass ``language=<code>`` (not documented; off unless verified)
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import serve


def load(params: dict, lang: str):
    import torch
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(params.get("model", "k2-fsa/OmniVoice"),
                                      revision=params.get("revision"),
                                      device_map=params.get("device", "cuda:0"),
                                      dtype=getattr(torch, params.get("dtype", "float16")))
    return {"model": model, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    seed = dv.item_seed(item, p)
    dv.seed_all(seed)
    kw: dict = {"text": dv.text_of(item), "num_step": int(p.get("num_step", 32))}
    if p.get("mode", "clone") == "clone":
        ref, ref_text = dv.ref_of(item)
        kw["ref_audio"] = ref
        if ref_text:
            kw["ref_text"] = ref_text
    else:
        kw["instruct"] = p.get("instruct", "{gender}").format(gender=dv.gender_of(item) or
                                                              "female")
    tgt = dv.target_s(item)
    if tgt and p.get("use_target_duration", True):
        kw["duration"] = tgt
    if p.get("pass_language"):
        kw["language"] = state["lang"]
    audio = state["model"].generate(**kw)
    wav = dv.write_wav(dv.wav_path(out), audio[0] if isinstance(audio, list) else audio, 24000)
    return dv.audio_payload(wav, seed=seed)


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"model": state["params"].get("model", "k2-fsa/OmniVoice"),
            "omnivoice": version("omnivoice")}


if __name__ == "__main__":
    serve(load, run, describe)
