"""D1/D2/D4/D5 worker: VoxCPM2 (openbmb/VoxCPM2) via the ``voxcpm`` package (2.0.3).
Verified against github.com/OpenBMB/VoxCPM README (2026-09-28).

params:
  model        openbmb/VoxCPM2; revision
  mode         clone (prompt_wav + prompt_text + reference_wav, the README's max-similarity
               setting) | design (voice description in parentheses before the text, no audio)
  cfg_value (2.0), inference_timesteps (10), load_denoiser (false)
  design       (design) description, "{gender}"/"{emotion}" filled from the item
  emotion_style (clone) prefix "(<emotion>)" from inputs.emotion — the README's instruction
               control; off by default

generate() has no language argument (the model reads the script), so the declared language list
is enforced here. 30 languages incl. en, hi, ja. 48 kHz output.
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import serve

SUPPORTED = ("ar", "my", "zh", "da", "nl", "en", "fi", "fr", "de", "el", "he", "hi", "id", "it",
             "ja", "km", "ko", "lo", "ms", "no", "pl", "pt", "ru", "es", "sw", "sv", "tl", "th",
             "tr", "vi")


def load(params: dict, lang: str):
    from voxcpm import VoxCPM

    dv.require_lang(lang, SUPPORTED, "VoxCPM2")
    kwargs = {"load_denoiser": bool(params.get("load_denoiser", False))}
    if params.get("revision"):
        from huggingface_hub import snapshot_download

        path = snapshot_download(params.get("model", "openbmb/VoxCPM2"),
                                 revision=params["revision"])
        model = VoxCPM.from_pretrained(path, **kwargs)
    else:
        model = VoxCPM.from_pretrained(params.get("model", "openbmb/VoxCPM2"), **kwargs)
    return {"model": model, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, model = state["params"], state["model"]
    text = dv.text_of(item)
    seed = dv.item_seed(item, p)
    common = {"cfg_value": float(p.get("cfg_value", 2.0)),
              "inference_timesteps": int(p.get("inference_timesteps", 10)), "seed": seed}
    mode = p.get("mode", "clone")
    if mode == "clone":
        ref, ref_text = dv.ref_of(item)
        emo = dv.emotion_of(item)
        if p.get("emotion_style") and emo:
            text = f"({emo}){text}"
        kw = {"reference_wav_path": ref}
        if ref_text:
            kw.update(prompt_wav_path=ref, prompt_text=ref_text)
        wav = model.generate(text=text, **kw, **common)
    else:
        desc = p.get("design", "A natural {gender} voice").format(
            gender=dv.gender_of(item) or "female", emotion=dv.emotion_of(item) or "calm")
        wav = model.generate(text=f"({desc}){text}", **common)
    path = dv.write_wav(dv.wav_path(out), wav, model.tts_model.sample_rate)
    return dv.audio_payload(path, seed=seed, mode=mode)


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"model": state["params"].get("model", "openbmb/VoxCPM2"),
            "revision": state["params"].get("revision"), "voxcpm": version("voxcpm")}


if __name__ == "__main__":
    serve(load, run, describe)
