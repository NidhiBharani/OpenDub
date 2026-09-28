"""E2 worker: AudioSR (Liu et al., MIT, ``pip install audiosr``), latent-diffusion versatile
super-resolution to 48 kHz, ``speech`` checkpoint. Slow (DDIM steps) and the literature documents
excess sibilance / hallucination: benchmarked for completeness, gated by ΔCER/ΔSIM.

params: model_name (speech), ddim_steps (50), guidance_scale (3.5), seed (42).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    from audiosr import build_model, super_resolution

    model = build_model(model_name=params.get("model_name", "speech"), device="auto")
    return {"model": model, "sr_fn": super_resolution, "p": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import soundfile as sf

    p = state["p"]
    y = state["sr_fn"](state["model"], item["inputs"]["audio"], seed=int(p.get("seed", 42)),
                       guidance_scale=float(p.get("guidance_scale", 3.5)),
                       ddim_steps=int(p.get("ddim_steps", 50)))
    y = np.asarray(y).squeeze()
    x, sr = sf.read(item["inputs"]["audio"], dtype="float32", always_2d=True)
    y = y[: round(len(x) * 48000 / sr)]  # AudioSR pads to its latent frame grid
    path = out.with_suffix(".wav")
    sf.write(path, y, 48000, subtype="PCM_16")
    return {"files": {"audio": str(path)}}


def describe(state: dict) -> dict:
    return {"model": f"audiosr/{state['p'].get('model_name', 'speech')}"}


if __name__ == "__main__":
    serve(load, run, describe)
