"""E2 worker: FLowHigh (Yun et al., ICASSP 2025), single-step flow-matching audio super-resolution.

Uses the packaged fork https://github.com/resemble-ai/flowhigh (MIT), whose
``FlowHighSR.from_pretrained`` downloads ``ResembleAI/FlowHigh`` from the HF Hub — note this is the
``FLowHigh_basic_400k`` (basic CFM) checkpoint with a BigVGAN 48 kHz vocoder, not the paper's main
``indep_adaptive`` model (that one is only on the authors' Google Drive). Input is upsampled with
``scipy.resample_poly`` inside ``generate`` (the paper's protocol); output is 48 kHz. The model
peak-normalises its conditioning, so the output is rescaled to the input's peak.

params: timestep (ODE steps, default 1), ode_method (midpoint).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    import torch
    from flowhigh import FlowHighSR

    model = FlowHighSR.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    return {"model": model, "params": params, "torch": torch}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(item["inputs"]["audio"], dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    y = state["model"].generate(x, sr, 48000, timestep=int(state["params"].get("timestep", 1)))
    y = y.squeeze().float().cpu().numpy()
    peak_in, peak_out = float(np.max(np.abs(x))), float(np.max(np.abs(y)))
    if peak_out > 0:
        y = y * (peak_in / peak_out)
    path = out.with_suffix(".wav")
    sf.write(path, y, 48000, subtype="PCM_16")
    return {"files": {"audio": str(path)}}


def describe(state: dict) -> dict:
    return {"model": "ResembleAI/FlowHigh (FLowHigh_basic_400k + BigVGAN 48k)"}


if __name__ == "__main__":
    serve(load, run, describe)
