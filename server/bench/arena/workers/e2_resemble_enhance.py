"""E2 worker: Resemble Enhance (MIT, https://github.com/resemble-ai/resemble-enhance), a
denoiser + latent-CFM enhancer trained on 44.1 kHz speech; output 44.1 kHz (judges resample).

API as in the repo's app.py: ``enhance(dwav, sr, device, nfe, solver, lambd, tau)`` (``lambd``
0.9 = denoise before enhancing, 0.1 = not). params: nfe (64), solver (midpoint), lambd (0.1),
tau (0.5), denoise_only (false).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    import torch
    from resemble_enhance.enhancer.inference import denoise, enhance

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {"denoise": denoise, "enhance": enhance, "dev": dev, "p": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import soundfile as sf
    import torchaudio

    p = state["p"]
    dwav, sr = torchaudio.load(item["inputs"]["audio"])
    dwav = dwav.mean(dim=0)
    if p.get("denoise_only"):
        wav, new_sr = state["denoise"](dwav, sr, state["dev"])
    else:
        wav, new_sr = state["enhance"](dwav, sr, state["dev"], nfe=int(p.get("nfe", 64)),
                                       solver=p.get("solver", "midpoint"),
                                       lambd=float(p.get("lambd", 0.1)),
                                       tau=float(p.get("tau", 0.5)))
    path = out.with_suffix(".wav")
    sf.write(path, wav.cpu().numpy(), int(new_sr), subtype="PCM_16")
    return {"files": {"audio": str(path)}}


def describe(state: dict) -> dict:
    return {"model": "resemble-enhance", "device": state["dev"]}


if __name__ == "__main__":
    serve(load, run, describe)
