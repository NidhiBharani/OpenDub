"""A2 worker: Resemble Enhance (MIT; ``resemble-enhance`` 0.0.1, HF ResembleAI/resemble-enhance).
``resemble_enhance.enhancer.inference.denoise(wav_1d, sr, device)`` and ``enhance(wav_1d, sr,
device, nfe, solver, lambd, tau)`` → (wav, 44100). README checked 2026-09-28.
params: mode (denoise | enhance), nfe (32), solver (midpoint), lambd (0.5), tau (0.5).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    from resemble_enhance.enhancer import inference

    return {"inf": inference, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import soundfile as sf
    import torch

    p, inf = state["params"], state["inf"]
    x, sr = sf.read(item["inputs"]["audio"], always_2d=True, dtype="float32")
    wav = torch.from_numpy(x.mean(axis=1))
    if p.get("mode", "enhance") == "denoise":
        y, osr = inf.denoise(wav, sr, "cuda")
    else:
        y, osr = inf.enhance(wav, sr, "cuda", nfe=int(p.get("nfe", 32)),
                             solver=p.get("solver", "midpoint"), lambd=float(p.get("lambd", 0.5)),
                             tau=float(p.get("tau", 0.5)))
    dst = out.with_suffix(".wav")
    sf.write(str(dst), y.cpu().numpy(), osr)
    return {"files": {"audio": str(dst)}, "sample_rate": osr}


if __name__ == "__main__":
    serve(load, run)
