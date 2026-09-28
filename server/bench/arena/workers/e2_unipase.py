"""E2 worker: UniPASE (Rong et al., IEEE TASLP 2026), universal speech enhancement/restoration
(DeWavLM-Omni + adapter + vocoder + 48 kHz PostNet). Backbone of the URGENT 2026 objective winner.

Code: https://github.com/Xiaobin-Rong/unipase (MIT) cloned to ``~/.opendub/src/unipase``;
weights ``Xiaobin-Rong/unipase`` on the HF Hub. Mirrors ``inference/inference.py``
(``model(x, sr_in, sr_out=48000, enable_plc)``, output peak-matched to the input). A generative
restorer: gated by E2's ΔCER/ΔSIM (the paper reports CER and SpkSim regressions vs a predictive
baseline).

params: repo, revision (HF sha), enable_plc (true), device.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve

HF_REPO = "Xiaobin-Rong/unipase"
FILES = ("DeWavLM-Omni.pt", "Adapter.pt", "Vocoder_DWO-L1.pt", "PostNet.pt", "config.json")


def load(params: dict, lang: str):
    import torch
    from huggingface_hub import hf_hub_download

    repo = Path(params.get("repo", "~/.opendub/src/unipase")).expanduser()
    sys.path.insert(0, str(repo))
    from models.unipase import UniPASE

    got = {f: hf_hub_download(HF_REPO, f, revision=params.get("revision")) for f in FILES}
    dev = torch.device(params.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    model = UniPASE(dewavlm_ckpt_path=got["DeWavLM-Omni.pt"], adapter_ckpt_path=got["Adapter.pt"],
                    vocoder_ckpt_path=got["Vocoder_DWO-L1.pt"],
                    postnet_ckpt_path=got["PostNet.pt"]).to(dev).eval()
    return {"torch": torch, "model": model, "dev": dev, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import soundfile as sf

    torch = state["torch"]
    x, sr = sf.read(item["inputs"]["audio"], dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    with torch.inference_mode():
        y = state["model"](torch.FloatTensor(x).unsqueeze(0).to(state["dev"]), sr_in=sr,
                           sr_out=48000,
                           enable_plc=bool(state["params"].get("enable_plc", True)))
    y = y.cpu().numpy().squeeze()
    y = y / (np.max(np.abs(y)) + 1e-8) * float(np.max(np.abs(x)))
    path = out.with_suffix(".wav")
    sf.write(path, y, 48000, subtype="PCM_16")
    return {"files": {"audio": str(path)}}


def describe(state: dict) -> dict:
    return {"model": HF_REPO, "revision": state["params"].get("revision"),
            "device": str(state["dev"])}


if __name__ == "__main__":
    serve(load, run, describe)
