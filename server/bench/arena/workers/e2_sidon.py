"""E2 worker: Sidon (Nakata & Saito, ICASSP 2026), w2v-BERT 2.0 feature cleanser + 48 kHz vocoder.

Weights: ``sarulab-speech/sidon-v0.1`` (MIT; TorchScript ``feature_extractor_cuda.pt`` /
``decoder_cuda.pt``) with the ``facebook/w2v-bert-2.0`` feature extractor. Inference follows the
authors' demo Space (``sarulab-speech/sidon_demo_beta``/app.py): peak-normalise to 0.9, 50 Hz
high-pass, resample to 16 kHz, 96 s chunks through the feature extractor, decoder → 48 kHz.
A *restorer* (resynthesises the voice), so identity drift is what the E2 ΔSIM gate is for.

params: revision (HF sha), device.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve

REPO = "sarulab-speech/sidon-v0.1"


def load(params: dict, lang: str):
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rev = params.get("revision")
    fe = torch.jit.load(hf_hub_download(REPO, f"feature_extractor_{dev}.pt", revision=rev),
                        map_location=dev).to(dev)
    dec = torch.jit.load(hf_hub_download(REPO, f"decoder_{dev}.pt", revision=rev),
                         map_location=dev).to(dev)
    pre = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
        "facebook/w2v-bert-2.0", revision=params.get("w2v_revision"))
    return {"torch": torch, "fe": fe, "dec": dec, "pre": pre, "dev": dev}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import soundfile as sf
    import torchaudio

    torch, dev = state["torch"], state["dev"]
    x, sr = sf.read(item["inputs"]["audio"], dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    peak = float(np.abs(x).max()) or 1.0
    target_n = int(48000 / sr * len(x))
    wav = torch.tensor(0.9 * x / peak, dtype=torch.float32).view(1, -1)
    wav = torchaudio.functional.highpass_biquad(wav, sr, 50)
    wav16 = torch.nn.functional.pad(torchaudio.functional.resample(wav, sr, 16000), (0, 24000))
    outs, cache = [], None
    with torch.inference_mode():
        for chunk in wav16.view(-1).split(16000 * 96):
            feats = state["pre"](torch.nn.functional.pad(chunk, (160, 160)), return_tensors="pt")
            f = state["fe"](feats["input_features"].to(dev))["last_hidden_state"]
            if cache is not None:
                f = torch.cat([cache, f], dim=1)
            outs.append(state["dec"](f.transpose(1, 2)).view(-1)[:-960])
            cache = f[:, -1:]
    y = torch.cat(outs).cpu().numpy()[:target_n] * (peak / 0.9)
    path = out.with_suffix(".wav")
    sf.write(path, np.clip(y, -1, 1), 48000, subtype="PCM_16")
    return {"files": {"audio": str(path)}}


def describe(state: dict) -> dict:
    return {"model": REPO, "device": state["dev"]}


if __name__ == "__main__":
    serve(load, run, describe)
