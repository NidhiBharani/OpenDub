"""A7 worker: microsoft/wavlm-base-plus-sv (transformers ``WavLMForXVector`` +
``Wav2Vec2FeatureExtractor``; ``.embeddings`` → cosine). 16 kHz. Server env (transformers is
already there). Note: also the default G2 similarity judge, so A2/D judges and this candidate
share a lineage — rank A7 with that in mind. params: model, revision.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a7_common import score_item


def load(params: dict, lang: str):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    mid = params.get("model", "microsoft/wavlm-base-plus-sv")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(mid, revision=params.get("revision"))
    model = WavLMForXVector.from_pretrained(mid, revision=params.get("revision"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {"fe": fe, "model": model.to(dev).eval(), "dev": dev, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import librosa
    import torch

    def embed(path: str):
        y, _ = librosa.load(path, sr=16000, mono=True)
        x = state["fe"](y, sampling_rate=16000, return_tensors="pt").to(state["dev"])
        with torch.no_grad():
            return state["model"](**x).embeddings[0].cpu().numpy()

    return score_item(embed, item, out, state["params"])


if __name__ == "__main__":
    serve(load, run)
