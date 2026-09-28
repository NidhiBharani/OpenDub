"""A3/A8 worker: AudioSet taggers on a sliding window → speech / music / singing regions
(task=regions) or nonverbal events (task=events, A8).

params: family (ced), model (mispeech/ced-base …), revision, window_s (1.0), hop_s (0.5),
threshold (0.3), min_dur, task. CED (Xiaomi, Apache-2.0; HF model card checked 2026-09-28):
``AutoModelForAudioClassification`` / ``AutoFeatureExtractor`` with ``trust_remote_code=True``,
16 kHz, multi-label sigmoid over the 527 AudioSet classes in ``config.id2label``.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a3_common import group_of, to_segments


def load(params: dict, lang: str):
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    mid, rev = params.get("model", "mispeech/ced-base"), params.get("revision")
    model = AutoModelForAudioClassification.from_pretrained(mid, trust_remote_code=True,
                                                            revision=rev)
    fe = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=True, revision=rev)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {"model": model.to(dev).eval(), "fe": fe, "dev": dev, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import librosa
    import numpy as np
    import torch

    p = state["params"]
    task = p.get("task", "regions")
    y, _ = librosa.load(item["inputs"]["audio"], sr=16000, mono=True)
    win, hop = int(16000 * float(p.get("window_s", 1.0))), int(16000 * float(p.get("hop_s", 0.5)))
    labels = state["model"].config.id2label
    groups: dict[str, list[int]] = {}
    for k, name in labels.items():
        g = group_of(name, task)
        if g:
            groups.setdefault(g, []).append(int(k))
    starts = list(range(0, max(1, len(y) - win + hop), hop))
    probs = {g: [] for g in groups}
    batch = int(p.get("batch", 32))
    for b in range(0, len(starts), batch):
        chunk = [np.pad(y[s:s + win], (0, max(0, win - len(y[s:s + win])))) for s in
                 starts[b:b + batch]]
        x = state["fe"](chunk, sampling_rate=16000, return_tensors="pt").to(state["dev"])
        with torch.no_grad():
            pr = state["model"](**x).logits.sigmoid().cpu().numpy()
        for g, idx in groups.items():
            probs[g] += pr[:, idx].max(axis=1).tolist()
    times = [(s + win / 2) / 16000 for s in starts]
    segs = to_segments(times, hop / 16000, probs, threshold=float(p.get("threshold", 0.3)),
                       min_dur=float(p.get("min_dur", 0.2)))
    return {"segments": segs, "duration_s": len(y) / 16000}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model"), "revision": state["params"].get("revision")}


if __name__ == "__main__":
    serve(load, run, describe)
