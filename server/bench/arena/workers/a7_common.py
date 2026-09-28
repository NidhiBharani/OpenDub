"""Shared A7 scoring (not a worker): embed the test clip and every bank entry (cached per path
for the job, so a trial list re-uses embeddings), cosine-score, write the test embedding.

Payload: ``{scores: [cos per bank entry], match: argmax | None, embedding_file}``. ``match`` is
None only with ``params.abstain`` (a cosine threshold) and a best score below it.
"""
from __future__ import annotations

from pathlib import Path

_CACHE: dict[str, object] = {}


def bank_of(inputs: dict) -> list[str]:
    if inputs.get("bank"):
        return list(inputs["bank"])
    return [inputs[k] for k in sorted(k for k in inputs if k.startswith("bank_"))]


def crop(path: str, max_s: float | None, tmp_dir: Path) -> str:
    if not max_s:
        return path
    import soundfile as sf

    info = sf.info(path)
    if info.duration <= max_s:
        return path
    x, sr = sf.read(path, frames=int(max_s * info.samplerate), dtype="float32")
    dst = tmp_dir / f"crop_{abs(hash(path))}.wav"
    sf.write(str(dst), x, sr)
    return str(dst)


def score_item(embed, item: dict, out: Path, params: dict) -> dict:
    import numpy as np

    def get(path: str):
        if path not in _CACHE:
            e = np.asarray(embed(path), dtype=np.float32).reshape(-1)
            _CACHE[path] = e / (np.linalg.norm(e) + 1e-9)
        return _CACHE[path]

    test = get(item["inputs"]["audio"])
    scores = [float(np.dot(test, get(b))) for b in bank_of(item["inputs"])]
    match = int(np.argmax(scores)) if scores else None
    thr = params.get("abstain")
    if thr is not None and scores and max(scores) < float(thr):
        match = None
    emb_path = out.with_suffix(".npy")
    np.save(emb_path, test)
    return {"scores": scores, "match": match, "embedding_file": str(emb_path)}
