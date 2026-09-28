"""A6 worker: OpenDub's default ``diarization.single_speaker`` — every voiced region is one
speaker. Energy VAD (10 ms frames, threshold ``floor_db`` below the 95th percentile, gaps under
``min_gap`` bridged). The DER floor. Server env, CPU."""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    return {"params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import soundfile as sf

    p = state["params"]
    x, sr = sf.read(item["inputs"]["audio"], always_2d=True, dtype="float32")
    x = x.mean(axis=1)
    hop = int(sr * 0.01)
    fr = x[: len(x) // hop * hop].reshape(-1, hop)
    db = 10 * np.log10((fr ** 2).mean(axis=1) + 1e-10)
    act = db > np.percentile(db, 95) - float(p.get("floor_db", 35))
    turns, start, gap = [], None, int(float(p.get("min_gap", 0.3)) / 0.01)
    last = -10**9
    for k, a in enumerate(act):
        if a:
            if start is None:
                start = k
            elif k - last > gap:
                turns.append({"start": start * 0.01, "end": (last + 1) * 0.01, "speaker": "S0"})
                start = k
            last = k
    if start is not None:
        turns.append({"start": start * 0.01, "end": (last + 1) * 0.01, "speaker": "S0"})
    return {"turns": turns}


if __name__ == "__main__":
    serve(load, run)
