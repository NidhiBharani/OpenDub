"""A5 worker: proportional word interpolation (OpenDub's builtin A5 fallback) — distribute the
given text's words over the voiced span of the clip in proportion to their character count.
The floor every aligner must beat. Server env (numpy + soundfile), CPU.
params: floor_db (activity threshold below the 95th-percentile frame level, default 30).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a5_common import _norm, tokenize


def load(params: dict, lang: str):
    return {"params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(item["inputs"]["audio"], always_2d=True, dtype="float32")
    x = x.mean(axis=1)
    hop = int(sr * 0.01)
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-10)
    active = np.flatnonzero(db > np.percentile(db, 95) - float(state["params"].get("floor_db",
                                                                                   30)))
    t0 = active[0] * 0.01 if len(active) else 0.0
    t1 = (active[-1] + 1) * 0.01 if len(active) else len(x) / sr
    toks = tokenize(item["inputs"]["text"], state["lang"])
    weights = [max(1, len(_norm(t))) for t in toks]
    total, t, words = float(sum(weights)), t0, []
    for tok, w in zip(toks, weights):
        d = (t1 - t0) * w / total
        words.append({"start": round(t, 4), "end": round(t + d, 4), "word": tok})
        t += d
    return {"words": words}


if __name__ == "__main__":
    serve(load, run)
