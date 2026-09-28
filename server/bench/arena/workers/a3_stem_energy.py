"""A3 worker: cross-stem energy heuristic (OpenDub's builtin A3 plan): separate with Demucs, then
speech where the dialogue stem is active, music where the bed is active. Cannot tell singing
from speech (both land in the vocal stem) — that is what the model candidates are for.

params: model (demucs name), frame_s (0.1), speech_db / music_db (activity thresholds relative
to each stem's 95th-percentile frame level), min_dur. Server env.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a1_common import read
from a3_common import to_segments


def load(params: dict, lang: str):
    import torch
    from demucs.pretrained import get_model

    m = get_model(params.get("model", "htdemucs"))
    m.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return {"model": m, "params": params}


def _frame_db(x, n):
    import numpy as np

    x = x.mean(axis=0)
    frames = x[: len(x) // n * n].reshape(-1, n)
    return 10 * np.log10((frames ** 2).mean(axis=1) + 1e-10)


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import torch
    from demucs.apply import apply_model

    m, p = state["model"], state["params"]
    mix, sr = read(item["inputs"]["audio"], m.samplerate)
    if mix.shape[0] == 1:
        mix = np.repeat(mix, 2, axis=0)
    x = torch.from_numpy(mix[:2]).to(next(m.parameters()).device)
    with torch.no_grad():
        est = apply_model(m, x[None], shifts=1, overlap=0.25, progress=False)[0].cpu().numpy()
    voc = est[list(m.sources).index("vocals")]
    bed = mix[:2] - voc
    hop = float(p.get("frame_s", 0.1))
    n = int(sr * hop)
    vdb, bdb = _frame_db(voc, n), _frame_db(bed, n)
    ref_v, ref_b = np.percentile(vdb, 95), np.percentile(bdb, 95)
    speech = (vdb > ref_v - float(p.get("speech_db", 25))) & (vdb > bdb - 6)
    music = bdb > ref_b - float(p.get("music_db", 20))
    times = [(k + 0.5) * hop for k in range(len(vdb))]
    segs = to_segments(times, hop, {"speech": speech.astype(float).tolist(),
                                    "music": music.astype(float).tolist()},
                       threshold=0.5, min_dur=float(p.get("min_dur", 0.3)))
    return {"segments": segs, "duration_s": mix.shape[-1] / sr}


if __name__ == "__main__":
    serve(load, run)
