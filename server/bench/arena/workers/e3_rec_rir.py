"""E3 worker: Rec-RIR (Audio-WestlakeU, arXiv 2509.15628; MIT, checkpoint ``ckpt/epoch35.tar``
in the repo) — blind monaural RIR identification from reverberant speech; the estimated RIR is
convolved with the dry line, RMS-matched to the reference.

The repo only ships a directory CLI (``inference.py -c config/Rec-RIR.toml --ckpt … -i in -o out``,
writing ``out/rir/<name>.wav``, peak-normalised). To load the model once per job the worker runs
that CLI a single time in ``load`` over every item's reference (read from the job file), then
``run`` only convolves. Consequence: per-item seconds (and so rtfx) exclude the estimation; the
job's ``load_seconds`` in worker.json holds it.

params: repo (~/.opendub/src/rec-rir), config (config/Rec-RIR.toml), ckpt (ckpt/epoch35.tar),
device (cuda:0), sr (16000: the rate references are converted to before estimation).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from _sdk import serve


def _job_items() -> list[dict]:
    argv = sys.argv
    job = argv[argv.index("--job") + 1] if "--job" in argv else None
    return json.loads(Path(job).read_text())["items"] if job else []


def load(params: dict, lang: str):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    repo = Path(params.get("repo", "~/.opendub/src/rec-rir")).expanduser()
    sr = int(params.get("sr", 16000))
    work = Path(tempfile.mkdtemp(prefix="rec-rir-"))
    (work / "in").mkdir()
    names: dict[str, str] = {}
    for k, it in enumerate(_job_items()):
        ref = it["inputs"]["reference"]
        x, fs = sf.read(ref, dtype="float64", always_2d=True)
        x = x.mean(axis=1)
        if fs != sr:
            g = math.gcd(int(fs), sr)
            x = resample_poly(x, sr // g, int(fs) // g)
        name = f"r{k:05d}.wav"
        sf.write(work / "in" / name, x, sr, subtype="PCM_16")
        names[it["id"]] = name
    if names:
        subprocess.run([sys.executable, "inference.py", "-c", params.get("config",
                        "config/Rec-RIR.toml"), "--ckpt", params.get("ckpt", "ckpt/epoch35.tar"),
                        "-i", str(work / "in"), "-o", str(work / "out"), "-d",
                        params.get("device", "cuda:0")], cwd=repo, check=True)
    return {"np": np, "sf": sf, "names": names, "work": work, "sr": sr}


def run(state: dict, item: dict, out: Path) -> dict:
    from scipy.signal import fftconvolve, resample_poly

    np, sf = state["np"], state["sf"]
    rir_path = state["work"] / "out" / "rir" / state["names"].get(item["id"], "missing.wav")
    if not rir_path.exists():
        raise RuntimeError(f"Rec-RIR produced no RIR for {item['id']}")
    h, hsr = sf.read(rir_path, dtype="float64", always_2d=True)
    h = h.mean(axis=1)
    x, sr = sf.read(item["inputs"]["audio"], dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if hsr != sr:
        g = math.gcd(int(hsr), int(sr))
        h = resample_poly(h, sr // g, hsr // g)
    ref, _ = sf.read(item["inputs"]["reference"], dtype="float64", always_2d=True)
    y = fftconvolve(x, h)[: len(x) + int(0.3 * sr)]
    y *= math.sqrt(float(np.mean(ref ** 2)) / max(float(np.mean(y ** 2)), 1e-20))
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y *= 0.99 / peak
    wav, rir = out.with_suffix(".wav"), out.with_suffix(".rir.wav")
    sf.write(wav, y, sr, subtype="PCM_16")
    sf.write(rir, h, sr, subtype="FLOAT")
    return {"files": {"audio": str(wav), "rir": str(rir)}}


def describe(state: dict) -> dict:
    return {"model": "Rec-RIR epoch35", "work_dir": str(state["work"])}


if __name__ == "__main__":
    serve(load, run, describe)
