"""Shared helpers for the judge workers (``judge_*.py``, ``g*_*.py``); not a worker itself.

Stdlib only at import time. numpy / soundfile / torch are imported inside the functions, so this
module can be imported by any worker env (every judge env installs numpy + soundfile).
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path


def load_mono(path: str | Path, sr: int):
    """Decode any audio (or the audio track of a video) to mono float32 at ``sr``.

    Uses soundfile when the file is already at ``sr``; otherwise ffmpeg (one resampler for every
    judge, so no judge's score depends on which resampler its env happened to have), falling back
    to torchaudio's resampler when ffmpeg is missing.
    """
    import numpy as np

    path = str(path)
    mono = rate = None
    try:
        import soundfile as sf

        data, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
    except Exception:  # noqa: BLE001 - containers soundfile cannot read (mp4, m4a) go to ffmpeg
        mono = None
    if mono is not None and rate == sr:
        return np.ascontiguousarray(mono, dtype=np.float32)
    if shutil.which("ffmpeg"):
        raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1",
                              "-ar", str(sr), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32).copy()
    if mono is None:
        raise RuntimeError(f"cannot decode {path}: soundfile failed and ffmpeg is not on PATH")
    import torch
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(mono), int(rate), sr).numpy().astype(np.float32)


def write_wav(path: str | Path, samples, sr: int) -> str:
    import numpy as np
    import soundfile as sf

    sf.write(str(path), np.asarray(samples, dtype=np.float32), sr, subtype="PCM_16")
    return str(path)


def cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape or not a.size:
        raise ValueError(f"embedding shapes differ: {a.shape} vs {b.shape}")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not denom or not math.isfinite(denom):
        raise ValueError("zero or non-finite embedding")
    return max(-1.0, min(1.0, float(a @ b) / denom))


def as_list(value) -> list[str]:
    """A reference may be one path or a list of paths (a reference set)."""
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]


class RefCache:
    """Per-job cache of reference embeddings (the same speaker/source clip is compared against
    many outputs). Keyed by path + size + mtime so an edited file is re-embedded."""

    def __init__(self, fn):
        self.fn = fn
        self.store: dict[tuple[str, int, int], object] = {}

    def __call__(self, path: str):
        st = Path(path).stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self.store:
            self.store[key] = self.fn(path)
        return self.store[key]


def mean_embedding(paths: list[str], embed):
    """Average of L2-normalised embeddings of a reference set."""
    import numpy as np

    vecs = []
    for p in paths:
        v = np.asarray(embed(p), dtype=np.float64).ravel()
        n = np.linalg.norm(v)
        vecs.append(v / n if n else v)
    return np.mean(vecs, axis=0)


def device_of(params: dict) -> str:
    dev = params.get("device")
    if dev:
        return dev
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def duration_s(path: str | Path) -> float | None:
    """Media duration via ffprobe (video or audio); None when unknown."""
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "default=nw=1:nk=1", str(path)], capture_output=True, text=True,
                         check=False).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None
