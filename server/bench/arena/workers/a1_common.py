"""Shared helpers for A1 separation workers (not a worker). numpy + soundfile, imported lazily
by the caller's env.

``finish(out, sr, mix, dialogue, background=None, music=None, effects=None, subtract=…)`` writes
the stems next to ``out`` at the model's rate and returns the A1 payload. With
``bed_by_subtraction`` (OpenDub's A1 rule) the bed is ``mix − dialogue``, so separator error
becomes quiet dialogue bleed instead of a hollow bed; otherwise the bed is ``music + effects``
(or the model's own background stem).
"""
from __future__ import annotations

from pathlib import Path


def read(path: str, sr: int | None = None, mono: bool = False):
    """(channels, samples) float32 and rate; resampled with librosa when ``sr`` differs."""
    import numpy as np
    import soundfile as sf

    x, rate = sf.read(path, always_2d=True, dtype="float32")
    x = x.T
    if sr and rate != sr:
        import librosa

        x = librosa.resample(x, orig_sr=rate, target_sr=sr)
        rate = sr
    if mono:
        x = x.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(x, dtype=np.float32), rate


def _fit(x, n):
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.shape[-1] >= n:
        return x[..., :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[-1])))


def finish(out: Path, sr: int, mix, dialogue, background=None, music=None, effects=None,
           subtract: bool = True, extra: dict | None = None) -> dict:
    import soundfile as sf

    n = mix.shape[-1]
    stems = {"dialogue": _fit(dialogue, n)}
    if music is not None:
        stems["music"] = _fit(music, n)
    if effects is not None:
        stems["effects"] = _fit(effects, n)
    if subtract or (background is None and music is None and effects is None):
        stems["background"] = mix - stems["dialogue"]
    elif background is not None:
        stems["background"] = _fit(background, n)
    else:
        stems["background"] = sum(stems[k] for k in ("music", "effects") if k in stems)
    files = {}
    for name, x in stems.items():
        path = out.with_name(f"{out.name}.{name}.wav")
        sf.write(str(path), x.T, sr, subtype="FLOAT")
        files[name] = str(path)
    return {"files": files, "sample_rate": sr, "bed_by_subtraction": bool(subtract),
            "duration_s": n / sr, **(extra or {})}
