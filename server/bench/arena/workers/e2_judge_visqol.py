"""Judge worker (E2/E5): ViSQOL v3 via ``visqol-python`` (Apache-2.0, pure-Python port that
passes the C++ implementation's conformance tests). Audio mode at 48 kHz for bandwidth work.

item.inputs: {audio: degraded/processed, reference: clean}. Both are resampled to ``sr`` and
trimmed to the same length. Returns {"metrics": {"visqol": MOS-LQO}}.
params: mode (audio | speech), sr (48000 audio / 16000 speech).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    from visqol import VisqolApi

    api = VisqolApi()
    api.create(mode=params.get("mode", "audio"))
    return {"api": api, "sr": int(params.get("sr", 48000))}


def _prep(path: str, sr: int):
    import math

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    x, fs = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        g = math.gcd(int(fs), sr)
        x = resample_poly(x, sr // g, int(fs) // g)
    return np.asarray(x)


def run(state: dict, item: dict, out: Path) -> dict:
    import soundfile as sf

    sr = state["sr"]
    ref, deg = _prep(item["inputs"]["reference"], sr), _prep(item["inputs"]["audio"], sr)
    n = min(len(ref), len(deg))
    with tempfile.TemporaryDirectory() as td:
        r, d = Path(td) / "ref.wav", Path(td) / "deg.wav"
        sf.write(r, ref[:n], sr, subtype="PCM_16")
        sf.write(d, deg[:n], sr, subtype="PCM_16")
        res = state["api"].measure(str(r), str(d))
    return {"metrics": {"visqol": float(res.moslqo)}}


if __name__ == "__main__":
    serve(load, run)
