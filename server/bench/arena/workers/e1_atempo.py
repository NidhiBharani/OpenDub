"""E1 baseline: the app's own timing fit, ``app.pipeline.audio.fit_to_duration`` (ffmpeg atempo).

It trims outer silence, never slows a take down (short takes are padded at the end) and speeds up
at most ``max_rate`` (1.15 in the app). Beyond that the app raises and regenerates the line; the
arena instead keeps the best effort at the clamp (whole take at ``max_rate``, no trim) and reports
``overflow: true`` so the item is scored (as an overflow) rather than dropped from the pairing.

params: max_rate (1.15), min_rate (0.85; unused by the app's fit, recorded for completeness).
Runs in the ``server`` env: imports the app from ``server/`` inside ``load``.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from _sdk import serve

SERVER_DIR = Path(__file__).resolve().parents[3]


def load(params: dict, lang: str):
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    from app.media import ffmpeg
    from app.pipeline import audio

    return {"audio": audio, "ffmpeg": ffmpeg, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    audio, ffmpeg, p = state["audio"], state["ffmpeg"], state["params"]
    take = Path(item["inputs"]["audio"])
    target = float(item["inputs"]["target_s"])
    wav = out.with_suffix(".wav")
    max_rate = float(p.get("max_rate", 1.15))
    overflow = False
    try:
        rate = asyncio.run(audio.fit_to_duration(take, wav, target,
                                                 min_rate=float(p.get("min_rate", 0.85)),
                                                 max_rate=max_rate))
    except audio.TimingOverflowError:
        overflow = True
        tmp = out.with_suffix(".std.tmp.wav")
        try:
            asyncio.run(ffmpeg.to_std_wav(take, tmp))
            asyncio.run(ffmpeg.atempo(tmp, wav, max_rate))
        finally:
            tmp.unlink(missing_ok=True)
        rate = max_rate
    return {"files": {"audio": str(wav)}, "params": {"rate": rate, "overflow": overflow,
                                                     "max_rate": max_rate}}


def describe(state: dict) -> dict:
    return {"impl": "app.pipeline.audio.fit_to_duration (ffmpeg atempo)"}


if __name__ == "__main__":
    serve(load, run, describe)
