"""E4 baseline: the app's mix stage, ``app.pipeline.audio.mix_tracks`` (the ``loudness.static``
builtin): dub vocals get one static gain to the separated original dialogue's integrated loudness
(else −18 LUFS), are amixed with the bed at unity, and the premix is moved by one static gain to
the delivery's integrated target (programme-gated) through a −1.5 dBFS *sample-peak* alimiter.
It measures programme loudness even for dialogue-gated deliveries and uses the same −1.5 ceiling
for −1 and −2 dBTP presets: exactly what the arena should expose.

params: ducking_mix (0 = app default; >0 = the app's optional sidechain duck of the bed),
background_gain_db (0).
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
    from app.pipeline import audio

    return {"audio": audio, "p": params}


def run(state: dict, item: dict, out: Path) -> dict:
    inp, p = item["inputs"], state["p"]
    ref = inp.get("reference_dialogue")
    wav = out.with_suffix(".wav")
    levels = asyncio.run(state["audio"].mix_tracks(
        Path(inp["dialogue"]), Path(inp["background"]), wav,
        reference_vocals=Path(ref) if ref else None,
        target_lufs=float(inp["delivery"]["target_lufs"]),
        background_gain_db=float(p.get("background_gain_db", 0.0)),
        ducking_mix=float(p.get("ducking_mix", 0.0))))
    return {"files": {"audio": str(wav)}, "params": levels}


def describe(state: dict) -> dict:
    return {"impl": "app.pipeline.audio.mix_tracks"}


if __name__ == "__main__":
    serve(load, run, describe)
