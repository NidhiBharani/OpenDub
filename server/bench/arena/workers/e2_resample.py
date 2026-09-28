"""E2 baseline: what OpenDub does today — no bandwidth extension, the take is only resampled to
48 kHz (``app/media/ffmpeg.to_std_wav``: ffmpeg's default swresample). Server env, CPU.

params: {} (ffmpeg on PATH).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not on PATH")
    return {"ffmpeg": exe}


def run(state: dict, item: dict, out: Path) -> dict:
    wav = out.with_suffix(".wav")
    subprocess.run([state["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y", "-i",
                    item["inputs"]["audio"], "-ar", "48000", "-c:a", "pcm_s16le", str(wav)],
                   check=True)
    return {"files": {"audio": str(wav)}}


def describe(state: dict) -> dict:
    v = subprocess.run([state["ffmpeg"], "-version"], capture_output=True, text=True,
                       check=False).stdout
    return {"impl": "ffmpeg -ar 48000", "ffmpeg": v.splitlines()[0] if v else None}


if __name__ == "__main__":
    serve(load, run, describe)
