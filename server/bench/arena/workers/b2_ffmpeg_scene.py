"""B2 baseline worker: ffmpeg's ``scene`` frame-difference score, exactly what OpenDub does today
(``app/media/ffmpeg.py: detect_scene_changes`` — ``scale=320:-2,select='gt(scene,T)'``, used by
``app/pipeline/analysis.py`` at T = 0.3). Server env, no model; needs the ``ffmpeg`` tool.

params: threshold (0.3).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from _sdk import serve
from b_common import probe

PTS = re.compile(r"pts_time:([0-9.]+)")
SCORE = re.compile(r"lavfi\.scene_score=([0-9.]+)")


def load(params: dict, lang: str) -> dict:
    return {"threshold": float(params.get("threshold", 0.3))}


def run(state: dict, item: dict, out: Path) -> dict:
    video = item["inputs"]["video"]
    info = probe(video)
    meta = out.parent / f"{out.name}.scenes.txt"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", video, "-an", "-vf",
         f"scale=320:-2,select='gt(scene,{state['threshold']:g})',metadata=print:file={meta}",
         "-f", "null", "-"], check=True)
    transitions, t = [], None
    for line in (meta.read_text() if meta.exists() else "").splitlines():
        if m := PTS.search(line):
            t = float(m.group(1))
        elif (m := SCORE.search(line)) and t is not None:
            transitions.append({"start": round(t, 3), "end": round(t, 3), "type": "cut",
                                "score": min(1.0, float(m.group(1)))})
            t = None
    return {"transitions": transitions, "fps": info["fps"], "duration_s": info["duration_s"]}


if __name__ == "__main__":
    serve(load, run)
