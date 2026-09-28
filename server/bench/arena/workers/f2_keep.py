"""F2 worker: KEEP (jnjaby/KEEP, Kalman-inspired video face SR; NTU S-Lab License 1.0: NC).

``inference_keep.py -i <video> --save_video -s 1`` (no background upsampler). params:
face_upsample (false), mask_only (true), repo_dir (default ~/.opendub/src/KEEP). Weights: the
repo's v1.0.0 GitHub release (fetched by the script / env recipe).
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve
from f_common import cleanup, newest, repo_dir, video_payload, workdir
from f_common import run as sh


def load(params: dict, lang: str):
    return {"repo": repo_dir(params, "KEEP"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    work = workdir(out)
    cmd = [sys.executable, "inference_keep.py", f"-i={item['inputs']['video']}",
           f"-o={work / 'res'}", "--save_video", "-s=1"]
    if p.get("face_upsample"):
        cmd.append("--face_upsample")
    sh(cmd, cwd=state["repo"])
    produced = newest(work / "res")
    from f_frames import restore_finish

    audio = item["inputs"].get("audio")
    final = restore_finish(produced, item, out.with_suffix(".mp4"),
                           mask_only=bool(p.get("mask_only", True)))
    payload = video_payload(out, final, Path(audio) if audio else None, remux=False)
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": "KEEP"}


if __name__ == "__main__":
    serve(load, run, describe)
