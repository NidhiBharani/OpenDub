"""F2 worker: CodeFormer (sczhou/CodeFormer, NTU S-Lab License 1.0: non-commercial).

Single-image codebook restorer applied frame by frame (``inference_codeformer.py`` accepts a
video path); the research brief keeps it only as a floor, at high fidelity weight. params: w
(fidelity 0..1, default 0.7), face_upsample (false), detection_model (retinaface_resnet50),
mask_only (true: keep restored pixels only inside the F1 edit mask), repo_dir (default
~/.opendub/src/CodeFormer). Weights auto-download from the GitHub release on first use.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve
from f_common import cleanup, newest, repo_dir, video_payload, workdir
from f_common import run as sh


def load(params: dict, lang: str):
    return {"repo": repo_dir(params, "CodeFormer"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    work = workdir(out)
    cmd = [sys.executable, "inference_codeformer.py", "--input_path", item["inputs"]["video"],
           "-o", work / "res", "-w", str(p.get("w", 0.7)), "-s", "1", "--bg_upsampler", "None",
           "--detection_model", p.get("detection_model", "retinaface_resnet50"),
           "--save_video_fps", "25"]
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
    return {"model": "CodeFormer", "w": state["params"].get("w", 0.7)}


if __name__ == "__main__":
    serve(load, run, describe)
