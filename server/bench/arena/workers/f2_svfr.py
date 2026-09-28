"""F2 worker: SVFR (wangzhiyaoo/SVFR, SVD-based generalized video face restoration).

Code MIT, weights non-commercial research only. ``infer.py --config config/infer.yaml
--task_ids 0 --input_path <mp4> --output_dir <dir> --crop_face_region`` (task 0 = blind face
restoration). The README asks for ≥16 GB VRAM. params: task_ids ("0"), crop_face_region (true),
mask_only (true), repo_dir (default ~/.opendub/src/SVFR).
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve
from f_common import cleanup, newest, repo_dir, video_payload, workdir
from f_common import run as sh


def load(params: dict, lang: str):
    return {"repo": repo_dir(params, "SVFR"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    work = workdir(out)
    cmd = [sys.executable, "infer.py", "--config", "config/infer.yaml",
           "--task_ids", str(p.get("task_ids", "0")), "--input_path", item["inputs"]["video"],
           "--output_dir", work / "res"]
    if p.get("crop_face_region", True):
        cmd.append("--crop_face_region")
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
    return {"model": "SVFR", "task_ids": state["params"].get("task_ids", "0")}


if __name__ == "__main__":
    serve(load, run, describe)
