"""F2 worker: DicFace (fudan-generative-vision/DicFace, Dirichlet codebook video restoration).

Licence is inconsistent (README: NTU S-Lab 1.0; HF card: MIT; no LICENSE file) → treated as NC.
``scripts/inference.py -i <video> -o <dir> --save_video`` with the BFR checkpoint from HF
``fudan-generative-ai/DicFace_model`` (downloaded by the env recipe into <repo>/ckpts).
params: ckpt_path (default <repo>/ckpts/CodeFormer/bfr_100k.pth), max_length (10),
mask_only (true), repo_dir (default ~/.opendub/src/DicFace).
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve
from f_common import cleanup, newest, repo_dir, video_payload, workdir
from f_common import run as sh


def load(params: dict, lang: str):
    repo = repo_dir(params, "DicFace")
    ckpt = Path(params.get("ckpt_path") or repo / "ckpts" / "CodeFormer" / "bfr_100k.pth")
    return {"repo": repo, "ckpt": ckpt, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    work = workdir(out)
    sh([sys.executable, "scripts/inference.py", "-i", item["inputs"]["video"], "-o", work / "res",
        "--max_length", str(p.get("max_length", 10)), "--save_video_fps", "25",
        "--ckpt_path", state["ckpt"], "--save_video"],
       cwd=state["repo"], env={"PYTHONPATH": str(state["repo"])})
    produced = newest(work / "res")
    from f_frames import restore_finish

    audio = item["inputs"].get("audio")
    final = restore_finish(produced, item, out.with_suffix(".mp4"),
                           mask_only=bool(p.get("mask_only", True)))
    payload = video_payload(out, final, Path(audio) if audio else None, remux=False)
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": "DicFace", "checkpoint": str(state["ckpt"])}


if __name__ == "__main__":
    serve(load, run, describe)
