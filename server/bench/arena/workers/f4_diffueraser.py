"""F4 worker: DiffuEraser / ProPainter video inpainting (lixiaowen-xw/DiffuEraser) + render.

Mask-based: the pack's text boxes (padded) become a binary mask video over each text's span;
``run_diffueraser.py --input_video --input_mask --save_path`` writes the ProPainter prior and the
DiffuEraser result. ``use: propainter`` takes the prior only (the generic-inpainting baseline the
research says specialised erasers beat by ~9 dB). Then the translation is rendered in each box.
Licence: DiffuEraser Apache-2.0, but the ProPainter prior is NTU S-Lab (non-commercial).
params: use (diffueraser | propainter), max_img_size (960), mask_dilation_iter (8), pad (6),
render (true), font_dir, repo_dir (default ~/.opendub/src/DiffuEraser). VRAM per README: 33 GB
at 1280x720, 20 GB at 960x540, 12 GB at 640x360.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve
from f_common import cleanup, repo_dir, video_payload, workdir
from f_common import run as sh


def load(params: dict, lang: str):
    return {"repo": repo_dir(params, "DiffuEraser"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    from f4_common import render_replacement
    from f_frames import read_frames, write_frames

    p = state["params"]
    work = workdir(out)
    src, fps = read_frames(item["inputs"]["video"])
    n, h, w, _ = src.shape
    mask = np.zeros_like(src)
    pad = int(p.get("pad", 6))
    for t in item["inputs"]["texts"]:
        x0, y0, x1, y1 = (int(v) for v in t["box"])
        i0, i1 = int(t["start"] * fps), min(n, int(np.ceil(t["end"] * fps)))
        mask[i0:i1, max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 255
    mask_path = write_frames(mask, fps, work / "mask.mp4")
    res = work / "res"
    sh([sys.executable, "run_diffueraser.py", "--input_video", item["inputs"]["video"],
        "--input_mask", mask_path, "--save_path", res,
        "--video_length", str(int(np.ceil(n / fps)) + 1),
        "--max_img_size", str(p.get("max_img_size", 960)),
        "--mask_dilation_iter", str(p.get("mask_dilation_iter", 8))], cwd=state["repo"])
    videos = sorted(res.rglob("*.mp4"), key=lambda f: f.stat().st_mtime)
    want = "prior" if p.get("use", "diffueraser") == "propainter" else "diffueraser"
    pick = [v for v in videos if want in v.name.lower()] or videos
    if not pick:
        raise RuntimeError("DiffuEraser wrote no mp4")
    frames, _ = read_frames(pick[-1], fps=fps, size=(w, h))
    if len(frames) < n:
        frames = np.concatenate([frames, src[len(frames):]], axis=0)
    if p.get("render", True):
        frames = render_replacement(frames, fps, item["inputs"]["texts"], p, plate=False)
    dst = write_frames(frames[:n], fps, out.with_suffix(".mp4"))
    payload = video_payload(out, dst, None, remux=False)
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": state["params"].get("use", "diffueraser")}


if __name__ == "__main__":
    serve(load, run, describe)
