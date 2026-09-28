"""F4 worker: CLEAR mask-free video subtitle removal (silent-commit/CLEAR, Apache-2.0) + render.

Erase with CLEAR (a rank-64 LoRA on Wan2.1-Fun-V1.1-1.3B-Control via DiffSynth-Studio), then
draw the translation into each box on the erased frames (same renderer as the overlay method,
without a plate). CLEAR is trained on burned-in subtitles: its prior fits captions and lower
thirds, not scene text. ~4.86 s/frame in the paper.
params: model_base_path (default ~/.opendub/weights/Wan2.1-Fun-V1.1-1.3B-Control),
lora_checkpoint (default <repo>/checkpoints/CLEAR-mask-free-subtitle-removal.pt), lora_rank
(64), lora_scale (1.0), num_steps (5, as the README; the code default is 20), cfg_scale (1.0),
sliding_window (true), render (true), font_dir, repo_dir (default ~/.opendub/src/CLEAR).
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve
from f_common import cleanup, newest, repo_dir, video_payload, workdir
from f_common import run as sh


def load(params: dict, lang: str):
    repo = repo_dir(params, "CLEAR")
    base = Path(params.get("model_base_path") or Path.home() / ".opendub" / "weights"
                / "Wan2.1-Fun-V1.1-1.3B-Control").expanduser()
    lora = Path(params.get("lora_checkpoint")
                or repo / "checkpoints" / "CLEAR-mask-free-subtitle-removal.pt").expanduser()
    for pth in (base, lora):
        if not pth.exists():
            raise FileNotFoundError(f"CLEAR weights missing: {pth}")
    return {"repo": repo, "base": base, "lora": lora, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    from f4_common import render_replacement
    from f_frames import read_frames, write_frames

    p = state["params"]
    work = workdir(out)
    cmd = [sys.executable, "inference.py", "--model_base_path", state["base"],
           "--lora_checkpoint", state["lora"], "--lora_rank", str(p.get("lora_rank", 64)),
           "--lora_scale", str(p.get("lora_scale", 1.0)), "--input_video",
           item["inputs"]["video"], "--output_dir", work / "res",
           "--num_steps", str(p.get("num_steps", 5)), "--cfg_scale", str(p.get("cfg_scale", 1.0))]
    if p.get("sliding_window", True):
        cmd.append("--use_sliding_window")
    sh(cmd, cwd=state["repo"], env={"PYTHONPATH": str(state["repo"])}, timeout=8 * 3600)
    erased = newest(work / "res")
    src, fps = read_frames(item["inputs"]["video"])
    frames, _ = read_frames(erased, fps=fps, size=(src.shape[2], src.shape[1]))
    if p.get("render", True):
        frames = render_replacement(frames, fps, item["inputs"]["texts"], p, plate=False)
    dst = write_frames(frames, fps, out.with_suffix(".mp4"))
    payload = video_payload(out, dst, None, remux=False)
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": "CLEAR", "base": state["base"].name,
            "num_steps": state["params"].get("num_steps", 5)}


if __name__ == "__main__":
    serve(load, run, describe)
