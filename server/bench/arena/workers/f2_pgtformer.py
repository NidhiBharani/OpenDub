"""F2 worker: PGTFormer (kepengxu/PGTFormer, parsing-guided temporal transformer, IJCAI 2024).

Licence: BSD-style with a non-commercial clause (code and weights). The model wants a 512x512
face video ("Input Video Width and Height == 512", no pre-alignment needed), so the worker crops
a square around the F1 edit region (union over the clip, from the lip-synced vs untouched
source difference), resizes it to 512, restores, and pastes it back. params: mask_only (true),
pad (0.6: crop margin around the edit region), repo_dir (default ~/.opendub/src/PGTFormer).
Weights: HF ``kepeng/pgtformer-base`` (fetched by the script / env recipe).
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve
from f_common import cleanup, repo_dir, video_payload, workdir
from f_common import run as sh


def load(params: dict, lang: str):
    return {"repo": repo_dir(params, "PGTFormer"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    from _sdk import Unsupported
    from f_frames import mask_bbox, read_frames

    p = state["params"]
    ref = item["inputs"].get("ref_video")
    if not ref:
        raise Unsupported("PGTFormer crop needs inputs.ref_video to locate the edited face")
    work = workdir(out)
    edited, fps = read_frames(item["inputs"]["video"])
    h, w = edited.shape[1:3]
    source, _ = read_frames(ref, fps=fps, size=(w, h))
    box = mask_bbox(edited, source, pad=float(p.get("pad", 0.6)))
    x0, y0, x1, y1 = box
    crop = work / "crop512.mp4"
    sh(["ffmpeg", "-y", "-v", "error", "-i", item["inputs"]["video"], "-vf",
        f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale=512:512", "-c:v", "libx264", "-crf", "12",
        "-pix_fmt", "yuv420p", "-an", crop])
    produced = work / "restored512.mp4"
    sh([sys.executable, "inference.py", f"--input_video={crop}", f"--output_video={produced}"],
       cwd=state["repo"])
    from f_frames import restore_finish

    audio = item["inputs"].get("audio")
    final = restore_finish(produced, item, out.with_suffix(".mp4"),
                           mask_only=bool(p.get("mask_only", True)), box=box)
    payload = video_payload(out, final, Path(audio) if audio else None, remux=False)
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": "PGTFormer-base"}


if __name__ == "__main__":
    serve(load, run, describe)
