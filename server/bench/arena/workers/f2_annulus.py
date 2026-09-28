"""F2 worker: classical annulus matching (no model). Research baseline "every model must beat".

Inside the F1 edit mask (where the lip-synced video differs from the untouched source): per-frame
colour mean/std transfer and grain re-synthesis from a ring of untouched pixels, then a feathered
seam. Only statistics from outside the mask are read, so on self-reenactment packs it cannot copy
ground-truth pixels. params: ring (25 px), grain (true), seed (0). Server env (numpy + ffmpeg).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve
from f_common import video_payload


def load(params: dict, lang: str):
    return {"params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    from f_frames import annulus_match, read_frames, write_frames

    ref = item["inputs"].get("ref_video")
    if not ref:
        raise Unsupported("annulus matching needs inputs.ref_video (the untouched source)")
    edited, fps = read_frames(item["inputs"]["video"])
    h, w = edited.shape[1:3]
    source, _ = read_frames(ref, fps=fps, size=(w, h))
    p = state["params"]
    result = annulus_match(edited, source, ring=int(p.get("ring", 25)),
                           grain=bool(p.get("grain", True)), seed=int(p.get("seed", 0)))
    dst = write_frames(result, fps, out.with_suffix(".mp4"), audio=item["inputs"].get("audio"))
    audio = item["inputs"].get("audio")
    return video_payload(out, dst, Path(audio) if audio else None, remux=False)


if __name__ == "__main__":
    serve(load, run)
