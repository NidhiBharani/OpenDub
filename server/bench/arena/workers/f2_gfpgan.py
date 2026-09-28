"""F2 worker: GFPGAN v1.4 / RestoreFormer via TencentARC/GFPGAN (image-only, per frame).

Licence: Apache-2.0 except third-party components (commercial status unclear, research brief);
RestoreFormer weights' terms unconfirmed. Frames are extracted, restored with
``inference_gfpgan.py -i <dir> -o <dir> -v <version> -s 1 --bg_upsampler none``, re-assembled at
the source rate and composited inside the F1 edit mask. params: version ("1.4" |
"RestoreFormer"), weight (0.5 blend), mask_only (true), repo_dir (default ~/.opendub/src/GFPGAN).
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve
from f_common import assemble_images, cleanup, extract_pngs, probe, repo_dir, video_payload, workdir
from f_common import run as sh


def load(params: dict, lang: str):
    return {"repo": repo_dir(params, "GFPGAN"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    work = workdir(out)
    fps = probe(item["inputs"]["video"])["fps"] or 25.0
    frames = extract_pngs(item["inputs"]["video"], work / "frames")
    sh([sys.executable, "inference_gfpgan.py", "-i", frames, "-o", work / "res", "-v",
        str(p.get("version", "1.4")), "-s", "1", "--bg_upsampler", "none",
        "-w", str(p.get("weight", 0.5))], cwd=state["repo"])
    restored = sorted((work / "res" / "restored_imgs").glob("*.png"))
    if not restored:
        raise RuntimeError("GFPGAN wrote no restored_imgs")
    produced = assemble_images(restored, fps, work / "restored.mp4")
    from f_frames import restore_finish

    audio = item["inputs"].get("audio")
    final = restore_finish(produced, item, out.with_suffix(".mp4"),
                           mask_only=bool(p.get("mask_only", True)))
    payload = video_payload(out, final, Path(audio) if audio else None, remux=False)
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": f"GFPGAN-{state['params'].get('version', '1.4')}"}


if __name__ == "__main__":
    serve(load, run, describe)
