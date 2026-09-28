"""F2 worker: GPEN-BFR-512 (yangxy/GPEN, image-only, per frame).

No licence file; the README calls the released model "not our best model due to commercial
issues" and the research brief records it as Alibaba academic / non-commercial. ``demo.py
--task FaceEnhancement --model GPEN-BFR-512 --in_size 512 --indir <frames> --outdir <dir>``;
the enhanced full frames are matched back by file stem (the output naming is not documented, so
any file starting with the frame stem and not containing "_face" is taken). params: model
(GPEN-BFR-512), in_size (512), mask_only (true), repo_dir (default ~/.opendub/src/GPEN).
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve
from f_common import assemble_images, cleanup, extract_pngs, probe, repo_dir, video_payload, workdir
from f_common import run as sh


def load(params: dict, lang: str):
    return {"repo": repo_dir(params, "GPEN"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    work = workdir(out)
    fps = probe(item["inputs"]["video"])["fps"] or 25.0
    frames = extract_pngs(item["inputs"]["video"], work / "frames")
    res = work / "res"
    res.mkdir(exist_ok=True)
    sh([sys.executable, "demo.py", "--task", "FaceEnhancement", "--model",
        p.get("model", "GPEN-BFR-512"), "--in_size", str(p.get("in_size", 512)),
        "--channel_multiplier", "2", "--narrow", "1", "--use_cuda", "--indir", frames,
        "--outdir", res], cwd=state["repo"])
    outs = []
    for f in sorted(frames.glob("*.png")):
        hits = sorted(x for x in res.glob(f"{f.stem}*") if "_face" not in x.name)
        outs.append(hits[-1] if hits else f)
    produced = assemble_images(outs, fps, work / "restored.mp4")
    from f_frames import restore_finish

    audio = item["inputs"].get("audio")
    final = restore_finish(produced, item, out.with_suffix(".mp4"),
                           mask_only=bool(p.get("mask_only", True)))
    payload = video_payload(out, final, Path(audio) if audio else None, remux=False)
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model", "GPEN-BFR-512")}


if __name__ == "__main__":
    serve(load, run, describe)
