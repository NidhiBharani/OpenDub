"""F1 worker: HighSync (saeed5959/high_sync, latent-diffusion lip sync, MIT, weights on HF
``saeed-5959/high_sync``).

params: repo_dir (default ~/.opendub/src/high_sync), extra_args (list of extra CLI flags).
One ``python -m inference --source_video --driving_audio --output`` call per item; input must be
25 fps, so the clip is re-encoded to 25 fps CFR first.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve
from f_common import (
    cleanup,
    driving_audio,
    item_io,
    keep_audio_copy,
    repo_dir,
    to_cfr,
    video_payload,
    workdir,
)
from f_common import run as sh


def load(params: dict, lang: str):
    return {"repo": repo_dir(params, "high_sync"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    video, _ = item_io(item)
    work = workdir(out)
    clip = to_cfr(video, work / "clip.mp4", fps=25)
    audio = driving_audio(item, work)
    produced = work / "highsync.mp4"
    sh([sys.executable, "-m", "inference", "--source_video", clip, "--driving_audio", audio,
        "--output", produced, *state["params"].get("extra_args", [])],
       cwd=state["repo"], env={"PYTHONPATH": str(state["repo"])})
    payload = video_payload(out, produced, keep_audio_copy(audio, out))
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": "HighSync"}


if __name__ == "__main__":
    serve(load, run, describe)
