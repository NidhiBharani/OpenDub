"""F1 worker: Wav2Lip (Rudrabha/Wav2Lip, 96x96 mouth, GAN checkpoint). Current baseline.

params: checkpoint (default <repo>/checkpoints/wav2lip_gan.pth), resize_factor (1), nosmooth
(false), pads ("0 10 0 0"), repo_dir (default ~/.opendub/src/Wav2Lip).

Driven as a subprocess of ``inference.py`` per item, exactly like the app provider
(``app/providers/lipsync/wav2lip.py``); the model reloads per item (~5 s, negligible next to
S3FD face detection). Weights are research/non-commercial only (trained on LRS2).
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import Unsupported, serve
from f_common import (
    cleanup,
    driving_audio,
    item_io,
    keep_audio_copy,
    repo_dir,
    video_payload,
    workdir,
)
from f_common import run as sh


def load(params: dict, lang: str):
    repo = repo_dir(params, "Wav2Lip")
    ckpt = Path(params.get("checkpoint") or repo / "checkpoints" / "wav2lip_gan.pth").expanduser()
    if not ckpt.exists():
        raise FileNotFoundError(f"Wav2Lip checkpoint missing: {ckpt}")
    (repo / "temp").mkdir(exist_ok=True)  # inference.py writes temp/result.avi
    return {"repo": repo, "ckpt": ckpt, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    video, _ = item_io(item)
    work = workdir(out)
    audio = driving_audio(item, work)
    p = state["params"]
    produced = work / "wav2lip.mp4"
    cmd = [sys.executable, "inference.py", "--checkpoint_path", state["ckpt"], "--face", video,
           "--audio", audio, "--outfile", produced,
           "--resize_factor", str(int(p.get("resize_factor", 1))),
           "--pads", *str(p.get("pads", "0 10 0 0")).split()]
    if p.get("nosmooth"):
        cmd.append("--nosmooth")
    try:
        sh(cmd, cwd=state["repo"])
    except RuntimeError as exc:
        if "Face not detected" in str(exc):
            raise Unsupported("Wav2Lip: face not detected in some frames") from exc
        raise
    payload = video_payload(out, produced, keep_audio_copy(audio, out))
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": "Wav2Lip-GAN", "checkpoint": str(state["ckpt"])}


if __name__ == "__main__":
    serve(load, run, describe)
