"""F1 worker: Wan2.2-S2V-14B (audio-driven video from a single reference image).

Not true video-to-video: it takes one frame of the source (``ref_time`` seconds in, default the
first frame) as the reference image and regenerates the whole clip from the audio — a
whole-frame "synthetic presenter" comparator, expected to lose on identity and fidelity to the
source. params: size ("1024*704"), infer_frames (80), offload_model (false on the Spark),
convert_model_dtype (true), prompt, ckpt_dir (default ~/.opendub/weights/Wan2.2-S2V-14B),
repo_dir (default ~/.opendub/src/Wan2.2). Needs ≥80 GB on one GPU per the README.
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
    video_payload,
    workdir,
)
from f_common import run as sh


def load(params: dict, lang: str):
    repo = repo_dir(params, "Wan2.2")
    ckpt = Path(params.get("ckpt_dir") or Path.home() / ".opendub" / "weights"
                / "Wan2.2-S2V-14B").expanduser()
    if not ckpt.exists():
        raise FileNotFoundError(f"Wan2.2-S2V-14B weights missing: {ckpt}")
    return {"repo": repo, "ckpt": ckpt, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    video, _ = item_io(item)
    work = workdir(out)
    audio = driving_audio(item, work)
    p = state["params"]
    ref = work / "ref.png"
    sh(["ffmpeg", "-y", "-v", "error", "-ss", str(p.get("ref_time", 0.0)), "-i", video,
        "-frames:v", "1", ref])
    produced = work / "s2v.mp4"
    cmd = [sys.executable, "generate.py", "--task", "s2v-14B", "--size", p.get("size", "1024*704"),
           "--ckpt_dir", state["ckpt"], "--prompt", p.get("prompt", "A person is talking."),
           "--image", ref, "--audio", audio, "--infer_frames", str(p.get("infer_frames", 80)),
           "--offload_model", str(bool(p.get("offload_model", False))), "--save_file", produced]
    if p.get("convert_model_dtype", True):
        cmd.append("--convert_model_dtype")
    sh(cmd, cwd=state["repo"], timeout=8 * 3600)
    payload = video_payload(out, produced, keep_audio_copy(audio, out))
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": "Wan2.2-S2V-14B", "size": state["params"].get("size", "1024*704")}


if __name__ == "__main__":
    serve(load, run, describe)
