"""F1 worker: InfiniteTalk (MeiGen-AI, sparse-frame video dubbing on Wan2.1-I2V-14B-480P).

Video-to-video mode (``cond_video`` + ``cond_audio``): regenerates head, expression and body
from the audio, so identity drifts (CSIM 0.775 in its own HDTF table) — the F1 CSIM gate is what
should catch it. params: size (infinitetalk-480 | infinitetalk-720), sample_steps (40),
mode (streaming), motion_frame (9), quant (none | fp8 | int8), quant_file (e.g.
quant_models/infinitetalk_single_fp8.safetensors), num_persistent_param_in_dit (unset = keep all
on GPU; 0 = offload, never on Thalassa), prompt ("A person is talking"), weights_dir (default
~/.opendub/weights/infinitetalk; the env recipe downloads the three HF repos there),
repo_dir (default ~/.opendub/src/InfiniteTalk). Model reloads per item (subprocess per clip).
"""
from __future__ import annotations

import json
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
    repo = repo_dir(params, "InfiniteTalk")
    weights = Path(params.get("weights_dir") or Path.home() / ".opendub" / "weights"
                   / "infinitetalk").expanduser()
    for sub in ("Wan2.1-I2V-14B-480P", "chinese-wav2vec2-base", "InfiniteTalk"):
        if not (weights / sub).exists():
            raise FileNotFoundError(f"InfiniteTalk weights missing: {weights / sub}")
    return {"repo": repo, "weights": weights, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    video, _ = item_io(item)
    work = workdir(out)
    clip = to_cfr(video, work / "clip.mp4", fps=25)
    audio = driving_audio(item, work)
    p, w = state["params"], state["weights"]
    spec = work / "input.json"
    spec.write_text(json.dumps({"prompt": p.get("prompt", "A person is talking"),
                                "cond_video": str(clip), "cond_audio": {"person1": str(audio)}}))
    save = work / "infinitetalk"
    cmd = [sys.executable, "generate_infinitetalk.py",
           "--ckpt_dir", w / "Wan2.1-I2V-14B-480P", "--wav2vec_dir", w / "chinese-wav2vec2-base",
           "--infinitetalk_dir", w / "InfiniteTalk" / "single" / "infinitetalk.safetensors",
           "--input_json", spec, "--size", p.get("size", "infinitetalk-480"),
           "--sample_steps", str(p.get("sample_steps", 40)), "--mode", p.get("mode", "streaming"),
           "--motion_frame", str(p.get("motion_frame", 9)), "--save_file", save]
    if p.get("num_persistent_param_in_dit") is not None:
        cmd += ["--num_persistent_param_in_dit", str(p["num_persistent_param_in_dit"])]
    if p.get("quant") and p["quant"] != "none":
        cmd += ["--quant", p["quant"], "--quant_dir",
                w / "InfiniteTalk" / p.get("quant_file",
                                           f"quant_models/infinitetalk_single_{p['quant']}"
                                           ".safetensors")]
    sh(cmd, cwd=state["repo"], timeout=6 * 3600)
    produced = save.with_suffix(".mp4")
    if not produced.exists():
        produced = max(work.glob("*.mp4"), key=lambda f: f.stat().st_mtime)
    payload = video_payload(out, produced, keep_audio_copy(audio, out))
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    p = state["params"]
    return {"model": "InfiniteTalk-single", "size": p.get("size", "infinitetalk-480"),
            "quant": p.get("quant", "none")}


if __name__ == "__main__":
    serve(load, run, describe)
