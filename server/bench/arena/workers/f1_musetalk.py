"""F1 worker: MuseTalk 1.5 (TMElyralab/MuseTalk, 256x256 latent inpainting, MIT weights).

params: version ("v15"), unet_model_path / unet_config (default models/musetalkV15/...),
bbox_shift (0; the README suggests tuning per face, -7 in its example), use_float16 (true),
repo_dir (default ~/.opendub/src/MuseTalk).

One ``python -m scripts.inference`` call per item with a one-task YAML config; the result is the
newest mp4 under the result dir.
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
    repo = repo_dir(params, "MuseTalk")
    return {"repo": repo, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    video, _ = item_io(item)
    work = workdir(out)
    clip = to_cfr(video, work / "clip.mp4", fps=25)
    audio = driving_audio(item, work)
    p = state["params"]
    cfg = work / "task.yaml"
    # YAML is a superset of JSON: no yaml dependency needed to write the config.
    cfg.write_text(json.dumps({"task_0": {"video_path": str(clip), "audio_path": str(audio),
                                          "bbox_shift": int(p.get("bbox_shift", 0))}}))
    res = work / "results"
    cmd = [sys.executable, "-m", "scripts.inference", "--inference_config", cfg,
           "--result_dir", res, "--version", p.get("version", "v15"),
           "--unet_model_path", p.get("unet_model_path", "models/musetalkV15/unet.pth"),
           "--unet_config", p.get("unet_config", "models/musetalkV15/musetalk.json")]
    if p.get("use_float16", True):
        cmd.append("--use_float16")
    sh(cmd, cwd=state["repo"], env={"PYTHONPATH": str(state["repo"])})
    produced = max(res.rglob("*.mp4"), key=lambda f: f.stat().st_mtime, default=None)
    if produced is None:
        raise RuntimeError("MuseTalk produced no mp4")
    payload = video_payload(out, produced, keep_audio_copy(audio, out))
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": f"MuseTalk-{state['params'].get('version', 'v15')}"}


if __name__ == "__main__":
    serve(load, run, describe)
