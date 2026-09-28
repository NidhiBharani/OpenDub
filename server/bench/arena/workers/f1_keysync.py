"""F1 worker: KeySync (antonibigata/keysync, keyframe + interpolation latent diffusion).

params: keyframes_ckpt / interpolation_ckpt (default <repo>/pretrained_models/checkpoints/
{keyframe_dub,interpolation_dub}.pt), compute_until (seconds, default = clip length),
fix_occlusion (false), repo_dir (default ~/.opendub/src/keysync).

``scripts/infer_raw.sh`` takes a folder of 25 fps videos and a folder of 16 kHz wavs with
matching stems; one item per call, so the model reloads per item. The output file is the newest
mp4 in the output folder (the script's naming is not documented).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from _sdk import serve
from f_common import (
    cleanup,
    driving_audio,
    item_io,
    keep_audio_copy,
    probe,
    repo_dir,
    to_cfr,
    video_payload,
    workdir,
)
from f_common import run as sh


def load(params: dict, lang: str):
    repo = repo_dir(params, "keysync")
    ck = repo / "pretrained_models" / "checkpoints"
    return {"repo": repo, "params": params,
            "kf": Path(params.get("keyframes_ckpt") or ck / "keyframe_dub.pt"),
            "interp": Path(params.get("interpolation_ckpt") or ck / "interpolation_dub.pt")}


def run(state: dict, item: dict, out: Path) -> dict:
    video, _ = item_io(item)
    work = workdir(out)
    vids, auds, res = work / "videos", work / "audios", work / "out"
    for d in (vids, auds, res):
        d.mkdir(exist_ok=True)
    to_cfr(video, vids / "clip.mp4", fps=25)
    audio = driving_audio(item, work)
    shutil.copyfile(audio, auds / "clip.wav")
    p = state["params"]
    until = p.get("compute_until") or int(probe(audio)["duration"]) + 1
    cmd = ["bash", "scripts/infer_raw.sh", "--file_list", vids, "--file_list_audio", auds,
           "--output_folder", res, "--keyframes_ckpt", state["kf"], "--interpolation_ckpt",
           state["interp"], "--compute_until", str(until)]
    if p.get("fix_occlusion"):
        cmd.append("--fix_occlusion")
    sh(cmd, cwd=state["repo"])
    produced = max(res.rglob("*.mp4"), key=lambda f: f.stat().st_mtime, default=None)
    if produced is None:
        raise RuntimeError("KeySync produced no mp4")
    payload = video_payload(out, produced, keep_audio_copy(audio, out))
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": "KeySync", "keyframes": str(state["kf"]), "interpolation": str(state["interp"])}


if __name__ == "__main__":
    serve(load, run, describe)
