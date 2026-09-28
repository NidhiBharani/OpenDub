"""F1 worker: ByteDance LatentSync 1.5 / 1.6, in-process (the pipeline loads once per job).

params: version ("1.6" | "1.5"), unet_config (default configs/unet/stage2_512.yaml for 1.6,
stage2.yaml for 1.5), checkpoint (relative to the repo), inference_steps (20), guidance_scale
(1.5), enable_deepcache (true), seed (1247), repo_dir (default ~/.opendub/src/LatentSync; the
user's existing clone at ~/sources/LatentSync works too).

Mirrors ``scripts/inference.py`` of the pinned commit: 25 fps / 16 kHz internally, the 512 px
(1.6) or 256 px (1.5) face crop composited back into the frame, output length = audio length
(the video is looped if the audio is longer). The repo's relative paths (``configs/``,
``checkpoints/whisper``, ``checkpoints/auxiliary``) require running from the repo dir.
"""
from __future__ import annotations

import os
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

DEFAULTS = {"1.6": ("configs/unet/stage2_512.yaml", "checkpoints/latentsync_unet.pt"),
            "1.5": ("configs/unet/stage2.yaml", "checkpoints/1.5/latentsync_unet.pt")}


def load(params: dict, lang: str):
    version = str(params.get("version", "1.6"))
    repo = repo_dir(params, "LatentSync")
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import torch
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDIMScheduler
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature
    from omegaconf import OmegaConf

    cfg_path, ckpt = DEFAULTS[version]
    config = OmegaConf.load(params.get("unet_config", cfg_path))
    ckpt = params.get("checkpoint", ckpt)
    dtype = torch.float16 if torch.cuda.get_device_capability()[0] > 7 else torch.float32
    whisper = ("checkpoints/whisper/small.pt" if config.model.cross_attention_dim == 768
               else "checkpoints/whisper/tiny.pt")
    audio_encoder = Audio2Feature(model_path=whisper, device="cuda",
                                  num_frames=config.data.num_frames,
                                  audio_feat_length=config.data.audio_feat_length)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(config.model), ckpt,
                                                   device="cpu")
    pipe = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet.to(dtype=dtype),
                           scheduler=DDIMScheduler.from_pretrained("configs")).to("cuda")
    if params.get("enable_deepcache", True):
        from DeepCache import DeepCacheSDHelper

        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
    return {"pipe": pipe, "config": config, "dtype": dtype, "params": params, "seed_fn": set_seed,
            "version": version, "repo": repo, "ckpt": ckpt}


def run(state: dict, item: dict, out: Path) -> dict:
    video, _ = item_io(item)
    work = workdir(out)
    audio = driving_audio(item, work)
    p, cfg = state["params"], state["config"]
    state["seed_fn"](int(p.get("seed", 1247)))
    produced = work / "latentsync.mp4"
    try:
        state["pipe"](video_path=str(video), audio_path=str(audio), video_out_path=str(produced),
                      num_frames=cfg.data.num_frames,
                      num_inference_steps=int(p.get("inference_steps", 20)),
                      guidance_scale=float(p.get("guidance_scale", 1.5)),
                      weight_dtype=state["dtype"], width=cfg.data.resolution,
                      height=cfg.data.resolution, mask_image_path=cfg.data.mask_image_path,
                      temp_dir=str(work / "temp"))
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "face" in msg and "detect" in msg:
            raise Unsupported(f"no usable face: {exc}") from exc
        raise
    # LatentSync already muxes the driving audio; keep a copy so files.audio survives cleanup.
    payload = video_payload(out, produced, keep_audio_copy(audio, out), remux=False)
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    import subprocess

    sha = subprocess.run(["git", "-C", str(state["repo"]), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False).stdout.strip()
    return {"model": f"LatentSync-{state['version']}", "checkpoint": state["ckpt"],
            "repo_commit": sha, "resolution": int(state["config"].data.resolution)}


if __name__ == "__main__":
    serve(load, run, describe)
