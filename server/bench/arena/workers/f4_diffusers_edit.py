"""F4 worker: open instruction image editors via diffusers, on the mid-span keyframe.

pipeline:
  ``flux2-klein`` — Flux2KleinPipeline (black-forest-labs/FLUX.2-klein-4B, Apache-2.0, ~13 GB;
      FLUX.2-klein-9B: FLUX non-commercial, gated, ~29 GB), 4 steps, guidance 1.0
  ``flux2`` — Flux2Pipeline (black-forest-labs/FLUX.2-dev, 32B, non-commercial, gated)
  ``qwen-edit-plus`` — QwenImageEditPlusPipeline (Qwen/Qwen-Image-Edit-2511, Apache-2.0),
      40 steps, true_cfg_scale 4.0; ``transformer_gguf`` (e.g.
      unsloth/Qwen-Image-Edit-2511-GGUF:qwen-image-edit-2511-Q4_K_M.gguf) swaps in a GGUF
      transformer; ``nunchaku`` (repo:file of an SVDQuant checkpoint) swaps in Nunchaku's
      transformer (x86_64 wheels only)
params: model (HF repo), revision, dtype (bfloat16), steps, guidance, true_cfg_scale, seed (0),
cpu_offload (false: model-level offload of the text encoder is allowed only where the
candidate says so — never block-swap a 14B+ model on Thalassa), margin, max_side (1024: the
keyframe is resized so its long side fits; the edit is resized back).
Flow: edit the keyframe with ``f4_common.edit_prompt`` → paste the box region into the span.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from f_common import video_payload


def _hf_file(spec: str) -> str:
    from huggingface_hub import hf_hub_download

    repo, _, fname = spec.partition(":")
    return hf_hub_download(repo, fname)


def load(params: dict, lang: str):
    import torch

    kind = params["pipeline"]
    dtype = getattr(torch, params.get("dtype", "bfloat16"))
    kw = {"torch_dtype": dtype}
    if params.get("revision"):
        kw["revision"] = params["revision"]
    if kind == "flux2-klein":
        from diffusers import Flux2KleinPipeline as Pipe
    elif kind == "flux2":
        from diffusers import Flux2Pipeline as Pipe
    elif kind == "qwen-edit-plus":
        from diffusers import QwenImageEditPlusPipeline as Pipe

        if params.get("transformer_gguf"):
            from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel

            kw["transformer"] = QwenImageTransformer2DModel.from_single_file(
                _hf_file(params["transformer_gguf"]), torch_dtype=dtype,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                config=params["model"], subfolder="transformer")
        elif params.get("nunchaku"):
            from nunchaku import NunchakuQwenImageTransformer2DModel

            kw["transformer"] = NunchakuQwenImageTransformer2DModel.from_pretrained(
                _hf_file(params["nunchaku"]))
    else:
        raise ValueError(f"unknown pipeline {kind!r}")
    pipe = Pipe.from_pretrained(params["model"], **kw)
    if params.get("cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return {"pipe": pipe, "params": params, "kind": kind}


def _edit(state: dict):
    import torch

    p, pipe = state["params"], state["pipe"]

    def fn(image, prompt: str, text: dict):
        w, h = image.size
        side = int(p.get("max_side", 1024))
        scale = min(1.0, side / max(w, h))
        tw, th = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16
        img = image.resize((tw, th))
        gen = torch.Generator("cuda").manual_seed(int(p.get("seed", 0)))
        kw = {"prompt": prompt, "generator": gen, "num_inference_steps": int(p.get("steps", 4))}
        if state["kind"] == "qwen-edit-plus":
            kw.update(image=[img], true_cfg_scale=float(p.get("true_cfg_scale", 4.0)),
                      negative_prompt=" ", num_inference_steps=int(p.get("steps", 40)))
        else:
            kw.update(image=img, guidance_scale=float(p.get("guidance", 1.0)), width=tw,
                      height=th)
        return pipe(**kw).images[0]
    return fn


def run(state: dict, item: dict, out: Path) -> dict:
    from f4_common import keyframe_edit

    dst = keyframe_edit(item, out, _edit(state), state["params"])
    return video_payload(out, dst, None, remux=False)


def describe(state: dict) -> dict:
    import diffusers

    p = state["params"]
    return {"model": p["model"], "pipeline": state["kind"], "diffusers": diffusers.__version__,
            "gguf": p.get("transformer_gguf"), "nunchaku": p.get("nunchaku")}


if __name__ == "__main__":
    serve(load, run, describe)
