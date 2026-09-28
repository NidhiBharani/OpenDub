"""G6 worker (also G2/G3/G4/G5 LLM-judge candidates): Qwen3-Omni-30B-A3B-Instruct, self-hosted.

The open-weights (Apache-2.0) audio+video reviewer, via transformers
(``Qwen3OmniMoeForConditionalGeneration`` + ``Qwen3OmniMoeProcessor`` + ``qwen_omni_utils.
process_mm_info``, talker disabled: text answers only). The model card lists 78.85 GB peak for
a 15 s video at BF16, so this is a DGX Spark candidate. Use the Instruct variant, never Thinking
(position-locked in the 2026 judge audit).

Protocol and prompts: ``g6_rubric``. params: model, revision, dtype (bfloat16), attn (sdpa;
flash-attn has no aarch64 sm_121 wheel), task, protocol, max_new_tokens.
"""
from __future__ import annotations

from pathlib import Path

import g6_rubric as R
from _sdk import serve


def load(params: dict, lang: str):
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    repo = params.get("model", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        repo, revision=params.get("revision"), dtype=getattr(torch, params.get("dtype",
                                                                               "bfloat16")),
        device_map=params.get("device_map", "cuda:0"),
        attn_implementation=params.get("attn", "sdpa"))
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(repo, revision=params.get("revision"))
    return {"model": model, "processor": processor, "params": params, "lang": lang,
            "repo": repo}


def ask(state: dict, parts: list, prompt: str) -> tuple[str, dict]:
    import torch
    from qwen_omni_utils import process_mm_info

    content, has_video = [], False
    for kind, val in parts:
        if kind == "text":
            content.append({"type": "text", "text": val})
        else:
            content.append({"type": kind, kind: val})
            has_video |= kind == "video"
    content.append({"type": "text", "text": prompt})
    conv = [{"role": "system", "content": [{"type": "text", "text": R.SYSTEM}]},
            {"role": "user", "content": content}]
    proc, model = state["processor"], state["model"]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=has_video)
    inputs = proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                  padding=True, use_audio_in_video=has_video).to(model.device).to(model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, return_audio=False, thinker_return_dict_in_generate=True,
                             use_audio_in_video=has_video, do_sample=False,
                             max_new_tokens=int(state["params"].get("max_new_tokens", 512)))
    if isinstance(out, tuple):
        out = out[0]
    seqs = getattr(out, "sequences", out)
    n_in = inputs["input_ids"].shape[1]
    answer = proc.batch_decode(seqs[:, n_in:], skip_special_tokens=True)[0]
    return answer, {"in": int(n_in), "out": int(seqs.shape[1] - n_in)}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    return R.run_item(ask, state, item, state["lang"], task=p.get("task", "review"),
                      protocol=p.get("protocol", "decomposed"), caps={"video": True})


def describe(state: dict) -> dict:
    import transformers

    p = state["params"]
    return {"model": state["repo"], "revision": p.get("revision"),
            "task": p.get("task", "review"), "protocol": p.get("protocol", "decomposed"),
            "transformers": transformers.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
