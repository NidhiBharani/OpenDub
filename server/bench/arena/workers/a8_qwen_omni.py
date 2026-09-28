"""A3/A4/A8 worker: Qwen3-Omni-30B-A3B (Instruct / Captioner; Apache-2.0) as a local audio-LLM
adjudicator / delivery-note writer / transcriber, with the ``a8_llm`` task prompts.

HF card (checked 2026-09-28): ``Qwen3OmniMoeForConditionalGeneration`` +
``Qwen3OmniMoeProcessor``; ``qwen_omni_utils.process_mm_info(messages, use_audio_in_video=False)``;
``generate(..., return_audio=False, thinker_return_dict_in_generate=True)``. BF16 needs ~79 GB
for a 15 s clip → DGX Spark only. params: model, revision, task, max_new_tokens, attn.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a8_llm import duration, prompt, to_payload


def load(params: dict, lang: str):
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    mid = params.get("model", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        mid, revision=params.get("revision"), dtype=torch.bfloat16, device_map="cuda",
        attn_implementation=params.get("attn", "sdpa"))
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    proc = Qwen3OmniMoeProcessor.from_pretrained(mid, revision=params.get("revision"))
    return {"model": model, "proc": proc, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    from qwen_omni_utils import process_mm_info

    p, lang = state["params"], state["lang"]
    task = p.get("task", "delivery")
    audio = item["inputs"]["audio"]
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": audio},
                                         {"type": "text",
                                          "text": prompt(task, lang, item.get("meta"))}]}]
    proc, model = state["proc"], state["model"]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(msgs, use_audio_in_video=False)
    inp = proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
               padding=True, use_audio_in_video=False).to(model.device).to(model.dtype)
    res = model.generate(**inp, return_audio=False, thinker_return_dict_in_generate=True,
                         max_new_tokens=int(p.get("max_new_tokens", 1024)),
                         use_audio_in_video=False)
    seqs = res[0].sequences if isinstance(res, tuple) else getattr(res, "sequences", res)
    reply = proc.batch_decode(seqs[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return {**to_payload(task, reply, duration(audio)), "language": lang}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model"), "task": state["params"].get("task")}


if __name__ == "__main__":
    serve(load, run, describe)
