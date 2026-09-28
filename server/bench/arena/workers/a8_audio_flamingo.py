"""A3/A8 worker: NVIDIA Audio Flamingo 3 / Audio Flamingo Next (8B, NVIDIA non-commercial /
OneWay Noncommercial licences — eval only) with the ``a8_llm`` task prompts.

transformers docs + HF cards (checked 2026-09-28): ``AutoProcessor`` +
``AudioFlamingoNextForConditionalGeneration`` (nvidia/audio-flamingo-next-hf, -think-hf,
-captioner-hf) or ``AudioFlamingo3ForConditionalGeneration`` (nvidia/audio-flamingo-3-hf);
conversation ``[{"role": "user", "content": [{"type": "text", …}, {"type": "audio", "path": …}]}]``
through ``apply_chat_template(tokenize=True, return_dict=True)``. 16 kHz mono.
params: model, family (next | af3), task, max_new_tokens.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a8_llm import duration, prompt, to_payload


def load(params: dict, lang: str):
    import torch
    import transformers as tf

    mid = params.get("model", "nvidia/audio-flamingo-next-hf")
    cls = (tf.AudioFlamingo3ForConditionalGeneration if params.get("family") == "af3"
           else tf.AudioFlamingoNextForConditionalGeneration)
    model = cls.from_pretrained(mid, revision=params.get("revision"), dtype=torch.bfloat16,
                                device_map="cuda")
    proc = tf.AutoProcessor.from_pretrained(mid, revision=params.get("revision"))
    return {"model": model, "proc": proc, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    task = p.get("task", "delivery")
    audio = item["inputs"]["audio"]
    conv = [{"role": "user", "content": [
        {"type": "text", "text": prompt(task, lang, item.get("meta"))},
        {"type": "audio", "path": audio}]}]
    proc, model = state["proc"], state["model"]
    inp = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                   return_dict=True).to(model.device, dtype=model.dtype)
    ids = model.generate(**inp, max_new_tokens=int(p.get("max_new_tokens", 800)))
    reply = proc.decode(ids[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return {**to_payload(task, reply, duration(audio)), "language": lang}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model"), "task": state["params"].get("task")}


if __name__ == "__main__":
    serve(load, run, describe)
