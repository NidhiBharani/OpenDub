"""G6 worker (also G3/G4 LLM-judge candidates): NVIDIA Audio Flamingo 3 (audio only, non-commercial).

transformers-native checkpoint ``nvidia/audio-flamingo-3-hf``
(``AudioFlamingo3ForConditionalGeneration`` + ``AutoProcessor``; NVIDIA OneWay Noncommercial
licence), 8B params in BF16 (~17 GB weights, so a Spark / 24 GB+ card). Best audio-only emotion
accuracy of six LALM judges in the 2026 audit (0.68). It cannot see video, so off_sync counts as a
miss, and each request carries one audio clip: two-clip questions are joined with 1 s of silence
and the layout is described in the prompt (``g6_rubric.concat_for_single_audio``).

params: model, revision, dtype, task, protocol, max_new_tokens.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import g6_rubric as R
from _sdk import serve


def load(params: dict, lang: str):
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    repo = params.get("model", "nvidia/audio-flamingo-3-hf")
    processor = AutoProcessor.from_pretrained(repo, revision=params.get("revision"))
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        repo, revision=params.get("revision"), device_map=params.get("device_map", "cuda:0"),
        dtype=getattr(torch, params.get("dtype", "bfloat16"))).eval()
    return {"model": model, "processor": processor, "params": params, "lang": lang,
            "repo": repo, "tmp": Path(tempfile.mkdtemp(prefix="opendub-af3-"))}


def ask(state: dict, parts: list, prompt: str) -> tuple[str, dict]:
    import torch

    audio_parts = [p for p in parts if p[0] == "audio"]
    note = ""
    if len(audio_parts) > 1:
        parts, note = R.concat_for_single_audio(parts, state["tmp"])
    audio = next((v for k, v in parts if k == "audio"), None)
    text = "\n".join([R.SYSTEM] + [v for k, v in parts if k == "text"] + [note, prompt]).strip()
    content = [{"type": "text", "text": text}]
    if audio:
        content.append({"type": "audio", "path": R.compact_audio(audio, state["tmp"])})
    proc, model = state["processor"], state["model"]
    inputs = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                      add_generation_prompt=True, return_dict=True
                                      ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False,
                             max_new_tokens=int(state["params"].get("max_new_tokens", 512)))
    n_in = inputs["input_ids"].shape[1]
    answer = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
    return answer, {"in": int(n_in), "out": int(out.shape[1] - n_in)}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    return R.run_item(ask, state, item, state["lang"], task=p.get("task", "review"),
                      protocol=p.get("protocol", "decomposed"), caps={"video": False})


def describe(state: dict) -> dict:
    import transformers

    return {"model": state["repo"], "revision": state["params"].get("revision"),
            "task": state["params"].get("task", "review"),
            "protocol": state["params"].get("protocol", "decomposed"),
            "transformers": transformers.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
