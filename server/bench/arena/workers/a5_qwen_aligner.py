"""A5 worker: Qwen3-ForcedAligner-0.6B (Apache-2.0) via ``qwen-asr``:
``Qwen3ForcedAligner.from_pretrained(id, dtype, device_map).align(audio=…, text=…,
language="Japanese")`` → items with ``text, start_time, end_time`` (seconds). 11 languages
(zh en yue fr de it ja ko pt ru es — no Hindi), clips ≤ 5 min (QwenLM/Qwen3-ASR README,
2026-09-28). Output units are projected onto the given text's tokens. Env qwen3asr.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve
from a5_common import project, tokenize

LANG_NAMES = {"en": "English", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
              "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
              "pt": "Portuguese", "ru": "Russian", "yue": "Cantonese"}


def load(params: dict, lang: str):
    import torch
    from qwen_asr import Qwen3ForcedAligner

    if lang not in LANG_NAMES:
        raise ValueError(f"Qwen3-ForcedAligner does not cover {lang!r}")
    model = Qwen3ForcedAligner.from_pretrained(
        params.get("model", "Qwen/Qwen3-ForcedAligner-0.6B"), revision=params.get("revision"),
        dtype=getattr(torch, params.get("dtype", "bfloat16")), device_map="cuda:0")
    return {"model": model, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    if float(item.get("meta", {}).get("duration_s") or 0) > 300:
        raise Unsupported("aligner takes clips ≤ 5 min")
    text = item["inputs"]["text"]
    res = state["model"].align(audio=item["inputs"]["audio"], text=text,
                               language=LANG_NAMES[state["lang"]])
    res = res[0] if res and isinstance(res[0], list) else res
    units = [{"start": float(r.start_time), "end": float(r.end_time), "word": r.text}
             for r in res]
    return {"words": project(tokenize(text, state["lang"]), units), "units": units}


if __name__ == "__main__":
    serve(load, run)
