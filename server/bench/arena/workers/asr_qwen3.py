"""A4 worker: Qwen3-ASR (transformers backend of the official ``qwen-asr`` package).

params: model (repo id or local dir), revision, dtype (bfloat16|float16), attn (sdpa|eager),
max_new_tokens, aligner (optional Qwen3-ForcedAligner repo id; adds word timestamps for the
languages the aligner supports, which excludes Hindi), aligner_revision.

The language is forced from the pack (Qwen3-ASR takes an English language name), so the
arena measures transcription, not language ID. Runs in its own env (qwen-asr pins
transformers 4.57), see bench/candidates/_envs.yaml.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve

LANG_NAMES = {"en": "English", "hi": "Hindi", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
              "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
              "pt": "Portuguese", "ru": "Russian"}


def load(params: dict, lang: str):
    import torch
    from qwen_asr import Qwen3ASRModel

    if lang and lang not in LANG_NAMES:
        raise ValueError(f"no Qwen3-ASR language name for {lang!r}")
    language = LANG_NAMES.get(lang)
    dtype = getattr(torch, params.get("dtype", "bfloat16"))
    device = params.get("device", "cuda:0")
    common = {"dtype": dtype, "device_map": device,
              "attn_implementation": params.get("attn", "sdpa")}

    aligner = params.get("aligner")
    kwargs: dict = {}
    if aligner:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(aligner, revision=params.get("aligner_revision"))
        if language in (getattr(cfg, "support_languages", None) or []):
            kwargs = {"forced_aligner": aligner,
                      "forced_aligner_kwargs": {**common,
                                                "revision": params.get("aligner_revision")}}
        else:
            aligner = None  # e.g. Hindi: the aligner does not cover it, emit text only

    model = Qwen3ASRModel.from_pretrained(
        params.get("model", "Qwen/Qwen3-ASR-1.7B"), revision=params.get("revision"),
        max_inference_batch_size=1, max_new_tokens=int(params.get("max_new_tokens", 448)),
        **common, **kwargs)
    return {"model": model, "params": params, "lang": lang, "language": language,
            "timestamps": bool(aligner)}


def run(state: dict, item: dict, out: Path) -> dict:
    res = state["model"].transcribe(audio=item["inputs"]["audio"], language=state["language"],
                                    return_time_stamps=state["timestamps"])[0]
    text = res.text.strip()
    segments = []
    if state["timestamps"] and res.time_stamps is not None and len(res.time_stamps):
        words = [{"start": float(w.start_time), "end": float(w.end_time), "word": w.text}
                 for w in res.time_stamps]
        segments = [{"start": words[0]["start"], "end": words[-1]["end"], "text": text,
                     "words": words}]
    return {"text": text, "segments": segments, "language": state["lang"],
            "detected_language": res.language}


def describe(state: dict) -> dict:
    from importlib.metadata import version

    import transformers

    p = state["params"]
    return {"model": p.get("model"), "revision": p.get("revision"),
            "aligner": p.get("aligner") if state["timestamps"] else None,
            "qwen_asr": version("qwen-asr"), "transformers": transformers.__version__,
            "attn": p.get("attn", "sdpa")}


if __name__ == "__main__":
    serve(load, run, describe)
