"""A4/A6 worker: MOSS-Transcribe-Diarize-0.9B (OpenMOSS, Apache-2.0) — joint ASR + speaker turns.

params: model (default OpenMOSS-Team/MOSS-Transcribe-Diarize), revision, dtype,
max_new_tokens. Output text is ``[start][S01]text[end]…`` (seconds); ``parse_transcript`` from
the repo package parses it, with a regex fallback. Returns ``text``, sentence-level
``segments`` (no word times) and ``turns`` for A6.

Usage per the model card / GitHub README (checked 2026-09-28): ``AutoModelForCausalLM`` and
``AutoProcessor`` with ``trust_remote_code=True``; ``build_transcription_messages(path)`` and
``generate_transcription(model, processor, msgs, max_new_tokens=…)`` from
``moss_transcribe_diarize.inference_utils``. The model has no language switch: the language is
recorded, not forced (declared languages gate which packs it runs on).
"""
from __future__ import annotations

import re
from pathlib import Path

from _sdk import serve

SEG_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.DOTALL)


def load(params: dict, lang: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    mid = params.get("model", "OpenMOSS-Team/MOSS-Transcribe-Diarize")
    kw = {"trust_remote_code": True, "revision": params.get("revision")}
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=getattr(torch, params.get("dtype", "bfloat16")), **kw).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(mid, **kw)
    return {"model": model, "proc": proc, "params": params, "lang": lang}


def parse(raw: str) -> list[dict]:
    try:
        from moss_transcribe_diarize import parse_transcript

        parsed = parse_transcript(raw)
        segs = []
        for s in parsed:
            get = s.get if isinstance(s, dict) else lambda k, _s=s: getattr(_s, k, None)
            segs.append({"start": float(get("start")), "end": float(get("end")),
                         "speaker": str(get("speaker")), "text": str(get("text")).strip()})
        if segs:
            return segs
    except Exception:  # noqa: BLE001, S110 - fall back to the documented tag format
        pass
    return [{"start": float(a), "end": float(d), "speaker": b, "text": c.strip()}
            for a, b, c, d in SEG_RE.findall(raw)]


def run(state: dict, item: dict, out: Path) -> dict:
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
    )

    msgs = build_transcription_messages(item["inputs"]["audio"])
    raw = generate_transcription(state["model"], state["proc"], msgs,
                                 max_new_tokens=int(state["params"].get("max_new_tokens", 4096)))
    segs = parse(raw)
    return {"text": " ".join(s["text"] for s in segs).strip() or raw.strip(),
            "segments": [{"start": s["start"], "end": s["end"], "text": s["text"], "words": []}
                         for s in segs],
            "turns": [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
                      for s in segs],
            "raw": raw, "language": state["lang"]}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model"), "revision": state["params"].get("revision")}


if __name__ == "__main__":
    serve(load, run, describe)
