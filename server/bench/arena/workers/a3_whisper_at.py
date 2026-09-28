"""A3 worker: Whisper-AT (MIT CSAIL; ``whisper-at`` PyPI, BSD) — AudioSet tags from a Whisper
encoder at ``at_time_res`` resolution (a multiple of 0.4 s), alongside the transcript.

params: model (large-v1 | large-v2 | medium … as published by the package), at_time_res (0.4),
top_k, threshold (logit threshold, default -1: the README's ``p_threshold``).
``parse_at_label(result, language='follow_asr', top_k, p_threshold, include_class_list)`` →
per-segment ``{time: {start, end}, audio tags: [(label, logit)]}`` (README checked 2026-09-28).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a3_common import group_of


def load(params: dict, lang: str):
    import whisper_at

    return {"wa": whisper_at, "model": whisper_at.load_model(params.get("model", "large-v1")),
            "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, wa = state["params"], state["wa"]
    res = state["model"].transcribe(item["inputs"]["audio"], language=state["lang"] or None,
                                    at_time_res=float(p.get("at_time_res", 0.4)))
    tags = wa.parse_at_label(res, language="en", top_k=int(p.get("top_k", 10)),
                             p_threshold=float(p.get("threshold", -1)),
                             include_class_list=list(range(527)))
    segs = []
    for t in tags:
        tm = t.get("time", {})
        start, end = float(tm.get("start", 0)), float(tm.get("end", 0))
        seen = set()
        for label, logit in t.get("audio tags", []):
            g = group_of(label)
            if g and g not in seen:
                seen.add(g)
                segs.append({"start": start, "end": end, "label": g, "score": float(logit)})
    return {"segments": segs, "text": res.get("text", "")}


if __name__ == "__main__":
    serve(load, run)
