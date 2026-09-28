"""G1 method worker: a two-family recogniser panel (Whisper seq2seq + wav2vec2 CTC, no LM).

Implements the "gate on agreement of lineage-disjoint recognisers" judge from the G1 research:
both recognisers transcribe the take, each is scored against the intended line, and the defect
score combines them (``combine: max`` = flag if either family hears an error; ``mean``). A take
both families transcribe correctly is almost certainly fine; disagreement is the review queue.

item.inputs: {audio, text}. Payload: ``{"text": <whisper transcript>, "texts": {...},
"defect_score", "metrics": {"cer_whisper", "cer_ctc", "cer_between"}}``.

params: whisper (asr_faster_whisper params), ctc (g1_ctc params), combine. Runs in the server env
(faster-whisper + transformers are both there); both models stay loaded for the whole job.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import asr_faster_whisper
import g1_ctc
from _sdk import Unsupported, serve


def _chars(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text).casefold()
    return [ch for ch in t if not unicodedata.category(ch).startswith(("P", "Z", "S", "C"))]


def _cer(ref: str, hyp: str) -> float:
    r, h = _chars(ref), _chars(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def load(params: dict, lang: str):
    wp = {"model": "large-v3", "compute_type": "float16", "beam_size": 5,
          "word_timestamps": False, "condition_on_previous_text": False,
          **params.get("whisper", {})}
    cp = {"model": "facebook/mms-1b-all", "mode": "greedy", **params.get("ctc", {})}
    return {"whisper": asr_faster_whisper.load(wp, lang), "ctc": g1_ctc.load(cp, lang),
            "combine": params.get("combine", "max"), "wp": wp, "cp": cp}


def run(state: dict, item: dict, out: Path) -> dict:
    intended = item["inputs"].get("text")
    if not intended:
        raise Unsupported("the panel judge needs inputs.text (the intended line)")
    w = asr_faster_whisper.run(state["whisper"], item, out)["text"]
    c = g1_ctc.run(state["ctc"], item, out)["text"]
    cw, cc = _cer(intended, w), _cer(intended, c)
    score = max(cw, cc) if state["combine"] == "max" else (cw + cc) / 2
    return {"text": w, "texts": {"whisper": w, "ctc": c}, "defect_score": score,
            "metrics": {"cer_whisper": cw, "cer_ctc": cc, "cer_between": _cer(w, c)}}


def describe(state: dict) -> dict:
    return {"whisper": state["wp"]["model"], "ctc": state["cp"]["model"],
            "combine": state["combine"]}


if __name__ == "__main__":
    serve(load, run, describe)
