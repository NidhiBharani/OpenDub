"""G1 — intelligibility judge (ASR round-trip), meta-evaluated on labelled defective takes.

Candidates are recognisers (the A4 workers, the strict CTC judge ``g1_ctc``, the two-family
panel ``g1_panel``). Each transcribes a take; the function judge scores the transcript against
the line it was meant to say (CER, every language, so scores compare across judges) — or uses
the worker's own ``defect_score`` (forced CTC likelihood, panel) — and the leaderboard ranks
judges by how well that score separates human/constructed "defective" takes from good ones
(detection AUC, primary). ``floor_cer`` is the judge's CER on the clean takes: the recogniser
floor on real speech that per-language thresholds are set against (plan §4.6).
"""
from __future__ import annotations

from typing import Any

from ..judges import CorpusMetric, Row, Spec, cost, error_rows, speed
from ..packs import Item
from ._gmeta import auc, binary_ref


def defect_rows(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    intended = item.refs.get("text") or item.inputs.get("text") or ""
    rows: list[Row] = []
    cer = None
    if intended and "text" in out:
        errs = {m: (v, w) for m, v, w in error_rows(item.lang, str(intended),
                                                    str(out.get("text") or ""))}
        if "cer" in errs:
            cer, w = errs["cer"]
            rows.append(("cer", cer, w))
            if item.refs.get("defective") == 0:
                rows.append(("floor_cer", cer, w))
    score = out.get("defect_score", cer)
    if score is not None:
        rows.append(("defect_score", float(score), 1.0))
    return rows


SPEC = Spec(
    id="G1",
    title="Intelligibility judge (ASR round-trip) — detection of defective takes",
    judges={"defect@1": defect_rows, "speed@1": speed, "cost@1": cost},
    primary={"*": "auc"},
    higher_is_better={"auc": True, "defect_score": False, "cer": False, "floor_cer": False,
                      "rtfx": True, "cost_usd": False},
    threshold={"auc": 0.02},
    secondary=["floor_cer", "cer", "rtfx", "cost_usd"],
    corpus={"auc": CorpusMetric("defect_score", binary_ref("defective"), auc)},
    packs=["g1_defects", "arena_audit", "user_ratings"],
    io="""item.inputs: {audio, text (the intended line)}; item.refs: {defective: 0|1, text}.
payload: A4 contract {text, segments…}; optional defect_score (higher = more likely defective,
e.g. g1_ctc forced mode: CTC NLL per token; g1_panel: max CER over two families).""",
)
