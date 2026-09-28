"""G6 — multimodal reviewer, meta-evaluated on an injected-defect pack.

Candidates: audio/video LLM reviewers (Gemini, GPT audio, Qwen3.5-Omni API, Qwen3-Omni-30B-A3B
on Spark, Audio Flamingo 3) in holistic and decomposed protocols (``g6_rubric``), and the
transcript-plus-signal text review (Claude). Each returns ``metrics.p_<defect>`` for
wrong_speaker, off_sync, clipping, untranslated. A defect is "flagged" at p ≥ 0.5.

Primary ``bal_acc``: per item 1 if the verdict is right (a defective item has its defect
flagged; a clean item has nothing flagged), weighted by refs.weight so the mean is the macro
recall over {clean, each defect}. Secondary: per-defect recall, the clean false-alarm rate and
per-defect AUC of p_<defect> (threshold-free). Audio-only reviewers cannot see the picture and
score off-sync items as misses.
"""
from __future__ import annotations

from typing import Any

from ..judges import CorpusMetric, Row, Spec, cost, speed
from ..packs import Item
from ._gmeta import auc

DEFECTS = ("wrong_speaker", "off_sync", "clipping", "untranslated")
FLAG = 0.5


def _applicable(item: Item, d: str) -> bool:
    if d == "off_sync":
        return bool(item.refs.get("has_video"))
    if d == "wrong_speaker":
        return bool(item.refs.get("has_ref", True))
    if d == "untranslated":
        return bool(item.inputs.get("text"))  # video items carry no intended line
    return True


def review_rows(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    m = out.get("metrics") or {}
    if not any(f"p_{d}" in m for d in DEFECTS):
        return []
    truth = set(item.refs.get("defects") or [])
    p = {d: float(m.get(f"p_{d}") or 0.0) for d in DEFECTS}
    flagged = {d for d in DEFECTS if p[d] >= FLAG and _applicable(item, d)}
    rows: list[Row] = [(f"p_{d}", p[d], 1.0) for d in DEFECTS if _applicable(item, d)]
    if truth:
        correct = truth <= flagged
        for d in truth:
            rows.append((f"recall_{d}", float(d in flagged), 1.0))
    else:
        correct = not flagged
        rows.append(("false_alarm", float(bool(flagged)), 1.0))
    rows.append(("bal_acc", float(correct), float(item.refs.get("weight", 1.0))))
    return rows


def _target(d: str):
    def target(item: Item) -> float | None:
        if not _applicable(item, d):
            return None
        return 1.0 if d in (item.refs.get("defects") or []) else 0.0
    return target


SPEC = Spec(
    id="G6",
    title="Multimodal reviewer — per-defect detection on injected defects",
    judges={"review@1": review_rows, "speed@1": speed, "cost@1": cost},
    primary={"*": "bal_acc"},
    higher_is_better={"bal_acc": True, "false_alarm": False, "rtfx": True, "cost_usd": False,
                      **{f"recall_{d}": True for d in DEFECTS},
                      **{f"auc_{d}": True for d in DEFECTS},
                      **{f"p_{d}": False for d in DEFECTS}},
    threshold={"bal_acc": 0.03},
    secondary=[*(f"recall_{d}" for d in DEFECTS), "false_alarm",
               *(f"auc_{d}" for d in DEFECTS), "cost_usd"],
    corpus={f"auc_{d}": CorpusMetric(f"p_{d}", _target(d), auc) for d in DEFECTS},
    packs=["g6_defects"],
    io="""item.inputs: {audio (take), ref_audio (intended voice), text, src_text, video?};
item.refs: {defects: [...], has_video, has_ref, weight}.
payload: {metrics: {p_wrong_speaker, p_off_sync, p_clipping, p_untranslated}, flags, answers}.""",
)
