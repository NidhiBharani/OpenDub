"""C3 — viseme-aware paraphrase (experimental).

Re-word a translated dub line so its mouth shapes (open vowels, rounded vowels, lip closures)
land where the original speaker's do, without losing meaning or leaving the time slot.

The real primary (docs/plans/model-ranking.md appendix, row C3) is **Δ G5 sync-panel score on
close-up frontal lines**: it needs every candidate rendered through D1 (TTS) and F1 (lip sync)
and scored by the G5 panel, which does not exist yet. Until then this spec ranks by a text-level
proxy and records the gates, so candidates can be registered and smoke-tested:

- primary (interim) ``viseme_dtw``: DTW distance between the source line's and the output's
  vowel-class/bilabial sequences (grapheme approximation from ``workers/c3_viseme_rerank.py``);
  lower is better;
- gate ``qe_drop`` ≤ 1 MetricX point: MetricX-QE of the output minus MetricX-QE of the unedited
  input line (both against the source), from two ``judge_mt_qe`` passes;
- gate ``dur_compliance``: the output still fits [0.9, 1.1]× the slot;
- the LaBSE ≥ 0.75 gate needs a sentence-embedding judge that no judgelib factory provides yet;
  it is not enforced.

When G5 renders exist, add ``sync_delta`` (from item.refs or a render judge) and switch primary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..hardware import Requires
from ..judgelib import QE_MODELS
from ..judges import ModelJudge, Row, Spec, cost
from ..packs import Item
from .c2 import duration_rows, langs_of, worker_module

METRICX = "metricx-24-hybrid-xl"


def viseme(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    vr = worker_module("c3_viseme_rerank")
    src, tgt = langs_of(item)
    source, text = item.inputs.get("source_text", ""), str(out.get("text", ""))
    if not source or not text:
        return []
    ref = vr.mouth_sequence(source, src)
    rows: list[Row] = [("viseme_dtw", vr.dtw(ref, vr.mouth_sequence(text, tgt)), 1.0),
                       ("changed", float(text.strip() != str(item.inputs.get("text", "")).strip()),
                        1.0)]
    base = vr.dtw(ref, vr.mouth_sequence(str(item.inputs.get("text", "")), tgt))
    rows.append(("viseme_gain", base - rows[0][1], 1.0))
    rows += [r for r in duration_rows(text, item) if r[0] == "dur_compliance"]
    return rows


def _qe_judge(which: str, version: int = 1) -> ModelJudge:
    """MetricX-QE of the output (``out``) or of the unedited input line (``in``) vs the source."""
    def inputs(item: Item, out: dict[str, Any], prefix: Path) -> dict[str, Any] | None:
        src = item.inputs.get("source_text")
        hyp = out.get("text") if which == "out" else item.inputs.get("text")
        if not src or not hyp:
            return None
        s, t = langs_of(item)
        return {"source": src, "hypothesis": hyp, "src_lang": s, "tgt_lang": t}

    def to_rows(item: Item, payload: dict[str, Any], out: dict[str, Any]) -> list[Row]:
        vals = [float(v) for v in (payload.get("metrics") or {}).values() if v is not None]
        return [(f"qe_{which}", vals[0], 1.0)] if vals else []

    return ModelJudge(id=f"c3_qe_{which}.{METRICX}@{version}", worker="judge_mt_qe",
                      env="judge_mt_qe", params={"model": METRICX, "use_reference": False},
                      requires=Requires.parse({"gpu": True, "vram_gb": QE_MODELS[METRICX]}),
                      lang_param=False, inputs=inputs, to_rows=to_rows)


def _qe_drop(values: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    a, b = values.get("qe_out"), values.get("qe_in")
    return (a[0] - b[0], 1.0) if a and b else None


SPEC = Spec(
    id="C3",
    title="Viseme-aware paraphrase (experimental)",
    judges={"viseme@1": viseme, "cost@1": cost},
    primary={"*": "viseme_dtw"},
    higher_is_better={"viseme_dtw": False, "viseme_gain": True, "changed": True,
                      "qe_out": False, "qe_in": False, "qe_drop": False,
                      "dur_compliance": True, "cost_usd": False},
    threshold={"viseme_dtw": 0.01},
    secondary=["viseme_gain", "qe_drop", "dur_compliance", "changed", "cost_usd"],
    model_judges=lambda lang: [_qe_judge("out"), _qe_judge("in")],
    gates={"qe_drop": ("<=", 1.0), "dur_compliance": (">=", 0.5)},
    derived={"qe_drop": _qe_drop},
    packs=[],
    io="""pack lang = direction (ja-en, ja-hi …); no builder yet: needs ~50 close-up frontal lines
per direction (B1 face track + C2 translation + A5 timing), built once G5/F1 renders exist.
item.inputs: {text (translated line), source_text (original line), src_lang, tgt_lang,
slot_s?}; payload: {text, candidates: [{text, cost, dtw, kept, seconds}], chosen}.""",
)
