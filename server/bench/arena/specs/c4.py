"""C4 — reference-free MT quality estimation, meta-evaluated against human ratings.

Candidates are QE models (MetricX-24 hybrid in QE mode, CometKiwi, xCOMET-QE, GEMBA-ESA LLM
judges, rule checks). Each one scores (source, hypothesis) pairs whose human quality is known
(WMT23 QE DA en-hi, WMT24/25 ESA/MQM via mt-metrics-eval); the leaderboard ranks candidates by
agreement with those humans (``Spec.corpus``, bootstrapped over documents).

- **Primary — ``acc_eq``**: within-source pairwise accuracy with tie calibration (Deutsch,
  Foster & Freitag 2023, the WMT23+ metrics-task segment-level statistic). Pairs are formed only
  between translations *of the same source segment*; a metric tie is declared when the score
  difference is within ε, with ε chosen to maximise accuracy (optimistic, as in WMT). Packs with
  a single translation per source (WMT23 QE DA) have no within-source pairs, so the statistic
  falls back to all pairs in the pack.
- Secondary: segment Kendall τ and Pearson r against the human score (global).

Scores are sign-normalised into ``qe_pred`` (higher = better): MetricX is an error score and is
negated. Source segment ids travel inside the corpus target (see :func:`encode_target`) because
``CorpusMetric`` callables only see (predictions, targets).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..judges import CorpusMetric, Row, Spec, cost, speed
from ..packs import Item

_SRC_SCALE = 10_000.0     # target = src_id * _SRC_SCALE + (human + _OFFSET)
_OFFSET = 5_000.0 / 2     # human scores (DA z, ESA 0-100, MQM ≤ 0) stay within ±2500


def encode_target(item: Item) -> float | None:
    h = item.refs.get("human")
    if h is None:
        return None
    return float(int(item.refs.get("src_id", 0))) * _SRC_SCALE + float(h) + _OFFSET


def decode_targets(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.floor(targets / _SRC_SCALE)
    return src, targets - src * _SRC_SCALE - _OFFSET


def _within_pairs(src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All index pairs (i < j) that share a source, vectorised (no Python loop per group)."""
    order = np.argsort(src, kind="stable")
    s = src[order]
    n = len(s)
    # end (exclusive) of each element's group in sorted order
    boundaries = np.r_[np.nonzero(s[1:] != s[:-1])[0] + 1, n]
    group_end = boundaries[np.searchsorted(boundaries, np.arange(n), side="right")]
    counts = group_end - np.arange(n) - 1
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, int), np.empty(0, int)
    ii = np.repeat(np.arange(n), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    jj = ii + 1 + (np.arange(total) - starts)
    return order[ii], order[jj]


def acc_eq(preds: list[float], targets: list[float]) -> float:
    """Tie-calibrated pairwise accuracy; within source when sources repeat (module docstring).

    Exact search over ε: with pairs sorted by |Δpred|, declaring the first k pairs metric ties
    makes the human-tied ones among them correct and the agreeing non-tied ones after them
    correct, so every threshold is evaluated with two cumulative sums (O(P log P))."""
    p = np.asarray(preds, float)
    src, human = decode_targets(np.asarray(targets, float))
    i, j = _within_pairs(src)
    if not len(i):
        i, j = np.triu_indices(len(p), 1)
    if not len(i):
        return float("nan")
    dh, dp = human[i] - human[j], p[i] - p[j]
    tie = dh == 0
    agree = (np.sign(dp) == np.sign(dh)) & ~tie
    order = np.argsort(np.abs(dp), kind="stable")
    a = np.abs(dp)[order]
    cum_tie, cum_agree = np.cumsum(tie[order]), np.cumsum(agree[order])
    valid = np.r_[a[1:] != a[:-1], True]           # thresholds between distinct |Δpred|
    acc = (cum_tie + (agree.sum() - cum_agree)) / len(a)
    return float(max(acc[valid].max(), agree.sum() / len(a)))    # ε below every |Δpred|


def qe_pred(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    metrics = out.get("metrics") or {}
    for name, value in metrics.items():
        if value is None or not math.isfinite(float(value)):
            continue
        if name.startswith(("gemba_major", "gemba_minor")):
            continue
        v = -float(value) if "metricx" in name else float(value)
        return [("qe_pred", v, 1.0)]
    return []


def _human(item: Item) -> float | None:
    h = item.refs.get("human")
    return None if h is None else float(h)


SPEC = Spec(
    id="C4",
    title="Reference-free MT quality estimation (meta-evaluation)",
    judges={"qe_pred@1": qe_pred, "speed@1": speed, "cost@1": cost},
    primary={"*": "acc_eq"},
    higher_is_better={"acc_eq": True, "kendall": True, "pearson": True, "qe_pred": True,
                      "rtfx": True, "cost_usd": False},
    threshold={"acc_eq": 0.005, "kendall": 0.01},
    secondary=["kendall", "pearson", "cost_usd"],
    corpus={"acc_eq": CorpusMetric(pred="qe_pred", target=encode_target, fn=acc_eq),
            "kendall": CorpusMetric(pred="qe_pred", target=_human, fn="kendall"),
            "pearson": CorpusMetric(pred="qe_pred", target=_human, fn="pearson")},
    packs=["wmt23_qe_da", "mtme"],
    io="""pack lang = direction (en-hi, en-ja, ja-en …).
item.inputs: {source, hypothesis, reference? (ignored by QE candidates), src_lang, tgt_lang};
item.refs: {human (DA z / ESA 0-100 / MQM ≤ 0; higher = better), src_id (int, same for all
translations of one source segment)}; group = document; meta: {system, source, protocol}.
payload: {metrics: {qe_<model> | hyb_<model> | gemba_esa | qe_rules: value}}.""",
)
