"""Shared pieces of the G-phase specs (judges meta-evaluated against human labels).

In G1–G6 the *candidates are judges*. Each candidate worker returns a uniform per-item
prediction (``score`` for G2–G4, ``defect_score``/transcript for G1, ``desync``/``offset_ms`` for
G5, ``metrics.p_<defect>`` for G6); a function judge copies it into a metric row, and
``CorpusMetric``s rank the candidates by agreement with the human value in ``item.refs``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..judges import Row
from ..packs import Item


def auc(preds: list[float], targets: list[float]) -> float:
    """ROC AUC of ``preds`` for binary ``targets`` (1 = positive), ties counted as 1/2
    (Mann–Whitney). NaN when the targets are not binary or one class is missing."""
    p = np.asarray(preds, dtype=float)
    t = np.asarray(targets, dtype=float)
    if not np.all(np.isin(t, (0.0, 1.0))):
        return float("nan")
    pos, neg = p[t == 1.0], p[t == 0.0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    r_pos = r[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def score_judge(metric: str = "pred"):
    """Function judge: the candidate's ``score`` as metric ``metric`` (weight 1)."""
    def judge(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
        v = out.get("score")
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return []
        return [(metric, float(v), 1.0)]
    return judge


def ref_value(key: str, *fallbacks: str):
    """CorpusMetric target: ``item.refs[key]`` (or the first fallback key present)."""
    def target(item: Item) -> float | None:
        for k in (key, *fallbacks):
            v = item.refs.get(k)
            if v is not None:
                return float(v)
        return None
    return target


def binary_ref(key: str):
    """Target for AUC: refs[key] when it is 0/1, else None (item excluded)."""
    def target(item: Item) -> float | None:
        v = item.refs.get(key)
        return float(v) if v in (0, 1, 0.0, 1.0, True, False) else None
    return target
