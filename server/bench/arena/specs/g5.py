"""G5 — lip-sync scorer, meta-evaluated on injected audio offsets (−200…+200 ms).

Candidates: Synchformer (offset + confidence), SyncNet (LSE-C/LSE-D, comparability only), PEAVS
(human-calibrated score) via ``judge_lipsync``, and video-LLM sync raters. Every candidate gives
``desync`` (higher = more out of sync); offset estimators also give a signed ``offset_ms``
(positive = audio lags). Ranked by the monotonic response of desync to |injected offset|
(``srcc_abs``, primary); ``auc_80`` = detecting |offset| ≥ 80 ms against in-sync copies;
``srcc_signed`` and ``offset_abs_err_ms`` only for offset estimators. Rule: never rank
SyncNet-supervised generators with SyncNet.
"""
from __future__ import annotations

import math
from typing import Any

from ..judges import CorpusMetric, Row, Spec, cost, speed
from ..packs import Item
from ._gmeta import auc

DETECT_MS = 80.0


def sync_rows(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    rows: list[Row] = []
    d = out.get("desync")
    if isinstance(d, (int, float)) and math.isfinite(d):
        rows.append(("desync", float(d), 1.0))
    off = out.get("offset_ms")
    if isinstance(off, (int, float)) and math.isfinite(off):
        rows.append(("offset_ms", float(off), 1.0))
        truth = item.refs.get("offset_ms")
        if truth is not None:
            rows.append(("offset_abs_err_ms", abs(float(off) - float(truth)), 1.0))
    return rows


def _abs_offset(item: Item) -> float | None:
    v = item.refs.get("offset_ms")
    return abs(float(v)) if v is not None else None


def _detect(item: Item) -> float | None:
    v = item.refs.get("offset_ms")
    if v is None:
        return None
    v = abs(float(v))
    return 1.0 if v >= DETECT_MS else (0.0 if v == 0 else None)


def _signed(item: Item) -> float | None:
    v = item.refs.get("offset_ms")
    return float(v) if v is not None else None


SPEC = Spec(
    id="G5",
    title="Lip-sync scorer — response to injected A/V offsets",
    judges={"sync@1": sync_rows, "speed@1": speed, "cost@1": cost},
    primary={"*": "srcc_abs"},
    higher_is_better={"srcc_abs": True, "auc_80": True, "srcc_signed": True, "desync": False,
                      "offset_ms": False, "offset_abs_err_ms": False, "rtfx": True,
                      "cost_usd": False},
    threshold={"srcc_abs": 0.02},
    secondary=["auc_80", "srcc_signed", "offset_abs_err_ms", "rtfx", "cost_usd"],
    corpus={"srcc_abs": CorpusMetric("desync", _abs_offset, "spearman"),
            "auc_80": CorpusMetric("desync", _detect, auc),
            "srcc_signed": CorpusMetric("offset_ms", _signed, "spearman")},
    packs=["lipsync_offsets"],
    io="""item.inputs: {video (muxed), audio (the shifted track)}; item.refs: {offset_ms}
(positive = audio lags picture). payload: {desync, offset_ms?, metrics: {...}}.""",
)
