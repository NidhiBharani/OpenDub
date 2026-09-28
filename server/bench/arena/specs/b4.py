"""B4 — content type classification (live action / 2D animation / 3D CG / mixed), per title.

Ranked by **title-level macro-F1** over the four classes (a corpus metric: every title is one
item and one bootstrap group). The class decides which lip-sync path a title may take, so the
costly error is calling drawn animation live action (a human-face model on a cel face).

Secondary:
- ``misroute_cost`` — mean routing cost with weights fixed before results are seen
  (``ROUTE_COST``): 2D called live action 1.0, 2D called 3D 0.5, live action called 2D 0.3,
  mixed confused with a pure class 0.2, any abstention 0.1, 3D↔live action 0.1 (a photoreal CG face
  lip-syncs like a real one).
- ``block_recall_2d`` — share of 2D titles not routed to human-face lip sync (predicted 2D,
  mixed or abstain).
- ``sel_acc95`` — accuracy on the 95% of titles the model is most confident about, and ``ece`` —
  expected calibration error (10 bins); both corpus metrics over ``conf_signed`` (confidence,
  negative when wrong), so models without probabilities rank last there, not first.
- ``brier`` — multiclass Brier score when the payload has ``probs``.
"""
from __future__ import annotations

from typing import Any

from ..judges import CorpusMetric, Spec, cost, speed
from ..packs import Item

LABELS = ["live_action", "2d", "3d", "mixed"]
ALIASES = {"live": "live_action", "live-action": "live_action", "live action": "live_action",
           "liveaction": "live_action", "photo": "live_action", "real": "live_action",
           "2d_animation": "2d", "2d animation": "2d", "anime": "2d", "cartoon": "2d",
           "cel": "2d", "drawn": "2d", "3d_animation": "3d", "3d animation": "3d", "cg": "3d",
           "cgi": "3d", "3d cg": "3d", "hybrid": "mixed"}

# (reference, prediction) -> cost; unlisted mismatches cost 0.2, matches 0, abstain 0.1.
ROUTE_COST = {("2d", "live_action"): 1.0, ("2d", "3d"): 0.5, ("live_action", "2d"): 0.3,
              ("3d", "live_action"): 0.1, ("live_action", "3d"): 0.1}
ABSTAIN_COST = 0.1


def canon(label: Any) -> str | None:
    if label is None:
        return None
    s = str(label).strip().lower()
    s = ALIASES.get(s, s)
    return s if s in LABELS else None


def _label_idx(item: Item) -> float | None:
    ref = canon(item.refs.get("label"))
    return float(LABELS.index(ref)) if ref else None


def content_type(item: Item, out: dict[str, Any], row: Any) -> list[tuple[str, float, float]]:
    ref = canon(item.refs.get("label"))
    if ref is None:
        return []
    abstain = bool(out.get("abstain")) or canon(out.get("label")) is None
    pred = None if abstain else canon(out.get("label"))
    probs = {canon(k): float(v) for k, v in (out.get("probs") or {}).items() if canon(k)}
    conf = out.get("confidence")
    if conf is None and pred is not None and pred in probs:
        conf = probs[pred]
    conf = float(conf) if conf is not None else 0.0
    correct = pred == ref
    rows = [("label_idx", float(LABELS.index(pred)) if pred else -1.0, 1.0),
            ("accuracy", float(correct), 1.0),
            ("abstain_rate", float(abstain), 1.0),
            # tiny epsilon keeps the sign of a wrong answer with zero stated confidence
            ("conf_signed", (conf + 1e-6) * (1.0 if correct else -1.0), 1.0),
            ("misroute_cost", ABSTAIN_COST if abstain else
             (0.0 if correct else ROUTE_COST.get((ref, pred), 0.2)), 1.0)]
    if ref == "2d":
        rows.append(("block_recall_2d", float(pred in ("2d", "mixed", None)), 1.0))
    if probs:
        total = sum(probs.values()) or 1.0
        rows.append(("brier", sum((probs.get(k, 0.0) / total - (k == ref)) ** 2
                                  for k in LABELS), 1.0))
    return rows


def macro_f1(preds: list[float], targets: list[float]) -> float:
    """Macro-F1 over the classes present in the targets (an abstention, -1, is wrong for all)."""
    classes = sorted({int(t) for t in targets})
    f1s = []
    for c in classes:
        tp = sum(1 for p, t in zip(preds, targets, strict=True) if int(p) == c and int(t) == c)
        fp = sum(1 for p, t in zip(preds, targets, strict=True) if int(p) == c and int(t) != c)
        fn = sum(1 for p, t in zip(preds, targets, strict=True) if int(p) != c and int(t) == c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1s) / len(f1s) if f1s else float("nan")


def selective_accuracy(coverage: float):
    def fn(preds: list[float], targets: list[float]) -> float:
        ranked = sorted(preds, key=lambda v: -abs(v))
        k = max(1, round(coverage * len(ranked)))
        return sum(v > 0 for v in ranked[:k]) / k
    return fn


def ece(preds: list[float], targets: list[float], bins: int = 10) -> float:
    n = len(preds)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [v for v in preds if lo < min(abs(v), 1.0) <= hi or (b == 0 and abs(v) <= lo)]
        if sel:
            conf = sum(min(abs(v), 1.0) for v in sel) / len(sel)
            acc = sum(v > 0 for v in sel) / len(sel)
            total += len(sel) / n * abs(conf - acc)
    return total


SPEC = Spec(
    id="B4",
    title="Content type classification (live action / 2D / 3D / mixed)",
    judges={"content_type@1": content_type, "speed@1": speed, "cost@1": cost},
    primary={"*": "macro_f1"},
    higher_is_better={"macro_f1": True, "label_idx": True, "accuracy": True,
                      "abstain_rate": False, "conf_signed": True, "misroute_cost": False,
                      "block_recall_2d": True, "brier": False, "sel_acc95": True, "ece": False,
                      "rtfx": True, "cost_usd": False},
    threshold={"macro_f1": 0.02},
    secondary=["misroute_cost", "block_recall_2d", "accuracy", "sel_acc95", "ece", "brier",
               "abstain_rate", "cost_usd"],
    corpus={"macro_f1": CorpusMetric(pred="label_idx", target=_label_idx, fn=macro_f1),
            "sel_acc95": CorpusMetric(pred="conf_signed", target=_label_idx,
                                      fn=selective_accuracy(0.95)),
            "ece": CorpusMetric(pred="conf_signed", target=_label_idx, fn=ece)},
    packs=["b4_titles", "b_local"],
    io="""item.inputs: {video} (a whole title, or a representative cut of it); one item per title,
group = title (or franchise). item.meta: {duration_s, title, source}.
item.refs: {label: live_action|2d|3d|mixed}.
payload: {label | abstain: true, probs?: {label: p}, confidence?: 0..1, frames?: [{t, label,
probs?}], reason?}""",
)
