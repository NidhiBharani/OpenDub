"""A8 — paralinguistic / delivery tagging.

Per-utterance emotion packs (EmoBox-style subsets, ``builders/emobox.py``) rank by **macro-F1**
per language (UAR, and arousal/valence CCC when the pack has dimensional labels, are
secondary); all three are corpus metrics over the items, bootstrapped over speaker groups.
Nonverbal events (laughter, crying, sighs …) with reference times score an event F1 at ±200 ms
onset tolerance (weight 2TP + FP + FN, so the weighted mean is the corpus F1).

Worker payload: ``{segments: [{start, end, label, score}], emotion?, dims?: {arousal, valence,
dominance} in [0, 1], note?}``. The utterance emotion is ``emotion`` if given, else the
highest-scoring segment whose label is an emotion class. Labels are mapped onto the canonical
set below; anything else counts as a wrong prediction.
"""
from __future__ import annotations

from typing import Any

from ..judges import CorpusMetric, Row, Spec, cost, speed
from ..packs import Item

EMOTIONS = ("neutral", "happy", "angry", "sad", "fear", "disgust", "surprise", "calm",
            "contempt", "other")
EMO_SYNONYMS = {
    "neu": "neutral", "neutral": "neutral", "hap": "happy", "happy": "happy", "joy": "happy",
    "happiness": "happy", "excited": "happy", "exc": "happy", "ang": "angry", "angry": "angry",
    "anger": "angry", "sad": "sad", "sadness": "sad", "fea": "fear", "fear": "fear",
    "fearful": "fear", "dis": "disgust", "disgust": "disgust", "disgusted": "disgust",
    "sur": "surprise", "surprise": "surprise", "surprised": "surprise", "calm": "calm",
    "contempt": "contempt", "other": "other", "oth": "other", "unk": "other", "<unk>": "other",
}
NONVERBAL = ("laughter", "crying", "sigh", "scream", "breath", "cough", "gasp", "sniff",
             "groan", "yawn", "throat_clear", "hum")
NV_TOL_S = 0.2


def canon_emotion(label: Any) -> str | None:
    if label is None:
        return None
    key = str(label).strip().lower().split("/")[-1]
    return EMO_SYNONYMS.get(key)


def predicted_emotion(out: dict[str, Any]) -> str | None:
    if out.get("emotion"):
        return canon_emotion(out["emotion"])
    best, best_score = None, float("-inf")
    for s in out.get("segments") or []:
        c = canon_emotion(s.get("label"))
        score = float(s.get("score") if s.get("score") is not None else 0.0)
        if c is not None and score > best_score:
            best, best_score = c, score
    return best


def _canon_nv(label: Any) -> str | None:
    key = str(label).strip().lower().replace(" ", "_")
    key = {"laugh": "laughter", "laughing": "laughter", "giggle": "laughter", "cry": "crying",
           "sob": "crying", "sobbing": "crying", "sighing": "sigh", "screaming": "scream",
           "breathing": "breath", "inhale": "breath", "coughing": "cough"}.get(key, key)
    return key if key in NONVERBAL else None


def event_counts(ref: list[dict[str, Any]], hyp: list[dict[str, Any]],
                 tol: float = NV_TOL_S) -> tuple[int, int, int]:
    """Greedy one-to-one onset matching per label: (TP, FP, FN)."""
    ref = [(float(e["start"]), _canon_nv(e.get("label"))) for e in ref]
    hyp = [(float(e["start"]), _canon_nv(e.get("label"))) for e in hyp]
    ref = [r for r in ref if r[1]]
    hyp = [h for h in hyp if h[1]]
    used: set[int] = set()
    tp = 0
    for t, lab in sorted(ref):
        cands = [(abs(ht - t), k) for k, (ht, hl) in enumerate(hyp)
                 if hl == lab and k not in used and abs(ht - t) <= tol]
        if cands:
            used.add(min(cands)[1])
            tp += 1
    return tp, len(hyp) - tp, len(ref) - tp


def delivery_rows(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    rows: list[Row] = []
    target = canon_emotion(item.refs.get("emotion"))
    if target is not None:
        pred = predicted_emotion(out)
        rows.append(("emo_pred", float(EMOTIONS.index(pred)) if pred else -1.0, 1.0))
        rows.append(("emo_acc", float(pred == target), 1.0))
    dims = out.get("dims") or {}
    for d in ("arousal", "valence", "dominance"):
        if dims.get(d) is not None and item.refs.get(d) is not None:
            rows.append((f"pred_{d}", float(dims[d]), 1.0))
    if item.refs.get("events") is not None:
        hyp = [s for s in out.get("segments") or [] if _canon_nv(s.get("label"))]
        tp, fp, fn = event_counts(item.refs["events"], hyp)
        if 2 * tp + fp + fn:
            rows.append(("nv_event_f1", 2 * tp / (2 * tp + fp + fn), float(2 * tp + fp + fn)))
    return rows


# ------------------------------------------------------------------ corpus metrics

def _classes(targets: list[float]) -> list[int]:
    return sorted({int(t) for t in targets})


def macro_f1(preds: list[float], targets: list[float]) -> float:
    f1s = []
    for c in _classes(targets):
        tp = sum(1 for p, t in zip(preds, targets) if int(p) == c and int(t) == c)
        fp = sum(1 for p, t in zip(preds, targets) if int(p) == c and int(t) != c)
        fn = sum(1 for p, t in zip(preds, targets) if int(p) != c and int(t) == c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1s) / len(f1s) if f1s else float("nan")


def uar(preds: list[float], targets: list[float]) -> float:
    recalls = []
    for c in _classes(targets):
        n = sum(1 for t in targets if int(t) == c)
        recalls.append(sum(1 for p, t in zip(preds, targets) if int(t) == c and int(p) == c) / n)
    return sum(recalls) / len(recalls) if recalls else float("nan")


def ccc(preds: list[float], targets: list[float]) -> float:
    """Lin's concordance correlation coefficient."""
    import numpy as np

    x, y = np.asarray(preds, float), np.asarray(targets, float)
    vx, vy = x.var(), y.var()
    denom = vx + vy + (x.mean() - y.mean()) ** 2
    return float(2 * ((x - x.mean()) * (y - y.mean())).mean() / denom) if denom else float("nan")


def _emo_target(it: Item) -> float | None:
    c = canon_emotion(it.refs.get("emotion"))
    return float(EMOTIONS.index(c)) if c else None


def _dim_target(dim: str):
    def fn(it: Item) -> float | None:
        v = it.refs.get(dim)
        return None if v is None else float(v)
    return fn


SPEC = Spec(
    id="A8",
    title="Paralinguistic / delivery tagging",
    judges={"delivery@1": delivery_rows, "speed@1": speed, "cost@1": cost},
    corpus={"macro_f1": CorpusMetric(pred="emo_pred", target=_emo_target, fn=macro_f1),
            "uar": CorpusMetric(pred="emo_pred", target=_emo_target, fn=uar),
            "ccc_arousal": CorpusMetric(pred="pred_arousal", target=_dim_target("arousal"),
                                        fn=ccc),
            "ccc_valence": CorpusMetric(pred="pred_valence", target=_dim_target("valence"),
                                        fn=ccc)},
    primary={"*": "macro_f1"},
    higher_is_better={"macro_f1": True, "uar": True, "emo_acc": True, "emo_pred": True,
                      "ccc_arousal": True, "ccc_valence": True, "pred_arousal": True,
                      "pred_valence": True, "pred_dominance": True, "nv_event_f1": True,
                      "rtfx": True, "cost_usd": False},
    threshold={"macro_f1": 0.01, "uar": 0.01},
    secondary=["uar", "emo_acc", "ccc_arousal", "ccc_valence", "nv_event_f1", "rtfx",
               "cost_usd"],
    packs=["a8-emobox", "a8-jvnv", "a8-labelled-ih"],
    io="""item.inputs: {audio: one utterance}; item.refs: {emotion} and/or {arousal, valence,
dominance in [0, 1]} and/or {events: [{start, end, label}]}; item.meta.classes: the pack's
emotion classes. payload: {segments: [{start, end, label, score}], emotion?, dims?, note?}.""",
)
