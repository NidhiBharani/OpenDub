"""A3 — speech / music / singing regions.

Multi-label regions (speech, music and singing can coexist). Scored sed_eval-style on a 100 ms
segment grid per class:

- ``f1_<class>`` with weight ``2TP + FP + FN``, so the weighted mean over items is exactly the
  corpus F1 of that class;
- ``macro_f1`` (primary): per item, the mean F1 over classes active in the reference or the
  prediction, weighted by item duration;
- ``speech_recall`` (gate ≥ 0.97, weight TP + FN = exact corpus recall);
- ``false_speech_per_h_music``: seconds of predicted speech inside reference music-only time,
  per hour of music-only time.
"""
from __future__ import annotations

from typing import Any

from ..judges import Row, Spec, cost, speed
from ..packs import Item

CLASSES = ("speech", "music", "singing")
RES_S = 0.1

# Vendor / model label → canonical class (None = ignore). Workers should already emit canonical
# labels; this keeps the judge tolerant of the obvious synonyms.
SYNONYMS = {
    "speech": "speech", "male": "speech", "female": "speech", "voice": "speech",
    "conversation": "speech", "narration": "speech", "dialogue": "speech", "talk": "speech",
    "speaking": "speech",
    "music": "music", "musical instrument": "music", "instrumental": "music", "song": "singing",
    "singing": "singing", "sing": "singing", "choir": "singing", "vocal music": "singing",
    "rapping": "singing", "humming": "singing",
    "noenergy": None, "noise": None, "silence": None, "other": None,
}


def canon_label(label: str) -> str | None:
    key = str(label).strip().lower().replace("_", " ")
    if key in SYNONYMS:
        return SYNONYMS[key]
    return key if key in CLASSES else None


def activity(segments: list[dict[str, Any]], n: int, classes=CLASSES, res: float = RES_S,
             min_score: float = 0.0) -> dict[str, list[bool]]:
    """Per-class boolean activity on an ``n``-frame grid (frame i covers [i·res, (i+1)·res))."""
    act = {c: [False] * n for c in classes}
    for s in segments or []:
        c = canon_label(s.get("label", ""))
        if c not in act or float(s.get("score", 1.0) if s.get("score") is not None else 1.0) \
                < min_score:
            continue
        a = max(0, round(float(s["start"]) / res))
        b = min(n, round(float(s["end"]) / res))
        for i in range(a, b):
            act[c][i] = True
    return act


def _counts(ref: list[bool], hyp: list[bool]) -> tuple[int, int, int]:
    tp = sum(1 for r, h in zip(ref, hyp) if r and h)
    fp = sum(1 for r, h in zip(ref, hyp) if h and not r)
    fn = sum(1 for r, h in zip(ref, hyp) if r and not h)
    return tp, fp, fn


def region_rows(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    ref_segs = item.refs.get("segments")
    if ref_segs is None:
        return []
    classes = tuple(item.meta.get("classes") or CLASSES)
    dur = float(item.meta.get("duration_s") or max(
        [float(s["end"]) for s in ref_segs] + [0.0]))
    n = max(1, round(dur / RES_S))
    ref = activity(ref_segs, n, classes)
    hyp = activity(out.get("segments") or [], n, classes)
    rows: list[Row] = []
    f1s = []
    for c in classes:
        tp, fp, fn = _counts(ref[c], hyp[c])
        denom = 2 * tp + fp + fn
        if denom:
            f1 = 2 * tp / denom
            rows.append((f"f1_{c}", f1, float(denom)))
            f1s.append(f1)
        if c == "speech" and tp + fn:
            rows.append(("speech_recall", tp / (tp + fn), float(tp + fn)))
    if f1s:
        rows.append(("macro_f1", sum(f1s) / len(f1s), dur))
    if "music" in ref and "speech" in ref:
        music_only = [m and not s for m, s in zip(ref["music"], ref["speech"])]
        mo = sum(music_only)
        if mo:
            false_s = sum(1 for m, h in zip(music_only, hyp["speech"]) if m and h) * RES_S
            hours = mo * RES_S / 3600
            rows.append(("false_speech_per_h_music", false_s / hours, hours))
    return rows


SPEC = Spec(
    id="A3",
    title="Speech / music / singing regions",
    judges={"regions@1": region_rows, "speed@1": speed, "cost@1": cost},
    primary={"*": "macro_f1"},
    higher_is_better={"macro_f1": True, "f1_speech": True, "f1_music": True, "f1_singing": True,
                      "speech_recall": True, "false_speech_per_h_music": False, "rtfx": True,
                      "cost_usd": False},
    threshold={"macro_f1": 0.01},
    gates={"speech_recall": (">=", 0.97)},
    secondary=["f1_speech", "f1_music", "f1_singing", "speech_recall",
               "false_speech_per_h_music", "rtfx", "cost_usd"],
    packs=["a3-regions-synth", "a3-regions-ih"],
    io="""item.inputs: {audio}; item.refs: {segments: [{start, end, label}]} with labels in
speech | music | singing (overlapping regions allowed); item.meta.classes optionally narrows the
scored classes (e.g. a pack without singing). payload: {segments: [{start, end, label, score}]}
with canonical labels (multi-label: emit one segment per active class).""",
)
