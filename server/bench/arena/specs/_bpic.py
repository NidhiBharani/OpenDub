"""Shared geometry, matching and text helpers for the phase-B (source picture) specs B1–B4.

Conventions every B pack, worker and judge agrees on:

- Boxes are ``[x1, y1, x2, y2]`` **normalised to 0..1** of the frame width/height, so outputs are
  resolution independent. Polygons (``poly: [[x, y], ...]``, also normalised) are reduced to their
  bounding box.
- Times are seconds from the start of the item's video.
- F-measures are emitted as per-item rows ``(f, 2·TP / (n_pred + n_ref), n_pred + n_ref)``: the
  weighted mean of those rows over any set of items is exactly the corpus (micro) F, so the
  leaderboard's group bootstrap resamples the right statistic.

Modules whose name starts with ``_`` are skipped by the spec loader.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from app.pipeline.quality import _distance

Box = list[float]


# ------------------------------------------------------------------ geometry

def as_box(obj: Any) -> Box | None:
    """``[x1, y1, x2, y2]`` from a box, a ``{"box"|"poly"}`` dict or a polygon; None if absent."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return as_box(obj.get("box") if obj.get("box") is not None else obj.get("poly"))
    if not isinstance(obj, (list, tuple)) or not obj:
        return None
    if isinstance(obj[0], (list, tuple)):  # polygon
        xs = [float(p[0]) for p in obj]
        ys = [float(p[1]) for p in obj]
        return [min(xs), min(ys), max(xs), max(ys)]
    if len(obj) == 8:  # flat quad x1,y1,...,x4,y4
        xs, ys = [float(v) for v in obj[0::2]], [float(v) for v in obj[1::2]]
        return [min(xs), min(ys), max(xs), max(ys)]
    if len(obj) >= 4:
        x1, y1, x2, y2 = (float(v) for v in obj[:4])
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return None


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def _center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _inside(p: tuple[float, float], b: Box) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def same_face(a: Box | None, b: Box | None, min_iou: float = 0.3) -> bool:
    """Two face boxes from different detectors denote the same face: IoU ≥ ``min_iou`` or each
    centre lies inside the other box (detectors disagree on how much forehead/chin to include)."""
    if a is None or b is None:
        return False
    return iou(a, b) >= min_iou or (_inside(_center(a), b) and _inside(_center(b), a))


def greedy_match(n_pred: int, n_ref: int, score: Callable[[int, int], float | None],
                 higher_is_better: bool = True) -> list[tuple[int, int]]:
    """One-to-one matching: every admissible (pred, ref) pair ranked by ``score`` (None = not
    admissible), taken greedily. Deterministic (ties broken by index)."""
    pairs = []
    for i in range(n_pred):
        for j in range(n_ref):
            s = score(i, j)
            if s is not None:
                pairs.append((-s if higher_is_better else s, i, j))
    pairs.sort()
    used_p, used_r, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_r:
            used_p.add(i)
            used_r.add(j)
            out.append((i, j))
    return out


def f_row(name: str, tp: float, n_pred: int, n_ref: int) -> list[tuple[str, float, float]]:
    """``2TP/(n_pred+n_ref)`` weighted by ``n_pred+n_ref`` (exact corpus F under weighted mean)."""
    w = n_pred + n_ref
    return [(name, 2.0 * tp / w, float(w))] if w else []


def ratio_row(name: str, num: float, den: float) -> list[tuple[str, float, float]]:
    return [(name, num / den, float(den))] if den else []


# ------------------------------------------------------------------ text

def norm_ocr(text: str) -> str:
    """NFKC + casefold; punctuation and symbols dropped, whitespace collapsed (ICDAR "generic"
    end-to-end matching is case-insensitive and ignores punctuation). Marks (matras, virama,
    dakuten after NFKC) are kept."""
    out = []
    for ch in unicodedata.normalize("NFKC", text).casefold():
        cat = unicodedata.category(ch)
        out.append(" " if cat[0] in "PS" else ch)
    return " ".join("".join(out).split())


def ned(a: str, b: str) -> float:
    """Normalised edit distance in [0, 1] over characters (whitespace ignored)."""
    ca, cb = list(a.replace(" ", "")), list(b.replace(" ", ""))
    if not ca and not cb:
        return 0.0
    return _distance(ca, cb) / max(len(ca), len(cb))


def multiset_f(pred: Iterable[str], ref: Iterable[str]) -> tuple[int, int, int]:
    """(matches, n_pred, n_ref) of two token multisets."""
    p, r = Counter(pred), Counter(ref)
    return sum((p & r).values()), sum(p.values()), sum(r.values())


def split_words(text: str, box: Box | None) -> list[tuple[str, Box | None]]:
    """Split a text line into words, dividing a horizontal box in proportion to character counts
    (the usual approximation when a model reports lines and the reference has words)."""
    words = text.split()
    if len(words) <= 1 or box is None or (box[2] - box[0]) < (box[3] - box[1]):
        return [(text, box)] if words else []
    total = sum(len(w) for w in words) + (len(words) - 1)
    out, x = [], box[0]
    width = box[2] - box[0]
    for k, w in enumerate(words):
        span = width * (len(w) + (1 if k < len(words) - 1 else 0)) / total
        out.append((w, [x, box[1], x + width * len(w) / total, box[3]]))
        x += span
    return out


# ------------------------------------------------------------------ time

def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def nearest(seq: Sequence[Any], t: float, key: Callable[[Any], float]) -> Any | None:
    """Element of ``seq`` whose ``key`` is closest to ``t`` (linear scan; seqs are small)."""
    best, dist = None, float("inf")
    for x in seq:
        d = abs(key(x) - t)
        if d < dist:
            best, dist = x, d
    return best
