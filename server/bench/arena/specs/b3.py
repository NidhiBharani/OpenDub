"""B3 — on-screen text reading (OCR on video frames).

Ranked by **end-to-end F at IoU ≥ 0.5** (a prediction counts when its box overlaps a reference
box at IoU ≥ 0.5 *and* its normalised text equals the reference text) for languages written with
spaces, where units are words; and by **1 − NED** for unspaced scripts (ja, zh), where units are
text lines and exact line match is too coarse to rank by. Both are always reported.

- Units: for spaced languages every line is split into words, its box divided in proportion to
  character counts (``_bpic.split_words``), so line-level models and word-level references
  compare. Text is NFKC-casefolded, punctuation dropped (ICDAR "generic" matching).
- Don't-care regions (``care: false``, e.g. TextOCR "." / ICDAR "###" illegible text): predictions
  that overlap them are neither true nor false positives.
- 1 − NED follows ICDAR-2019 ReCTS/LSVT: Σ(1 − NED) over IoU-matched pairs / max(#ref, #pred).
- ``text_f`` / ``char_f``: location-free multiset F over words (spaced) / characters — the only
  fair view of VLMs that return no boxes, and a check on reading quality independent of geometry.

Video items name the annotated frames in ``inputs.times``; references and predictions carry
``t`` and are compared per frame (a prediction is assigned to the nearest annotated time within
0.5 s; predictions far from any annotated frame are ignored).
"""
from __future__ import annotations

from typing import Any

from ..judges import UNSPACED, Spec, cost, speed
from ..packs import Item
from ._bpic import (
    Box,
    as_box,
    f_row,
    greedy_match,
    iou,
    multiset_f,
    ned,
    norm_ocr,
    ratio_row,
    split_words,
)

IOU_MIN = 0.5
FRAME_TOL = 0.5  # s


def _units(entries: list[dict[str, Any]], spaced: bool, ref: bool
           ) -> list[tuple[str, Box | None, bool]]:
    out = []
    for e in entries:
        raw = str(e.get("text") or "")
        care = bool(e.get("care", True)) and raw.strip() not in ("###", ".", "")
        text = norm_ocr(raw)
        if not text and not (ref and not care):
            continue
        box = as_box(e)
        if spaced and care:
            out += [(w, b, True) for w, b in split_words(text, box)]
        else:
            out.append((text.replace(" ", "") if not spaced else text, box, care))
    return out


def _frames(refs: list[dict[str, Any]], preds: list[dict[str, Any]]):
    """Pairs of (ref entries, pred entries) per annotated frame."""
    times = sorted({float(r["t"]) for r in refs if r.get("t") is not None})
    if not times:
        return [(refs, preds)]
    by_t: dict[float, tuple[list, list]] = {t: ([], []) for t in times}
    for r in refs:
        if r.get("t") is not None:
            by_t[float(r["t"])][0].append(r)
    for p in preds:
        if p.get("t") is None:
            continue
        t = min(times, key=lambda x: abs(x - float(p["t"])))
        if abs(t - float(p["t"])) <= FRAME_TOL:
            by_t[t][1].append(p)
    return list(by_t.values())


def ocr_scores(item: Item, out: dict[str, Any], row: Any) -> list[tuple[str, float, float]]:
    refs = item.refs.get("texts")
    if refs is None:
        return []
    spaced = item.lang not in UNSPACED
    tp_e2e = tp_det = n_pred = n_ref = 0
    ned_sum, ned_n = 0.0, 0
    words_p: list[str] = []
    words_r: list[str] = []
    chars_p: list[str] = []
    chars_r: list[str] = []
    for ref_entries, pred_entries in _frames(refs, out.get("texts") or []):
        R = _units(ref_entries, spaced, ref=True)
        P = _units(pred_entries, spaced, ref=False)
        care = [u for u in R if u[2]]
        dont = [u for u in R if not u[2]]
        # predictions on don't-care regions are ignored
        if dont:
            def on_dc(i: int, j: int, P=P, dont=dont) -> float | None:
                if P[i][1] is None or dont[j][1] is None:
                    return None
                s = iou(P[i][1], dont[j][1])
                return s if s >= IOU_MIN else None

            dc = greedy_match(len(P), len(dont), on_dc)
            drop = {i for i, _ in dc}
            P = [u for k, u in enumerate(P) if k not in drop]

        def det(i: int, j: int, P=P, care=care) -> float | None:
            if P[i][1] is None or care[j][1] is None:
                return None
            s = iou(P[i][1], care[j][1])
            return s if s >= IOU_MIN else None

        def e2e(i: int, j: int, P=P, care=care) -> float | None:
            return det(i, j) if P[i][0] == care[j][0] else None

        pairs = greedy_match(len(P), len(care), det)
        tp_det += len(pairs)
        tp_e2e += len(greedy_match(len(P), len(care), e2e))
        ned_sum += sum(1.0 - ned(care[j][0], P[i][0]) for i, j in pairs)
        ned_n += max(len(care), len(P))
        n_pred += len(P)
        n_ref += len(care)
        for text, _b, _c in P:
            words_p += text.split() if spaced else [text]
            chars_p += list(text.replace(" ", ""))
        for text, _b, _c in care:
            words_r += text.split() if spaced else [text]
            chars_r += list(text.replace(" ", ""))

    rows = f_row("e2e_f", tp_e2e, n_pred, n_ref)
    rows += f_row("det_f", tp_det, n_pred, n_ref)
    rows += ratio_row("one_minus_ned", ned_sum, ned_n)
    if spaced:
        rows += f_row("text_f", *multiset_f(words_p, words_r))
    rows += f_row("char_f", *multiset_f(chars_p, chars_r))
    return rows


SPEC = Spec(
    id="B3",
    title="On-screen text reading",
    judges={"ocr@1": ocr_scores, "speed@1": speed, "cost@1": cost},
    primary={"*": "e2e_f", "ja": "one_minus_ned", "zh": "one_minus_ned"},
    higher_is_better={"e2e_f": True, "det_f": True, "one_minus_ned": True, "text_f": True,
                      "char_f": True, "rtfx": True, "cost_usd": False},
    threshold={"e2e_f": 0.01, "one_minus_ned": 0.01},
    secondary=["e2e_f", "one_minus_ned", "det_f", "text_f", "char_f", "cost_usd"],
    packs=["b3_synth_overlay", "textocr", "mlt19", "b_local"],
    io="""item.inputs: {image} (a still frame) or {video, times: [t, ...]} (annotated frames);
item.refs: {texts: [{text, box: [x1,y1,x2,y2] normalised | poly: [[x,y],...], care?: bool,
t?: seconds, role?: title|lower_third|caption|sign|credit|ui}], granularity: word|line}.
payload: {texts: [{text, box? | poly?, score?, t?}]} (boxes normalised 0..1; boxless entries
score only on text_f/char_f)""",
)
