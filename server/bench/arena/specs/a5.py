"""A5 — forced alignment of known text to audio (word boundaries).

Gold word boundaries come from Praat TextGrids the user corrects (``builders/a5_align_gold.py``
writes stubs; there is no public word-boundary gold for hi/ja). Reference words and hypothesis
words are matched through their characters (normalised, spaces removed), so an aligner may
tokenise differently (ja morphemes vs characters) and still be scored per reference word.

Metrics per reference word boundary (start and end): ``boundary_mae_ms`` (primary),
``within_25ms`` / ``within_50ms`` (share), ``gross_err`` (> 200 ms, gate < 2%), plus
``unaligned`` (reference words the output does not cover). Items that only carry pseudo
references (``refs.words_pseudo``, e.g. another aligner's output) are scored under a
``pseudo_`` prefix and never feed the primary metric.
"""
from __future__ import annotations

import difflib
from typing import Any

from ..judges import Row, Spec, cost, normalize_text, speed
from ..packs import Item


def _chars(words: list[dict[str, Any]]) -> tuple[str, list[int]]:
    """Normalised character string of a word list and the word index of every character."""
    chars, owner = [], []
    for k, w in enumerate(words):
        for ch in normalize_text(str(w.get("word", ""))).replace(" ", ""):
            chars.append(ch)
            owner.append(k)
    return "".join(chars), owner


def project(ref_words: list[dict[str, Any]], hyp_words: list[dict[str, Any]]
            ) -> list[tuple[float, float] | None]:
    """Hypothesis (start, end) for every reference word via character alignment, or None."""
    rs, ro = _chars(ref_words)
    hs, ho = _chars(hyp_words)
    ref2hyp: dict[int, int] = {}
    for blk in difflib.SequenceMatcher(None, rs, hs, autojunk=False).get_matching_blocks():
        for i in range(blk.size):
            ref2hyp[blk.a + i] = blk.b + i
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for ci, wk in enumerate(ro):
        if ci in ref2hyp:
            first.setdefault(wk, ref2hyp[ci])
            last[wk] = ref2hyp[ci]
    spans: list[tuple[float, float] | None] = []
    for wk in range(len(ref_words)):
        if wk not in first:
            spans.append(None)
            continue
        h0, h1 = hyp_words[ho[first[wk]]], hyp_words[ho[last[wk]]]
        try:
            spans.append((float(h0["start"]), float(h1["end"])))
        except (KeyError, TypeError, ValueError):
            spans.append(None)
    return spans


def boundary_rows(ref_words: list[dict[str, Any]], hyp_words: list[dict[str, Any]],
                  prefix: str = "") -> list[Row]:
    ref_words = [w for w in ref_words if normalize_text(str(w.get("word", "")))]
    if not ref_words:
        return []
    spans = project(ref_words, hyp_words or [])
    errs: list[float] = []
    missing = 0
    for w, sp in zip(ref_words, spans):
        if sp is None:
            missing += 1
            continue
        errs += [abs(sp[0] - float(w["start"])) * 1000, abs(sp[1] - float(w["end"])) * 1000]
    rows: list[Row] = [(f"{prefix}unaligned", missing / len(ref_words), float(len(ref_words)))]
    if errs:
        n = float(len(errs))
        rows += [(f"{prefix}boundary_mae_ms", sum(errs) / n, n),
                 (f"{prefix}within_25ms", sum(e <= 25 for e in errs) / n, n),
                 (f"{prefix}within_50ms", sum(e <= 50 for e in errs) / n, n),
                 (f"{prefix}within_100ms", sum(e <= 100 for e in errs) / n, n)]
    # Gross error counts unaligned words as failures too (two boundaries each).
    n_b = 2.0 * len(ref_words)
    gross = sum(e > 200 for e in errs) + 2 * missing
    rows.append((f"{prefix}gross_err", gross / n_b, n_b))
    return rows


def align_rows(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    hyp = out.get("words") or []
    if item.refs.get("words"):
        return boundary_rows(item.refs["words"], hyp)
    if item.refs.get("words_pseudo"):
        return boundary_rows(item.refs["words_pseudo"], hyp, prefix="pseudo_")
    return []


_METRICS = ["boundary_mae_ms", "within_25ms", "within_50ms", "within_100ms", "gross_err",
            "unaligned"]
_HIB = {"boundary_mae_ms": False, "within_25ms": True, "within_50ms": True, "within_100ms": True,
        "gross_err": False, "unaligned": False}

SPEC = Spec(
    id="A5",
    title="Forced alignment of known text",
    judges={"align@1": align_rows, "speed@1": speed, "cost@1": cost},
    primary={"*": "boundary_mae_ms"},
    higher_is_better={**_HIB, **{f"pseudo_{k}": v for k, v in _HIB.items()},
                      "rtfx": True, "cost_usd": False},
    threshold={"boundary_mae_ms": 5.0, "pseudo_boundary_mae_ms": 5.0},
    gates={"gross_err": ("<=", 0.02)},
    secondary=[*[m for m in _METRICS if m != "boundary_mae_ms"],
               *[f"pseudo_{m}" for m in _METRICS], "rtfx", "cost_usd"],
    packs=["a5-align-gold"],
    io="""item.inputs: {audio, text} (text = the verbatim transcript, meta.text_source says
whether it is gold or an ASR hypothesis); item.refs: {words: [{start, end, word}]} (gold from
corrected TextGrids) or {words_pseudo: [...], pseudo_source} (secondary only).
payload: {words: [{start, end, word}]} covering the given text, times in seconds.""",
)
