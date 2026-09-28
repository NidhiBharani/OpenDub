"""C1 — dub-line segmentation: regroup a timed word list into speakable dub lines.

Primary (docs/plans/model-ranking.md appendix, row C1): **% TTS-fittable lines** — the share of
predicted lines that

1. fit a slot: last between :data:`MIN_S` and :data:`MAX_S` seconds, and are not crammed
   (spoken-unit rate ≤ 1.6× the language's dubbing rate, so the ASR timing is plausible);
2. end at a pause of ≥ 150 ms (the last line of the item is exempt);
3. do not split a constituent. Without a parser this is approximated: the boundary follows
   punctuation, or lies within ±1 word of a gold boundary, or sits on a pause ≥ 300 ms.

Gate: every word covered exactly once (``coverage`` ≥ 0.99). Secondary: boundary F1 (±1 word)
against gold line ends where the pack has them, line count vs gold, mean line length.
Downstream C2 retry rate is measured in the cascade, not here.
"""
from __future__ import annotations

from typing import Any

from ..judges import Row, Spec, cost, speed
from ..packs import Item
from .c2 import RATES, rate, src_lang

MIN_S, MAX_S = 0.6, 12.0
PAUSE_OK, PAUSE_CLEAN = 0.15, 0.30
CRAM = 1.6
_PUNCT = tuple(".,!?;:…。、！？，；：」』)\"'”’-—")


def _gold_ends(item: Item) -> set[int] | None:
    if "ends" in item.refs:
        return {int(e) for e in item.refs["ends"]}
    if "lines" in item.refs:
        return {int(ln["end_word"]) - 1 for ln in item.refs["lines"]}
    return None


def lines_of(out: dict[str, Any], n_words: int) -> list[dict]:
    lines = [ln for ln in out.get("lines") or [] if "start_word" in ln and "end_word" in ln]
    return [ln for ln in lines if 0 <= int(ln["start_word"]) < int(ln["end_word"]) <= n_words]


def segmentation(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    words = item.inputs.get("words") or []
    n = len(words)
    if not n:
        return []
    lang = src_lang(item)
    lines = lines_of(out, n)
    covered = [0] * n
    for ln in lines:
        for k in range(int(ln["start_word"]), int(ln["end_word"])):
            covered[k] += 1
    coverage = sum(c == 1 for c in covered) / n
    gold = _gold_ends(item)
    joiner = "" if lang in ("ja", "zh") else " "
    fittable = 0
    for idx, ln in enumerate(lines):
        a, b = int(ln["start_word"]), int(ln["end_word"])
        seg = words[a:b]
        dur = seg[-1]["end"] - seg[0]["start"]
        text = joiner.join(w["word"] for w in seg)
        units = rate.spoken_units(text, lang)
        fits = MIN_S <= dur <= MAX_S and units / max(dur, 1e-3) <= CRAM * rate.rate_for(lang,
                                                                                     RATES)
        last = idx == len(lines) - 1 or b >= n
        pause = 0.0 if b >= n else max(0.0, words[b]["start"] - seg[-1]["end"])
        pause_ok = last or pause >= PAUSE_OK
        clean = (last or seg[-1]["word"].strip().endswith(_PUNCT) or pause >= PAUSE_CLEAN
                 or (gold is not None and any(abs((b - 1) - g) <= 1 for g in gold)))
        fittable += int(fits and pause_ok and clean)
    rows: list[Row] = [("coverage", coverage, 1.0)]
    if lines:
        rows += [("fittable", fittable / len(lines), float(len(lines))),
                 ("mean_line_s", sum(ln["end"] - ln["start"] for ln in lines) / len(lines),
                  float(len(lines)))]
    if gold is not None:
        pred = {int(ln["end_word"]) - 1 for ln in lines} - {n - 1}
        g = gold - {n - 1}
        tp_p = sum(any(abs(p - x) <= 1 for x in g) for p in pred)
        tp_g = sum(any(abs(x - p) <= 1 for p in pred) for x in g)
        prec = tp_p / len(pred) if pred else float(not g)
        rec = tp_g / len(g) if g else float(not pred)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows += [("boundary_f1", f1, 1.0),
                 ("line_count_ratio", len(lines) / max(1, len(g) + 1), 1.0)]
    return rows


SPEC = Spec(
    id="C1",
    title="Dub-line segmentation",
    judges={"segmentation@1": segmentation, "speed@1": speed, "cost@1": cost},
    primary={"*": "fittable"},
    higher_is_better={"fittable": True, "coverage": True, "boundary_f1": True,
                      "line_count_ratio": True, "mean_line_s": False, "rtfx": True,
                      "cost_usd": False},
    threshold={"fittable": 0.01, "boundary_f1": 0.01},
    secondary=["boundary_f1", "coverage", "line_count_ratio", "mean_line_s", "cost_usd"],
    gates={"coverage": (">=", 0.99)},
    packs=["scene_c1", "c1_srt_words"],
    io="""pack lang = source language (ja, en, hi).
item.inputs: {words: [{start, end, word}], text?, src_lang, audio?};
item.refs: {ends?: [gold line-end word indices] | lines?: [{start_word, end_word, …}]};
item.meta: {duration_s, src_lang, source}.
payload: {lines: [{start_word, end_word (exclusive), start, end, text}], text}.
LLM workers run with params.task = segment.""",
)
