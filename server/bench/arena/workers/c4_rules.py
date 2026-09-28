"""C4 baseline worker: rule checks as a quality estimate (no model).

The rules that catch the failures metrics miss (docs/model-candidates-by-compute.md C4):
empty or untranslated output (copy of the source), wrong script for the target language,
runaway repetition, and a spoken-length ratio far from the source (per-language speaking-rate
model, not characters). Score = 100 − penalties, clipped to [0, 100]; higher is better.

item.inputs: {source, hypothesis, src_lang, tgt_lang}. payload: {metrics: {qe_rules}, flags}.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import c_common
import c_speech_rate as rate
from _sdk import serve

SCRIPT = {"hi": (0x0900, 0x097F), "ja": (0x3040, 0x9FFF), "en": (0x0041, 0x024F)}


def script_share(text: str, lang: str) -> float:
    lo, hi = SCRIPT.get(lang, (0, 0x10FFFF))
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def rule_score(source: str, hyp: str, src: str, tgt: str) -> tuple[float, list[str]]:
    flags: list[str] = []
    score = 100.0
    h = hyp.strip()
    if not h:
        return 0.0, ["empty"]
    if h == source.strip() and src != tgt:
        return 5.0, ["copy_of_source"]
    share = script_share(h, tgt)
    if share < 0.5:
        score -= 60 * (1 - share)
        flags.append("wrong_script")
    toks = re.findall(r"\w+", h.lower()) if tgt not in ("ja", "zh") else list(h)
    if len(toks) >= 8:
        grams = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
        rep = 1 - len(set(grams)) / len(grams)
        if rep > 0.3:
            score -= 50 * rep
            flags.append("repetition")
    s_sec, h_sec = rate.predicted_seconds(source, src), rate.predicted_seconds(h, tgt)
    if s_sec > 0 and h_sec > 0:
        dev = abs(math.log(h_sec / s_sec))
        if dev > math.log(1.5):
            score -= min(40.0, 40 * (dev - math.log(1.5)) / math.log(2))
            flags.append("length_ratio")
    return max(0.0, min(100.0, score)), flags


def load(params: dict, lang: str):
    return {"params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    src, tgt = c_common.direction(state["lang"], item)
    inp = item["inputs"]
    score, flags = rule_score(inp["source"], inp["hypothesis"], src, tgt)
    return {"metrics": {"qe_rules": score}, "flags": flags}


if __name__ == "__main__":
    serve(load, run)
