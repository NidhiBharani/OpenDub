"""C1 worker: Segment any Text (SaT, wtpsplit) boundary probabilities driving a pause/duration
split search over the timed word list.

The model is half the stage; the other half is the search (docs/model-candidates-by-compute.md
C1): every candidate boundary after word *k* scores
``w_sat·p_sat(k) + w_pause·min(pause_k, 0.6)/0.6 − line_cost`` and a DP picks the boundary set
with the best total, subject to every line lasting at most ``max_line_seconds`` and never
spanning a pause of ``hard_pause`` s or more. (The HW-TSC constituent cost needs a parser and is
not included; see the disabled ``hwtsc-parse-pause`` candidate.)

params: model (sat-12l-sm | sat-3l-sm | sat-12l …), style_or_domain + language (SaT LoRA
modules, optional), device (cuda), half (true), w_sat (1.0), w_pause (1.0), line_cost (0.6),
max_line_seconds (12), hard_pause (1.0).

item.inputs: {words: [{start, end, word}]}. payload: {lines, text, boundary_probs}.
"""
from __future__ import annotations

from pathlib import Path

import c_common
from _sdk import serve


def word_char_ends(words: list[dict], lang: str) -> tuple[str, list[int]]:
    """Joined text and the char offset just after each word."""
    joiner = "" if lang in ("ja", "zh", "th") else " "
    text, ends = "", []
    for i, w in enumerate(words):
        if i:
            text += joiner
        text += w["word"].strip()
        ends.append(len(text))
    return text, ends


def boundary_probs(char_probs: list[float], ends: list[int]) -> list[float]:
    """P(line ends after word k): max char probability over the word's last char and the
    following separator."""
    out = []
    for e in ends:
        window = char_probs[max(0, e - 1):e + 1]
        out.append(float(max(window)) if window else 0.0)
    return out


def split_search(words: list[dict], probs: list[float], params: dict) -> list[int]:
    """Indices of the last word of every line (DP over boundaries; see module docstring)."""
    n = len(words)
    if n == 0:
        return []
    w_sat, w_pause = float(params.get("w_sat", 1.0)), float(params.get("w_pause", 1.0))
    cost = float(params.get("line_cost", 0.6))
    max_s = float(params.get("max_line_seconds", 12.0))
    hard = float(params.get("hard_pause", 1.0))
    pause = [max(0.0, words[k + 1]["start"] - words[k]["end"]) if k + 1 < n else 0.0
             for k in range(n)]

    def gain(k: int) -> float:
        return w_sat * probs[k] + w_pause * min(pause[k], 0.6) / 0.6 - cost

    neg = float("-inf")
    best = [neg] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        end_gain = gain(j - 1) if j < n else 0.0
        for i in range(j - 1, -1, -1):
            if i < j - 1 and pause[i] >= hard:
                break                              # the line would span a hard pause
            if i < j - 1 and words[j - 1]["end"] - words[i]["start"] > max_s:
                break
            if best[i] == neg:
                continue
            v = best[i] + end_gain
            if v > best[j]:
                best[j], back[j] = v, i
        if best[j] == neg:                         # forced single-word line
            best[j], back[j] = best[j - 1] + end_gain, j - 1
    ends, j = [], n
    while j > 0:
        ends.append(j - 1)
        j = back[j]
    return sorted(ends)


def load(params: dict, lang: str):
    from wtpsplit import SaT

    kwargs = {k: params[k] for k in ("style_or_domain", "language") if params.get(k)}
    if "style_or_domain" in kwargs and "language" not in kwargs:
        kwargs["language"] = c_common.direction(lang)[0]   # LoRA modules are per language
    sat = SaT(params.get("model", "sat-12l-sm"), **kwargs)
    if params.get("half", True):
        sat.half()
    sat.to(params.get("device", "cuda"))
    return {"sat": sat, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    src, _ = c_common.direction(state["lang"], item)
    words = item["inputs"]["words"]
    text, ends = word_char_ends(words, src)
    char_probs = [float(x) for x in state["sat"].predict_proba(text)]
    probs = boundary_probs(char_probs, ends)
    lines = c_common.lines_from_ends(words, split_search(words, probs, state["params"]), src)
    return {"lines": lines, "text": "\n".join(ln["text"] for ln in lines),
            "boundary_probs": [round(p, 4) for p in probs]}


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"model": state["params"].get("model", "sat-12l-sm"), "wtpsplit": version("wtpsplit")}


if __name__ == "__main__":
    serve(load, run, describe)
