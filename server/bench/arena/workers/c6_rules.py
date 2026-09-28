"""C6 baseline worker: line-wrap only (no condensation) — what OpenDub does without a C6 model.

Each cue's text is re-wrapped greedily into at most ``lines`` lines of at most ``cpl``
characters (breaking at spaces; unspaced scripts break anywhere); nothing is shortened, so CPS
violations remain. The spec's CPS/CPL gates and SubER show what condensation buys over this.

item.inputs: {cues: [{start, end, text}], limits?}. payload: {cues, text}.
"""
from __future__ import annotations

from pathlib import Path

import c_common
from _sdk import serve


def wrap(text: str, cpl: int, unspaced: bool) -> str:
    flat = " ".join(text.split())
    if len(flat) <= cpl:
        return flat
    if unspaced:
        half = (len(flat) + 1) // 2
        return flat[:half] + "\n" + flat[half:]
    words, lines, cur = flat.split(" "), [], ""
    target = (len(flat) + 1) // 2 if len(flat) <= 2 * cpl else cpl
    for w in words:
        if cur and len(cur) + 1 + len(w) > target and len(lines) < 1:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


def load(params: dict, lang: str):
    return {"params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    _, tgt = c_common.direction(state["lang"], item)
    limits = item["inputs"].get("limits") or state["params"].get("limits", {}).get(tgt)
    if not limits:
        raise ValueError("C6 items need inputs.limits {cps, cpl, lines}")
    cues = [{**c, "text": wrap(c["text"], int(limits["cpl"]), tgt in ("ja", "zh"))}
            for c in item["inputs"]["cues"]]
    return {"cues": cues, "text": "\n".join(c["text"].replace("\n", " ") for c in cues)}


if __name__ == "__main__":
    serve(load, run)
