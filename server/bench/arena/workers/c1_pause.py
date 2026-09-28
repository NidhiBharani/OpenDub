"""C1 baseline worker: OpenDub's pause splitter (``app.pipeline.segmentation.resegment``).

Splits the timed word list where the speaker pauses (≥ ``min_pause`` s) or a sentence ends
before a short breath, and caps lines at ``max_line_seconds`` by cutting at the widest internal
gap. No model; runs in the ``server`` env (imports the app).

item.inputs: {words: [{start, end, word}], src_lang?}. payload: {lines: [{start_word, end_word,
start, end, text}], text}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import c_common
from _sdk import serve

SERVER_DIR = Path(__file__).resolve().parents[3]


def load(params: dict, lang: str):
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    from app.models import ASRSegment, Word
    from app.pipeline.segmentation import resegment

    return {"params": params, "lang": lang, "resegment": resegment, "Word": Word,
            "ASRSegment": ASRSegment}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    src, _ = c_common.direction(state["lang"], item)
    words = item["inputs"]["words"]
    unspaced = src in ("ja", "zh", "th")
    objs = [state["Word"](start=w["start"], end=w["end"],
                          text=(w["word"] if unspaced or i == 0 else " " + w["word"].lstrip()))
            for i, w in enumerate(words)]
    seg = state["ASRSegment"](start=words[0]["start"], end=words[-1]["end"],
                              text="".join(o.text for o in objs), words=objs)
    lines = state["resegment"]([seg], min_pause=float(p.get("min_pause", 0.35)),
                               max_line_seconds=float(p.get("max_line_seconds", 12.0)))
    # recover word indices by scanning forward for each line's last word (timing + text)
    ends, k = [], 0
    for line in lines:
        if not line.words:
            continue
        last = line.words[-1]
        while k < len(objs) and not (objs[k].start == last.start and objs[k].end == last.end
                                     and objs[k].text == last.text):
            k += 1
        if k < len(objs):
            ends.append(k)
    out_lines = c_common.lines_from_ends(words, ends, src)
    return {"lines": out_lines, "text": "\n".join(ln["text"] for ln in out_lines)}


if __name__ == "__main__":
    serve(load, run)
