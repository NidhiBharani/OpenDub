"""Judge worker (phase F4): OCR round trip over replaced text boxes (EasyOCR, Apache-2.0).

item.inputs: {video, boxes: [{box: [x0, y0, x1, y1], start, end}], src_lang}. For each box the
frame at mid-span is cropped (box + ``margin`` × its size, so a re-rendered string that grew a
little is still read) and read with EasyOCR for {target, source, en} — so leftover source text
is recognised too. Returns ``{"texts": [...]}``: one string per box (reading order: top-to-
bottom, left-to-right; CJK joined without spaces). The spec computes CER and source leakage.
params: margin (0.15), gpu (true). The pack language (target) comes from the job.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve

EASYOCR_CODE = {"en": "en", "hi": "hi", "ja": "ja"}


def load(params: dict, lang: str):
    import easyocr

    langs = {EASYOCR_CODE.get(lang, lang), "en"}
    return {"params": params, "lang": lang, "readers": {}, "easyocr": easyocr, "base": langs}


def _reader(state: dict, src_lang: str | None):
    langs = set(state["base"])
    if src_lang:
        langs.add(EASYOCR_CODE.get(src_lang, src_lang))
    # EasyOCR allows ja+en and hi+en but not ja+hi: fall back to the target's reader.
    if {"ja", "hi"} <= langs:
        langs = set(state["base"])
    key = tuple(sorted(langs))
    if key not in state["readers"]:
        state["readers"][key] = state["easyocr"].Reader(list(key),
                                                        gpu=bool(state["params"].get("gpu", True)))
    return state["readers"][key]


def run(state: dict, item: dict, out: Path) -> dict:
    from f_frames import read_frames

    inp = item["inputs"]
    reader = _reader(state, inp.get("src_lang"))
    frames, fps = read_frames(inp["video"])
    n, h, w, _ = frames.shape
    margin = float(state["params"].get("margin", 0.15))
    texts = []
    for b in inp["boxes"]:
        x0, y0, x1, y1 = (int(v) for v in b["box"])
        mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
        i = min(n - 1, int((b["start"] + b["end"]) / 2 * fps))
        crop = frames[i][max(0, y0 - my):min(h, y1 + my), max(0, x0 - mx):min(w, x1 + mx)]
        res = reader.readtext(crop, detail=1, paragraph=False)
        res.sort(key=lambda r: (round(min(p[1] for p in r[0]) / 20), min(p[0] for p in r[0])))
        sep = "" if state["lang"] == "ja" else " "
        texts.append(sep.join(r[1] for r in res).strip())
    return {"texts": texts}


def describe(state: dict) -> dict:
    return {"engine": "easyocr", "version": getattr(state["easyocr"], "__version__", None)}


if __name__ == "__main__":
    serve(load, run, describe)
