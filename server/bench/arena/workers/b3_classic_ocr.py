"""B3 worker: classical CTC OCR pipelines, the floor every candidate must beat — EasyOCR and
RapidOCR (PP-OCR ONNX exports).

params:
  engine: easyocr | rapidocr
  easyocr: langs_by_lang ({hi: [hi, en], ja: [ja, en], en: [en]}), gpu (true)
  rapidocr: config kwargs passed to ``RapidOCR(params=...)`` (e.g. Rec.lang_type), optional
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve
from b_common import item_frames, norm_poly

EASYOCR_LANGS = {"en": ["en"], "hi": ["hi", "en"], "ja": ["ja", "en"], "zh": ["ch_sim", "en"],
                 "ko": ["ko", "en"]}


def load(params: dict, lang: str) -> dict:
    engine = params.get("engine", "easyocr")
    if engine == "easyocr":
        langs = (params.get("langs_by_lang") or EASYOCR_LANGS).get(lang)
        if not langs:
            return {"skip": f"easyocr: no language list for {lang!r}", "engine": engine}
        import easyocr

        return {"reader": easyocr.Reader(langs, gpu=bool(params.get("gpu", True))),
                "engine": engine, "skip": None, "langs": langs}
    if engine == "rapidocr":
        from rapidocr import RapidOCR

        cfg = params.get("config")
        return {"reader": RapidOCR(params=cfg) if cfg else RapidOCR(), "engine": engine,
                "skip": None}
    raise ValueError(f"unknown engine {engine!r}")


def run(state: dict, item: dict, out: Path) -> dict:
    if state["skip"]:
        raise Unsupported(state["skip"])
    texts = []
    for fr in item_frames(item, out):
        if state["engine"] == "easyocr":
            rows = state["reader"].readtext(fr["path"])
        else:
            res = state["reader"](fr["path"])
            rows = list(zip(res.boxes if res.boxes is not None else [], res.txts or [],
                            res.scores or [], strict=False))
        for pts, text, conf in rows:
            if str(text).strip():
                texts.append({"text": str(text), "score": float(conf), "t": fr["t"],
                              "box": norm_poly(pts, fr["width"], fr["height"])})
    return {"texts": texts}


def describe(state: dict) -> dict:
    from importlib.metadata import version

    pkg = "easyocr" if state["engine"] == "easyocr" else "rapidocr"
    return {"engine": state["engine"], "version": version(pkg), "langs": state.get("langs")}


if __name__ == "__main__":
    serve(load, run, describe)
