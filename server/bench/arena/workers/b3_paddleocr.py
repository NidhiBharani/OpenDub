"""B3 worker: PaddleOCR 3.x pipelines (PP-OCRv6 tiers; PP-OCRv5 script models such as
Devanagari).

params:
  det_model: e.g. PP-OCRv6_medium_det
  rec_model: e.g. PP-OCRv6_medium_rec, or rec_model_by_lang: {hi: devanagari_PP-OCRv5_mobile_rec}
  engine: onnxruntime (default; no PaddlePaddle GPU build needed, the only GPU path on aarch64)
          | paddle | transformers
  textline_orientation: false; device: gpu:0 | cpu

Output lines (``rec_texts`` / ``rec_scores`` / ``rec_polys``) with boxes normalised by the frame.
PP-OCRv6 covers zh-Hans/zh-Hant/en/ja + 46 Latin-script languages; there is no v6 Devanagari
model, so ``hi`` needs a v5 script recogniser (``rec_model_by_lang``) or raises Unsupported.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve
from b_common import item_frames, norm_poly


def load(params: dict, lang: str) -> dict:
    by_lang = params.get("rec_model_by_lang") or {}
    rec = by_lang.get(lang) or params.get("rec_model")
    if not rec:
        return {"skip": f"no recognition model for {lang!r}", "params": params}
    if params.get("languages") and lang not in params["languages"]:
        return {"skip": f"{rec} does not read {lang!r}", "params": params}
    from paddleocr import PaddleOCR

    kwargs = {"text_detection_model_name": params.get("det_model"),
              "text_recognition_model_name": rec,
              "use_doc_orientation_classify": False, "use_doc_unwarping": False,
              "use_textline_orientation": bool(params.get("textline_orientation", False))}
    if params.get("engine"):
        kwargs["engine"] = params["engine"]
    if params.get("device"):
        kwargs["device"] = params["device"]
    return {"ocr": PaddleOCR(**kwargs), "params": params, "rec": rec, "skip": None}


def run(state: dict, item: dict, out: Path) -> dict:
    if state["skip"]:
        raise Unsupported(state["skip"])
    texts = []
    for fr in item_frames(item, out):
        res = state["ocr"].predict(fr["path"])[0]
        get = res.get if hasattr(res, "get") else (lambda k, d=None, r=res: getattr(r, k, d))
        polys = get("rec_polys")
        if polys is None:
            polys = get("dt_polys", [])
        for text, score, poly in zip(get("rec_texts", []), get("rec_scores", []), polys,
                                     strict=False):
            if not str(text).strip():
                continue
            texts.append({"text": str(text), "score": float(score), "t": fr["t"],
                          "box": norm_poly(poly, fr["width"], fr["height"])})
    return {"texts": texts}


def describe(state: dict) -> dict:
    from importlib.metadata import version

    p = state["params"]
    return {"det_model": p.get("det_model"), "rec_model": state.get("rec"),
            "engine": p.get("engine"), "paddleocr": version("paddleocr")}


if __name__ == "__main__":
    serve(load, run, describe)
