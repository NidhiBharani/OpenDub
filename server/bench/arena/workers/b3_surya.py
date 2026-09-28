"""B3 worker: Surya (datalab-to/surya) detection + recognition.

Written against the surya-ocr ≥ 0.14 predictor API (``FoundationPredictor`` →
``RecognitionPredictor`` with a ``DetectionPredictor``); results expose ``text_lines`` with
``text``, ``bbox`` (pixels) and ``confidence``. Surya 2 may rename these: the worker fails loudly
rather than guessing (verify the API before the first run). Weights: modified AI Pubs Open RAIL-M
(free for research/personal use and companies under $5M) — ship_ok false.

params: none required; ``langs`` is ignored by recent Surya (language-agnostic recogniser).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from b_common import item_frames, norm_poly


def load(params: dict, lang: str) -> dict:
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    return {"rec": RecognitionPredictor(FoundationPredictor()), "det": DetectionPredictor(),
            "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    from PIL import Image

    texts = []
    for fr in item_frames(item, out):
        img = Image.open(fr["path"]).convert("RGB")
        pred = state["rec"]([img], det_predictor=state["det"])[0]
        for ln in pred.text_lines:
            if str(ln.text).strip():
                texts.append({"text": str(ln.text), "score": float(ln.confidence or 0.0),
                              "t": fr["t"], "box": norm_poly(ln.bbox, fr["width"],
                                                             fr["height"])})
    return {"texts": texts}


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"surya_ocr": version("surya-ocr")}


if __name__ == "__main__":
    serve(load, run, describe)
