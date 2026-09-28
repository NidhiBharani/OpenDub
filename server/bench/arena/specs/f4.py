"""F4 — on-screen text replacement.

Pack (``f4_overlay_synth``): procedurally generated backgrounds (or user clips) with source-
language text burned in by PIL at known boxes, so the clean plate and the intended translation
are known exactly. Pack language = the *target* language of the replacement text.

- ``ocr_cer`` (primary, ↓): an OCR round trip (EasyOCR, ``f_judge_ocr.py``) over each replaced
  box at mid-span, CER against the intended translation. It catches the dominant failure of
  diffusion editors: plausible but wrong glyphs.
- ``src_leak`` (↓): share of the source string still legible in the box (1 − CER vs source).
- ``psnr_outside`` (gate ≥ 30 dB): the picture outside the text boxes must stay untouched.
- ``box_flicker`` (↓): temporal std of the replaced region relative to the clean plate's.

The overlay/template-export method is a real contender here (research verdict: ship it first).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.pipeline.quality import _distance

from ..hardware import Requires
from ..judges import ModelJudge, Row, Spec, cost, normalize_text, output_file, speed
from ..packs import Item
from .f1 import psnr, read_gray, video_size

JUDGE_H = 360


def _scaled_boxes(item: Item, w: int, h: int, sw: int, sh: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for t in item.inputs.get("texts") or []:
        x0, y0, x1, y1 = t["box"]
        boxes.append((int(x0 * w / sw), int(y0 * h / sh), int(np.ceil(x1 * w / sw)),
                      int(np.ceil(y1 * h / sh))))
    return boxes


def collateral(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    clean, video = item.refs.get("clean_video"), output_file(out, "video")
    if not clean or not video or not Path(clean).exists():
        return []
    sw, sh = video_size(clean)
    h = JUDGE_H
    w = max(2, round(sw * h / sh / 2) * 2)
    a, b = read_gray(clean, width=w, height=h), read_gray(video, width=w, height=h)
    n = min(len(a), len(b))
    if n == 0:
        return [("frames_missing", 1.0, 1.0)]
    mask = np.ones((h, w), bool)
    boxes = _scaled_boxes(item, w, h, sw, sh)
    for x0, y0, x1, y1 in boxes:
        pad = 4
        mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = False
    rows: list[Row] = [("psnr_outside", psnr(a[:n][:, mask], b[:n][:, mask]), float(n))]
    flick = []
    for x0, y0, x1, y1 in boxes:
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        ra = a[:n, y0:y1, x0:x1].reshape(n, -1).mean(axis=1)
        rb = b[:n, y0:y1, x0:x1].reshape(n, -1).mean(axis=1)
        flick.append(max(0.0, float(np.std(np.diff(rb)) - np.std(np.diff(ra)))))
    if flick:
        rows.append(("box_flicker", float(np.mean(flick)), float(len(flick))))
    return rows


def cer(ref: str, hyp: str) -> float:
    r = normalize_text(ref).replace(" ", "")
    h = normalize_text(hyp).replace(" ", "")
    return min(1.0, _distance(list(r), list(h)) / len(r)) if r else 0.0


def ocr_judge(*, version: int = 1) -> ModelJudge:
    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        video, texts = output_file(out, "video"), item.inputs.get("texts")
        if not video or not texts:
            return None
        return {"video": video, "src_lang": item.meta.get("src_lang"),
                "boxes": [{"box": t["box"], "start": t["start"], "end": t["end"]}
                          for t in texts]}

    def to_rows(item: Item, payload: dict[str, Any], out: dict[str, Any]) -> list[Row]:
        texts = item.inputs.get("texts") or []
        read = payload.get("texts") or []
        rows: list[Row] = []
        for t, hyp in zip(texts, read, strict=False):
            tgt, src = t.get("tgt_text", ""), t.get("src_text", "")
            n = float(len(normalize_text(tgt).replace(" ", "")) or 1)
            rows.append(("ocr_cer", cer(tgt, hyp or ""), n))
            if src:
                rows.append(("src_leak", max(0.0, 1.0 - cer(src, hyp or "")), 1.0))
            rows.append(("ocr_exact", float(normalize_text(tgt) == normalize_text(hyp or "")),
                         1.0))
        return rows

    return ModelJudge(
        id=f"ocr.easyocr@{version}", worker="f_judge_ocr", env="f_judge_ocr",
        params={"engine": "easyocr", "margin": 0.15}, lang_param=True,
        requires=Requires.parse({"gpu": True, "vram_gb": 2}), languages=["en", "hi", "ja"],
        inputs=inputs, to_rows=to_rows)


SPEC = Spec(
    id="F4",
    title="On-screen text replacement",
    judges={"collateral@1": collateral, "speed@1": speed, "cost@1": cost},
    primary={"*": "ocr_cer"},
    higher_is_better={"ocr_cer": False, "src_leak": False, "ocr_exact": True,
                      "psnr_outside": True, "box_flicker": False, "frames_missing": False,
                      "rtfx": True, "cost_usd": False},
    threshold={"ocr_cer": 0.02},
    secondary=["ocr_exact", "src_leak", "psnr_outside", "box_flicker", "rtfx", "cost_usd"],
    model_judges=lambda lang: [ocr_judge()],
    gates={"psnr_outside": (">=", 30.0)},
    packs=["f4_overlay_synth"],
    io="""item.inputs: {video, texts: [{box: [x0, y0, x1, y1] (source pixels), start, end,
src_text, src_lang, tgt_text, tgt_lang, style: {font_px, color, bg?}}]};
item.refs: {clean_video: the plate without text}; item.meta: {src_lang, width, height, fps}.
payload: {files: {video: <out>.mp4, template?: json cue export}, duration_s}""",
)
