"""Shared F4 helpers (not a worker). Module level is stdlib-only; PIL/numpy load lazily.

- ``render_replacement``: draw each item text's translation into its box (PIL + Raqm, Noto
  fonts), optionally over an opaque plate whose colour is sampled from the ring around the box
  (the overlay-track method), or straight onto already-erased frames (erase + render).
- ``keyframe_edit``: the image-editor flow — edit the mid-span keyframe with an instruction,
  resize it back, paste the box region into every frame of the span (feathered), mux nothing
  (F4 packs are silent).
- ``edit_prompt``: one instruction wording for every editor, so they compete on the same ask.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SCRIPT_FONT = {"en": "noto-latin.ttf", "hi": "noto-deva.ttf", "ja": "noto-jp.ttf"}
LANG_NAMES = {"en": "English", "hi": "Hindi", "ja": "Japanese"}


def font_dir(params: dict) -> Path:
    if params.get("font_dir"):
        return Path(params["font_dir"]).expanduser()
    if os.environ.get("OPENDUB_FONT_DIR"):
        return Path(os.environ["OPENDUB_FONT_DIR"]).expanduser()
    data = Path(os.environ.get("OPENDUB_ARENA_DATA", Path(__file__).resolve().parents[4] / "data"))
    return data / "eval" / "_sources" / "fonts"


def font_for(lang: str, params: dict) -> Path:
    path = font_dir(params) / SCRIPT_FONT.get(lang, SCRIPT_FONT["en"])
    if not path.exists():
        raise FileNotFoundError(f"font {path} missing (the f4_overlay_synth builder downloads it)")
    return path


def edit_prompt(t: dict[str, Any]) -> str:
    return (f'Replace the on-screen text "{t["src_text"]}" with the {LANG_NAMES.get(t["tgt_lang"], t["tgt_lang"])} '
            f'text "{t["tgt_text"]}". Keep the same position, size, font style, colour, outline '
            f"and background. Change nothing else in the image.")


def _fit_font(draw, text: str, font_path: Path, box_w: int, box_h: int):
    from PIL import ImageFont

    size = max(10, int(box_h * 0.8))
    while size > 10:
        ft = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.RAQM)
        l, t, r, b = draw.textbbox((0, 0), text, font=ft)
        if r - l <= box_w and b - t <= box_h:
            return ft, (l, t, r, b)
        size -= 2
    ft = ImageFont.truetype(str(font_path), 10, layout_engine=ImageFont.Layout.RAQM)
    return ft, draw.textbbox((0, 0), text, font=ft)


def render_replacement(frames, fps: float, texts: list[dict], params: dict, *,
                       plate: bool, pad: int = 6):
    """Draw every text's translation into its box over [start, end]; returns new frames."""
    import numpy as np
    from PIL import Image, ImageDraw

    out = frames.copy()
    n, h, w, _ = frames.shape
    for t in texts:
        x0, y0, x1, y1 = (int(v) for v in t["box"])
        x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        i0, i1 = int(t["start"] * fps), min(n, int(np.ceil(t["end"] * fps)))
        if i1 <= i0:
            continue
        mid = frames[(i0 + i1) // 2]
        ring = np.concatenate([mid[max(0, y0 - 8):y0, x0:x1].reshape(-1, 3),
                               mid[y1:y1 + 8, x0:x1].reshape(-1, 3),
                               mid[y0:y1, max(0, x0 - 8):x0].reshape(-1, 3),
                               mid[y0:y1, x1:x1 + 8].reshape(-1, 3)])
        bg = np.median(ring, axis=0) if len(ring) else np.array([0, 0, 0])
        luma = float(bg @ np.array([0.299, 0.587, 0.114]))
        fg, stroke = ((0, 0, 0), (255, 255, 255)) if luma > 140 else ((255, 255, 255), (0, 0, 0))
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        if plate:
            draw.rectangle([x0, y0, x1, y1], fill=(*[int(c) for c in bg], 255))
        ft, (l, tt, r, b) = _fit_font(draw, t["tgt_text"], font_for(t["tgt_lang"], params),
                                      x1 - x0 - 4, y1 - y0 - 4)
        tx = x0 + (x1 - x0 - (r - l)) // 2 - l
        ty = y0 + (y1 - y0 - (b - tt)) // 2 - tt
        draw.text((tx, ty), t["tgt_text"], font=ft, fill=(*fg, 255),
                  stroke_width=0 if plate else 2, stroke_fill=(*stroke, 255))
        rgba = np.asarray(layer).astype(np.float32)
        a = rgba[..., 3:4] / 255.0
        for i in range(i0, i1):
            out[i] = np.clip(out[i] * (1 - a) + rgba[..., :3] * a, 0, 255).astype(np.uint8)
    return out


def keyframe_edit(item: dict, out: Path, edit_fn, params: dict) -> Path:
    """Edit each text's mid-span keyframe with ``edit_fn(PIL.Image, prompt, text) -> PIL.Image``
    and paste the box region into the span. Returns the written mp4."""
    import numpy as np
    from f_frames import paste_keyframe, read_frames, write_frames
    from PIL import Image

    frames, fps = read_frames(item["inputs"]["video"])
    n, h, w, _ = frames.shape
    for t in item["inputs"]["texts"]:
        mid = min(n - 1, int((t["start"] + t["end"]) / 2 * fps))
        edited = edit_fn(Image.fromarray(frames[mid]), edit_prompt(t), t)
        if edited.size != (w, h):
            edited = edited.convert("RGB").resize((w, h), Image.LANCZOS)
        frames = paste_keyframe(frames, fps, np.asarray(edited.convert("RGB")), t["box"],
                                t["start"], t["end"], margin=int(params.get("margin", 12)))
    return write_frames(frames, fps, out.with_suffix(".mp4"))
