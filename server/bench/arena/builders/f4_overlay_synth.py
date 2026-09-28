"""F4 synthetic-overlay pack: source-language text burned into clean moving backgrounds.

Every item has a known clean plate (``refs.clean_video``), a known text box and the intended
translation, so erasure collateral and the OCR round trip of the replacement are measurable
without annotation. Pack language = the *target* language: en items replace ja/hi text, hi and
ja items replace en text. Three overlay styles (lower third on a bar, outlined caption, large
title) over procedural backgrounds (drifting gradient + moving shapes + grain) or, when present,
user clips in ``<media_dir>/f4_backgrounds/*.mp4``.

Text is rendered with PIL's Raqm layout (needed for Devanagari conjuncts and matras) using Noto
fonts (SIL OFL 1.1) downloaded once into ``data/eval/_sources/fonts`` (or ``font_dir`` /
``OPENDUB_FONT_DIR``). No model runs; phrases are a fixed parallel list (hand translations).
"""
from __future__ import annotations

import os
import random
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from .. import paths
from . import builder
from ._fvideo import ffmpeg, find_media, merge_pack, probe

SOURCE = "f4_overlay_synth"
W, H, FPS, DUR = 1280, 720, 25, 6.0
SPAN = (1.0, 5.0)  # the text is on screen from 1 s to 5 s

FONT_URLS = {
    "latin": "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf",
    "deva": "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/"
            "NotoSansDevanagari%5Bwdth,wght%5D.ttf",
    "jp": "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf",
}
SCRIPT = {"en": "latin", "hi": "deva", "ja": "jp"}
SOURCES_FOR = {"en": ["ja", "hi"], "hi": ["en"], "ja": ["en"]}

# Parallel on-screen phrases (lower thirds, captions, titles): en / hi / ja.
PHRASES: list[dict[str, str]] = [
    {"en": "Chapter One", "hi": "पहला अध्याय", "ja": "第一章"},
    {"en": "Welcome back", "hi": "फिर से स्वागत है", "ja": "おかえりなさい"},
    {"en": "Three months later", "hi": "तीन महीने बाद", "ja": "三か月後"},
    {"en": "Live from Tokyo", "hi": "टोक्यो से सीधा प्रसारण", "ja": "東京から生中継"},
    {"en": "Breaking news", "hi": "ताज़ा ख़बर", "ja": "速報"},
    {"en": "Subscribe for more", "hi": "और देखने के लिए सब्सक्राइब करें", "ja": "チャンネル登録お願いします"},
    {"en": "Step 1: Open the settings", "hi": "चरण 1: सेटिंग्स खोलें", "ja": "ステップ1：設定を開く"},
    {"en": "Click Save", "hi": "सेव पर क्लिक करें", "ja": "保存をクリック"},
    {"en": "Limited time offer", "hi": "सीमित समय का ऑफ़र", "ja": "期間限定セール"},
    {"en": "The next morning", "hi": "अगली सुबह", "ja": "翌朝"},
    {"en": "Meanwhile, at the office", "hi": "इस बीच, दफ़्तर में", "ja": "その頃、オフィスでは"},
    {"en": "Top secret", "hi": "अति गोपनीय", "ja": "極秘"},
    {"en": "Thank you for watching", "hi": "देखने के लिए धन्यवाद", "ja": "ご視聴ありがとうございました"},
    {"en": "Coming soon", "hi": "जल्द आ रहा है", "ja": "近日公開"},
    {"en": "Chief Engineer", "hi": "मुख्य अभियंता", "ja": "主任技師"},
    {"en": "Do not enter", "hi": "प्रवेश निषेध", "ja": "立入禁止"},
    {"en": "Final exam results", "hi": "अंतिम परीक्षा के परिणाम", "ja": "期末試験の結果"},
    {"en": "Day 7", "hi": "सातवाँ दिन", "ja": "7日目"},
    {"en": "Price: 500 yen", "hi": "कीमत: 500 येन", "ja": "価格：500円"},
    {"en": "Episode 12", "hi": "एपिसोड 12", "ja": "第12話"},
    {"en": "Open 24 hours", "hi": "24 घंटे खुला", "ja": "24時間営業"},
    {"en": "Mission complete", "hi": "मिशन पूरा हुआ", "ja": "任務完了"},
    {"en": "Please wait", "hi": "कृपया प्रतीक्षा करें", "ja": "少々お待ちください"},
    {"en": "New record", "hi": "नया रिकॉर्ड", "ja": "新記録"},
]
STYLES = ("lower_third", "caption", "title")

LICENSE = """# F4 synthetic overlays (target {lang})

Built by `bench/arena/builders/f4_overlay_synth.py`: procedural backgrounds (or user clips from
`f4_backgrounds/`) with source-language text rendered by PIL/Raqm in Noto fonts (SIL OFL 1.1).
Phrases and translations are hand-written for this pack. Free to use; user background clips stay
private. Groups: background clip (texts over the same background are not independent).
"""


def font_path(script: str, font_dir: str | None = None) -> Path:
    root = Path(font_dir or os.environ.get("OPENDUB_FONT_DIR")
                or paths.eval_dir() / "_sources" / "fonts").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    dst = root / f"noto-{script}.ttf"
    if not dst.exists():
        with urllib.request.urlopen(FONT_URLS[script], timeout=120) as r:
            dst.write_bytes(r.read())
    return dst


def procedural_background(dst: Path, seed: int) -> None:
    """Drifting two-colour gradient, a few moving discs and film grain (6 s, 1280x720, 25 fps)."""
    rng = np.random.default_rng(seed)
    c0, c1 = rng.uniform(20, 235, 3), rng.uniform(20, 235, 3)
    discs = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(40, 160), rng.uniform(-6, 6),
              rng.uniform(-4, 4), rng.uniform(0, 255, 3)) for _ in range(4)]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-crf", "12", "-pix_fmt", "yuv420p", str(dst)],
                            stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for f in range(int(DUR * FPS)):
        t = f / FPS
        a = (np.sin((xx / W + t * 0.05) * np.pi) * 0.5 + 0.5)[..., None]
        img = c0 * a + c1 * (1 - a)
        for x, y, r, vx, vy, col in discs:
            cx, cy = (x + vx * f) % W, (y + vy * f) % H
            m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
            img[m] = img[m] * 0.4 + col * 0.6
        img += rng.normal(0, 4, img.shape)
        proc.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed while writing a procedural background")


def render_text(text: str, style: str, font: Path, rng: random.Random):
    """RGBA overlay (full frame) and its box [x0, y0, x1, y1] (all drawn pixels, bar included)."""
    from PIL import Image, ImageDraw, ImageFont, features

    if not features.check("raqm"):
        raise RuntimeError("Pillow lacks Raqm: Devanagari would be mis-shaped")
    size = {"lower_third": 44, "caption": 40, "title": 84}[style]
    ft = ImageFont.truetype(str(font), size, layout_engine=ImageFont.Layout.RAQM)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    l, t, r, b = d.textbbox((0, 0), text, font=ft)
    tw, th = r - l, b - t
    if style == "lower_third":
        x, y = 80, H - 170
        d.rectangle([x - 24, y - 16, x + tw + 24, y + th + 20], fill=(10, 20, 40, 200))
        fill, stroke = (255, 255, 255, 255), 0
    elif style == "caption":
        x, y = (W - tw) // 2, H - 90 - th
        fill, stroke = (255, 255, 255, 255), 3
    else:
        x, y = (W - tw) // 2 + rng.randint(-80, 80), H // 3 + rng.randint(-40, 40)
        fill, stroke = (rng.randint(180, 255), rng.randint(120, 255), rng.randint(0, 120), 255), 4
    d.text((x - l, y - t), text, font=ft, fill=fill, stroke_width=stroke,
           stroke_fill=(0, 0, 0, 255))
    # The box is everything the overlay drew (text, outline, bar): what a replacement must cover.
    x0, y0, x1, y1 = img.getchannel("A").getbbox() or (x, y, x + tw, y + th)
    return img, [max(0, x0 - 1), max(0, y0 - 1), min(W, x1 + 1), min(H, y1 + 1)]


@builder(SOURCE, capabilities=["F4"], langs=["en", "hi", "ja"], license="synthetic (OFL fonts)")
def f4_overlay_synth(lang: str, n: int | None = 48, seed: int = 0,
                     font_dir: str | None = None, media_dir: str | None = None) -> Path:
    """F4 pack: known text boxes over clean plates; replace source text with the translation."""
    rng = random.Random(f"{seed}-{lang}")
    out = paths.eval_dir() / "F4" / lang
    media = Path(media_dir or os.environ.get("OPENDUB_F1_MEDIA")
                 or paths.eval_dir() / "_sources" / "f1_selfreenact").expanduser()
    user_bgs = find_media(media / "f4_backgrounds") if (media / "f4_backgrounds").exists() else []
    n = n or 48
    n_bg = max(1, n // 3)
    bgs: list[Path] = []
    for i in range(n_bg):
        bg = out / SOURCE / "plates" / f"bg{i:03d}.mp4"
        if not bg.exists():
            if user_bgs:
                src = user_bgs[i % len(user_bgs)]
                ffmpeg("-i", str(src), "-t", f"{DUR}", "-vf",
                       f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,"
                       f"crop={W}:{H}", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                       "-an", str(bg))
            else:
                procedural_background(bg, seed * 1000 + i + 17 * len(lang))
        bgs.append(bg)
    rows = []
    for k in range(n):
        phrase = PHRASES[rng.randrange(len(PHRASES))]
        src_lang = SOURCES_FOR[lang][k % len(SOURCES_FOR[lang])]
        style = STYLES[k % len(STYLES)]
        bg = bgs[k % len(bgs)]
        img, box = render_text(phrase[src_lang], style, font_path(SCRIPT[src_lang], font_dir), rng)
        ov = out / SOURCE / "overlays" / f"t{k:03d}.png"
        ov.parent.mkdir(parents=True, exist_ok=True)
        img.save(ov)
        vid = out / SOURCE / "items" / f"t{k:03d}.mp4"
        vid.parent.mkdir(parents=True, exist_ok=True)
        if not vid.exists():
            ffmpeg("-i", str(bg), "-i", str(ov), "-filter_complex",
                   f"[0:v][1:v]overlay=0:0:enable='between(t,{SPAN[0]},{SPAN[1]})'",
                   "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-an", str(vid))
        rows.append({
            "id": f"ovl-{lang}-{k:03d}", "group": bg.stem, "split": "test",
            "inputs": {"video": str(vid.relative_to(out)),
                       "texts": [{"box": box, "start": SPAN[0], "end": SPAN[1],
                                  "src_text": phrase[src_lang], "src_lang": src_lang,
                                  "tgt_text": phrase[lang], "tgt_lang": lang,
                                  "style": {"kind": style}}]},
            "refs": {"clean_video": str(bg.relative_to(out))},
            "meta": {"src_lang": src_lang, "width": W, "height": H, "fps": FPS,
                     "duration_s": probe(vid)["duration"] or DUR, "style": style}})
    return merge_pack("F4", lang, SOURCE, rows, LICENSE.format(lang=lang))
