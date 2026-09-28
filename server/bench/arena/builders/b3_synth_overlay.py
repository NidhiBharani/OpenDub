"""Synthetic on-screen text for B3: subtitles, lower thirds, titles and signs rendered with PIL
onto video-like frames, so the ground truth (text, exact line box, role) is free — and available
for hi and ja, where public video-text sets are thin.

Backgrounds: frames from ``bg_dir`` (e.g. stills grabbed from Blender open movies or the
archive.org public-domain films the B4 builder fetches — any images you have the right to use),
centre-cropped to ``size``; without ``bg_dir``, procedural gradients with noise and shapes.
Fonts: Noto Sans / Noto Sans Devanagari / Noto Sans CJK JP (SIL OFL 1.1) downloaded from the
notofonts GitHub repos into ``data/eval/_sources/fonts`` (or ``fonts_dir``). Devanagari needs
Pillow built with libraqm (complex shaping); the builder refuses hi without it rather than write
wrongly shaped references.

Each image gets 1–3 overlays; JPEG quality 70–92 and an optional slight blur mimic video
compression. References are line-level (``granularity: line``) with a ``role``.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..packs import pack_dir, write_pack
from . import builder
from ._b_common import download, source_dir

NOTO = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts"
CJK = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Japanese"
FONTS = {
    "en": (f"{NOTO}/NotoSans/hinted/ttf/NotoSans-Regular.ttf",
           f"{NOTO}/NotoSans/hinted/ttf/NotoSans-Bold.ttf"),
    "hi": (f"{NOTO}/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf",
           f"{NOTO}/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf"),
    "ja": (f"{CJK}/NotoSansCJKjp-Regular.otf", f"{CJK}/NotoSansCJKjp-Bold.otf"),
}

CORPUS = {
    "en": {
        "subtitle": ["Where were you last night?", "We have to leave before sunrise.",
                     "I told you, it wasn't me.", "The train to Osaka leaves at nine.",
                     "Don't forget the documents.", "Everyone is waiting in the hall.",
                     "Call me when you get there.", "This changes everything.",
                     "How long have you known?", "Keep the lights off until I say so.",
                     "She never came back from the island.", "Thank you for coming today."],
        "lower_third": [("Dr. Maya Patel", "Marine Biologist"), ("Kenji Watanabe", "Chief Engineer"),
                        ("Sarah O'Connor", "City Council Member"),
                        ("Arjun Mehta", "Founder, BrightPath"), ("Lena Fischer", "Head of Design"),
                        ("Tomás Rivera", "Harbour Master")],
        "title": ["Chapter One: The Return", "Episode 4", "Three Years Later", "THE LAST HARBOR",
                  "Part II - Signals", "Quarterly Review 2026", "Getting Started"],
        "sign": ["EXIT", "Platform 3", "OPEN 24 HOURS", "Bakery & Cafe", "Pharmacy",
                 "Tokyo Station", "No Entry", "Settings", "Save changes"],
    },
    "hi": {
        "subtitle": ["कल रात तुम कहाँ थे?", "हमें सूरज निकलने से पहले निकलना होगा।",
                     "मैंने कहा था, वो मैं नहीं था।", "सब लोग हॉल में इंतज़ार कर रहे हैं।",
                     "वहाँ पहुँचकर मुझे फ़ोन करना।", "यह सब कुछ बदल देता है।",
                     "दस्तावेज़ भूलना मत।", "ट्रेन नौ बजे निकलेगी।",
                     "तुम्हें कब से पता था?", "आज आने के लिए धन्यवाद।"],
        "lower_third": [("डॉ. माया पटेल", "समुद्री जीवविज्ञानी"), ("अर्जुन मेहता", "संस्थापक, ब्राइटपाथ"),
                        ("सुनीता शर्मा", "नगर परिषद सदस्य"), ("राहुल वर्मा", "मुख्य अभियंता"),
                        ("प्रिया नायर", "डिज़ाइन प्रमुख")],
        "title": ["अध्याय एक: वापसी", "तीन साल बाद", "भाग दो", "आख़िरी बंदरगाह", "शुरुआत कैसे करें"],
        "sign": ["निकास", "प्लेटफ़ॉर्म 3", "दवाखाना", "चौबीस घंटे खुला", "रेलवे स्टेशन", "सावधान",
                 "प्रवेश निषेध"],
    },
    "ja": {
        "subtitle": ["昨日の夜、どこにいたの？", "日の出前に出発しなきゃ。", "言ったでしょ、私じゃないって。",
                     "みんなホールで待ってるよ。", "着いたら電話して。", "これで全部変わる。",
                     "書類を忘れないでね。", "大阪行きの電車は九時に出る。", "いつから知ってたの？",
                     "今日は来てくれてありがとう。"],
        "lower_third": [("佐藤 美咲", "海洋生物学者"), ("渡辺 健二", "チーフエンジニア"),
                        ("高橋 直樹", "市議会議員"), ("山本 あかり", "ブライトパス創業者"),
                        ("中村 蓮", "デザイン責任者")],
        "title": ["第一話 帰還", "三年後", "第二部", "最後の港", "はじめに"],
        "sign": ["出口", "3番線", "営業中", "薬局", "東京駅", "立入禁止", "パン屋", "設定"],
    },
}


def _fonts(lang: str, fonts_dir: str | None) -> tuple[Path, Path]:
    base = Path(fonts_dir).expanduser() if fonts_dir else source_dir("fonts")
    return tuple(download(u, base / Path(u).name) for u in FONTS[lang])  # type: ignore[return-value]


def _background(rng: random.Random, size: tuple[int, int], bgs: list[Path]):
    from PIL import Image, ImageDraw, ImageFilter

    W, H = size
    if bgs:
        im = Image.open(rng.choice(bgs)).convert("RGB")
        scale = max(W / im.width, H / im.height)
        im = im.resize((int(im.width * scale) + 1, int(im.height * scale) + 1))
        x0, y0 = (im.width - W) // 2, (im.height - H) // 2
        return im.crop((x0, y0, x0 + W, y0 + H))
    c1 = [rng.randint(0, 255) for _ in range(3)]
    c2 = [rng.randint(0, 255) for _ in range(3)]
    im = Image.new("RGB", size)
    px = ImageDraw.Draw(im)
    for y in range(H):
        a = y / H
        px.line([(0, y), (W, y)], fill=tuple(int(c1[k] * (1 - a) + c2[k] * a) for k in range(3)))
    for _ in range(rng.randint(3, 9)):
        x, y = rng.randint(0, W), rng.randint(0, H)
        r = rng.randint(20, 220)
        px.ellipse([x - r, y - r, x + r, y + r],
                   fill=tuple(rng.randint(0, 255) for _ in range(3)))
    return im.filter(ImageFilter.GaussianBlur(rng.uniform(2, 12)))


def _draw_text(draw, xy, text, font, fill, stroke=0, stroke_fill=None) -> list[float]:
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return list(draw.textbbox(xy, text, font=font, stroke_width=stroke))


def _render(rng: random.Random, im, lang: str, fonts: tuple[Path, Path], layout) -> list[dict]:
    from PIL import ImageDraw, ImageFont

    W, H = im.size
    draw = ImageDraw.Draw(im, "RGBA")
    reg, bold = fonts
    corpus = CORPUS[lang]
    roles = rng.sample(["subtitle", "lower_third", "title", "sign"], k=rng.randint(1, 3))
    if "title" in roles and "subtitle" in roles:
        roles.remove("title")
    out: list[dict] = []

    def font(path: Path, px: int):
        return ImageFont.truetype(str(path), px, layout_engine=layout)

    for role in roles:
        if role == "subtitle":
            text = rng.choice(corpus["subtitle"])
            f = font(reg, int(H * rng.uniform(0.045, 0.06)))
            tw = draw.textlength(text, font=f)
            box = _draw_text(draw, ((W - tw) / 2, H * rng.uniform(0.82, 0.88)), text, f,
                             (255, 255, 255), stroke=max(2, H // 240), stroke_fill=(0, 0, 0))
            out.append({"text": text, "box": box, "role": "caption"})
        elif role == "lower_third":
            name, title = rng.choice(corpus["lower_third"])
            f1, f2 = font(bold, int(H * 0.05)), font(reg, int(H * 0.034))
            x, y = W * rng.uniform(0.05, 0.1), H * rng.uniform(0.66, 0.72)
            w = max(draw.textlength(name, font=f1), draw.textlength(title, font=f2)) + W * 0.04
            draw.rectangle([x - W * 0.02, y - H * 0.015, x - W * 0.02 + w, y + H * 0.12],
                           fill=(10, 20, 40, rng.randint(150, 230)))
            out.append({"text": name, "box": _draw_text(draw, (x, y), name, f1, (255, 255, 255)),
                        "role": "lower_third"})
            out.append({"text": title, "box": _draw_text(draw, (x, y + H * 0.065), title, f2,
                                                         (220, 220, 160)),
                        "role": "lower_third"})
        elif role == "title":
            text = rng.choice(corpus["title"])
            f = font(bold, int(H * rng.uniform(0.08, 0.12)))
            tw = draw.textlength(text, font=f)
            box = _draw_text(draw, ((W - tw) / 2, H * rng.uniform(0.3, 0.45)), text, f,
                             (255, 255, 255), stroke=max(1, H // 360), stroke_fill=(30, 30, 30))
            out.append({"text": text, "box": box, "role": "title"})
        else:
            text = rng.choice(corpus["sign"])
            f = font(bold, int(H * rng.uniform(0.04, 0.07)))
            tw = draw.textlength(text, font=f)
            x = rng.uniform(0.05, 0.95) * (W - tw - W * 0.04)
            y = rng.uniform(0.05, 0.5) * H
            color = tuple(rng.randint(0, 160) for _ in range(3))
            draw.rectangle([x, y, x + tw + W * 0.04, y + f.size * 1.6], fill=(*color, 255))
            box = _draw_text(draw, (x + W * 0.02, y + f.size * 0.2), text, f, (255, 255, 255))
            out.append({"text": text, "box": box, "role": "sign"})
    for t in out:
        b = t["box"]
        t["box"] = [round(max(0.0, b[0] / W), 5), round(max(0.0, b[1] / H), 5),
                    round(min(1.0, b[2] / W), 5), round(min(1.0, b[3] / H), 5)]
    return out


@builder("b3_synth_overlay", capabilities=["B3"], langs=["en", "hi", "ja"],
         license="generated (Noto fonts SIL OFL 1.1; backgrounds per bg_dir)")
def b3_synth_overlay(lang: str, n: int | None = 300, *, bg_dir: str | None = None,
                     fonts_dir: str | None = None, size: tuple[int, int] = (1280, 720),
                     seed: int = 0, capability: str = "B3") -> Path:
    """B3 pack of ``n`` synthetic frames with rendered subtitles / lower thirds / titles / signs."""
    from PIL import ImageFilter, ImageFont, features

    if lang not in CORPUS:
        raise ValueError(f"no text corpus for {lang!r} (have {sorted(CORPUS)})")
    raqm = features.check("raqm")
    if lang == "hi" and not raqm:
        raise RuntimeError("Devanagari needs Pillow with libraqm (complex text shaping); "
                           "install libraqm and reinstall Pillow")
    layout = ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC
    fonts = _fonts(lang, fonts_dir)
    bgs = sorted(p for p in Path(bg_dir).expanduser().rglob("*")
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if bg_dir else []
    rng = random.Random(f"{seed}-{lang}")
    out_dir = pack_dir(capability, lang) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(n or 300):
        im = _background(rng, tuple(size), bgs)
        texts = _render(rng, im, lang, fonts, layout)
        if rng.random() < 0.3:
            im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
        name = f"synth_{lang}_{k:04d}.jpg"
        im.save(out_dir / name, quality=rng.randint(70, 92))
        rows.append({"id": f"synth-{lang}-{k:04d}", "group": f"s{k // 10}", "split": "test",
                     "inputs": {"image": f"images/{name}"},
                     "refs": {"texts": texts, "granularity": "line"},
                     "meta": {"source": "b3_synth_overlay", "roles": sorted({t["role"]
                                                                             for t in texts}),
                              "background": "bg_dir" if bgs else "procedural"}})
    return write_pack(capability, lang, rows, f"""# Synthetic text overlays ({lang}) → B3

Generated by bench/arena/builders/b3_synth_overlay.py (seed {seed}). Fonts: Noto Sans family,
SIL Open Font License 1.1. Backgrounds: {'user-supplied frames (' + str(bg_dir) + ')' if bgs
else 'procedural gradients'}. Ground truth is exact (rendered line boxes). Groups: blocks of 10.
""")
