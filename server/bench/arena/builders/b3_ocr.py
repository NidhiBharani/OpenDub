"""OCR packs for B3 from public scene-text sets.

- ``textocr`` (en) — TextOCR v0.1 val (textvqa.org/textocr, annotations CC BY 4.0, images from
  OpenImages CC BY 2.0): ``TextOCR_0.1_val.json`` (``imgs``: {id: {file_name, width, height}},
  ``anns``: {id: {image_id, bbox: [x, y, w, h], utf8_string}}, ``imgToAnns``) and
  ``train_val_images.zip`` (~7 GB; only the sampled images are extracted). Word-level; "." marks
  illegible text (don't care).
- ``mlt19`` (en / hi / ja) — ICDAR 2019 MLT (rrc.cvc.uab.es, registration required, research
  use): the user downloads the training images + ground truth into ``media_dir``; GT lines are
  ``x1,y1,x2,y2,x3,y3,x4,y4,script,transcription`` with scripts Latin / Hindi / Japanese / … and
  "###" for don't care. Images are chosen by their dominant script (en → Latin, hi → Hindi,
  ja → Japanese).

Both are still images (``inputs.image``); boxes are normalised by the image size.
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

from ..packs import pack_dir, write_pack
from . import builder
from ._b_common import download, link_or_copy, need_dir, sample, source_dir

TEXTOCR_ANN = "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json"
TEXTOCR_IMG = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
MLT_SCRIPT = {"en": "Latin", "hi": "Hindi", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
              "bn": "Bangla", "ar": "Arabic"}


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


@builder("textocr", capabilities=["B3"], langs=["en"], license="CC BY 4.0 / CC BY 2.0")
def textocr(lang: str, n: int | None = 300, *, min_words: int = 3, seed: int = 0,
            capability: str = "B3") -> Path:
    """B3 pack from TextOCR val: ``n`` images with ≥ ``min_words`` legible words."""
    if lang != "en":
        raise ValueError("TextOCR is English scene text; use mlt19 / b3_synth_overlay for hi, ja")
    src = source_dir("textocr")
    ann = json.loads(download(TEXTOCR_ANN, src / "TextOCR_0.1_val.json").read_text())
    img_to = ann.get("imgToAnns") or {}
    anns = ann["anns"]
    eligible = [i for i, ids in img_to.items()
                if sum(anns[a]["utf8_string"] != "." for a in ids) >= min_words]
    chosen = sample(sorted(eligible), n, seed)
    zpath = download(TEXTOCR_IMG, src / "train_val_images.zip")
    img_dir = pack_dir(capability, lang) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        members = {Path(m).name: m for m in zf.namelist()}
        rows = []
        for img_id in chosen:
            meta = ann["imgs"][img_id]
            name = Path(meta["file_name"]).name
            dst = img_dir / name
            if not dst.exists() and name in members:
                dst.write_bytes(zf.read(members[name]))
            if not dst.exists():
                continue
            W, H = float(meta["width"]), float(meta["height"])
            texts = []
            for a in img_to[img_id]:
                x, y, w, h = anns[a]["bbox"]
                s = anns[a]["utf8_string"]
                texts.append({"text": s, "box": [round(x / W, 5), round(y / H, 5),
                                                 round((x + w) / W, 5), round((y + h) / H, 5)],
                              "care": s != "."})
            rows.append({"id": f"textocr-{img_id}", "group": img_id, "split": "test",
                         "inputs": {"image": f"images/{name}"},
                         "refs": {"texts": texts, "granularity": "word"},
                         "meta": {"source": "textocr", "width": W, "height": H}})
    return write_pack(capability, lang, rows, """# TextOCR v0.1 val → B3

Source: https://textvqa.org/textocr (annotations CC BY 4.0; images from OpenImages, CC BY 2.0).
Word-level boxes; "." (illegible) is don't-care. Scene text, not video overlays: a proxy for
signs and on-screen text in live action. Groups: image.
""")


def _read_mlt_gt(path: Path) -> list[tuple[list[float], str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parts = line.strip().split(",", 9)
        if len(parts) < 10:
            continue
        try:
            quad = [float(v) for v in parts[:8]]
        except ValueError:
            continue
        out.append((quad, parts[8].strip(), parts[9]))
    return out


@builder("mlt19", capabilities=["B3"], langs=["en", "hi", "ja"], license="ICDAR MLT (research)")
def mlt19(lang: str, n: int | None = 300, *, media_dir: str | None = None, seed: int = 0,
          capability: str = "B3") -> Path:
    """B3 pack from ICDAR MLT-2019 (user-downloaded): images whose dominant script matches."""
    media = need_dir(media_dir, "mlt19", "MLT-2019 training images + gt_*.txt from "
                     "https://rrc.cvc.uab.es/?ch=15 (registration)")
    script = MLT_SCRIPT[lang]
    gts = {p.stem.removeprefix("gt_"): p for p in media.rglob("*.txt")}
    imgs = {p.stem: p for p in media.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png",
                                                                        ".gif", ".bmp")}
    eligible = []
    for stem, gt in sorted(gts.items()):
        if stem not in imgs:
            continue
        lines = _read_mlt_gt(gt)
        scripts = Counter(s for _q, s, t in lines if t != "###" and s not in ("None", "Symbols"))
        if scripts and scripts.most_common(1)[0][0] == script:
            eligible.append((stem, lines))
    rows = []
    for stem, lines in sample(eligible, n, seed):
        img = imgs[stem]
        rel = f"images/{img.name}"
        link_or_copy(img, pack_dir(capability, lang) / rel)
        W, H = _image_size(img)
        texts = []
        for quad, sc, text in lines:
            xs, ys = quad[0::2], quad[1::2]
            texts.append({"text": text, "script": sc,
                          "box": [round(min(xs) / W, 5), round(min(ys) / H, 5),
                                  round(max(xs) / W, 5), round(max(ys) / H, 5)],
                          "care": text != "###" and sc in (script, "Latin", "Mixed")})
        rows.append({"id": f"mlt19-{stem}", "group": stem, "split": "test",
                     "inputs": {"image": rel},
                     "refs": {"texts": texts, "granularity": "word"},
                     "meta": {"source": "mlt19", "script": script}})
    return write_pack(capability, lang, rows, f"""# ICDAR MLT-2019 ({script}) → B3

Source: https://rrc.cvc.uab.es/?ch=15 (registration; research use). Images downloaded by the
user; eval-only, not redistributed. Word (CJK: line) quadrilaterals reduced to boxes; "###" and
text in other scripts are don't-care. Groups: image.
""")
