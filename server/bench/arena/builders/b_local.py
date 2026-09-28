"""In-house B1–B4 packs from local media plus the user's own annotations (the "OpenDub film set"
layers: speaker face per line, shots, on-screen text, content type), and annotation templates to
start them.

Layout, per capability and language::

    data/eval/_sources/b_local/<ID>/<lang>/
        media/…                      videos / images (anything you may use: archive.org public
                                     domain, your own footage, private official clips — the pack
                                     LICENSE says eval-only, private)
        annotations.jsonl            one item per line, refs in the spec's format (below)
        annotations.template.jsonl   written by ``template=True``; fill in and rename

Item lines: ``{"id", "media": "media/x.mp4", "group"?, "split"?, "inputs"?: {...},
"refs": {...}, "meta"?: {...}}``. ``media`` becomes ``inputs.video`` (``inputs.image`` for image
files); refs follow the spec ``io`` (B1: ``lines`` with ``box`` or null; B2: ``transitions`` [+
``lines``]; B3: ``texts`` + ``inputs.times``; B4: ``label``).

``archive_pd_template`` downloads public-domain animation from archive.org (default: the 1940s
Japanese shorts and Fleischer/Popeye cartoons) into ``media/`` and writes a B2 template with empty
``transitions`` — hand annotation for anime-style cuts (impact flashes, holds, telecine) that no
public SBD set covers. Nothing is pre-filled from a detector, so the reference does not inherit
any candidate's errors.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..packs import pack_dir, write_pack
from . import builder
from ._b_common import download, link_or_copy, probe, source_dir
from .b4_titles import archive_video_url

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".ogv", ".mpg", ".mpeg")
EMPTY_REFS = {"B1": {"lines": []}, "B2": {"transitions": [], "lines": []},
              "B3": {"texts": [], "granularity": "line"}, "B4": {"label": None}}
ARCHIVE_PD_ANIMATION = ["kumo-to-churippu-1943", "momotaro-umi-no-shinpei-1945",
                        "superman_the_mechanical_monsters", "superman_electric_earthquake",
                        "popeye_patriotic_popeye", "bb_minnie_the_moocher"]


def local_dir(capability: str, lang: str) -> Path:
    d = source_dir("b_local") / capability / lang
    (d / "media").mkdir(parents=True, exist_ok=True)
    return d


def write_template(capability: str, lang: str, *, times_every_s: float = 10.0) -> Path:
    base = local_dir(capability, lang)
    lines = []
    for m in sorted((base / "media").rglob("*")):
        if m.suffix.lower() not in VIDEO_EXT + IMAGE_EXT:
            continue
        rel = m.relative_to(base).as_posix()
        row: dict = {"id": f"{capability.lower()}-{lang}-{m.stem}", "media": rel,
                     "group": m.stem, "split": "test",
                     "refs": json.loads(json.dumps(EMPTY_REFS[capability]))}
        if m.suffix.lower() in VIDEO_EXT:
            info = probe(m)
            row["meta"] = {"duration_s": round(info["duration_s"], 3), "fps": info["fps"]}
            if capability == "B3":
                n = max(1, int(info["duration_s"] // times_every_s))
                row["inputs"] = {"times": [round((k + 0.5) * info["duration_s"] / n, 2)
                                           for k in range(n)]}
            if capability == "B1":
                row["inputs"] = {"lines": []}
        lines.append(json.dumps(row, ensure_ascii=False))
    out = base / "annotations.template.jsonl"
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    return out


@builder("b_local", capabilities=["B1", "B2", "B3", "B4"], langs=["en", "hi", "ja"],
         license="per item (in-house; eval-only, private)")
def b_local(lang: str, n: int | None = None, *, capability: str = "B2",
            template: bool = False, annotations: str | None = None) -> Path:
    """In-house pack from ``_sources/b_local/<ID>/<lang>/annotations.jsonl`` (``template=True``
    writes ``annotations.template.jsonl`` for the media present and returns its path)."""
    if capability not in EMPTY_REFS:
        raise ValueError(f"b_local serves B1–B4, not {capability}")
    if template:
        return write_template(capability, lang)
    base = local_dir(capability, lang)
    ann = Path(annotations).expanduser() if annotations else base / "annotations.jsonl"
    if not ann.exists():
        raise FileNotFoundError(f"{ann} missing: put media in {base / 'media'}, run with "
                                "template=True, fill the refs, save as annotations.jsonl")
    rows = []
    for line in ann.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        src = Path(r["media"]) if Path(r["media"]).is_absolute() else base / r["media"]
        key = "image" if src.suffix.lower() in IMAGE_EXT else "video"
        rel = f"media/{src.name}"
        link_or_copy(src, pack_dir(capability, lang) / rel)
        meta = dict(r.get("meta") or {})
        if key == "video" and "duration_s" not in meta:
            info = probe(src)
            meta.update(duration_s=round(info["duration_s"], 3), fps=info["fps"])
        meta.setdefault("source", "b_local")
        rows.append({"id": r["id"], "group": r.get("group") or src.stem,
                     "split": r.get("split", "test"),
                     "inputs": {**(r.get("inputs") or {}), key: rel},
                     "refs": r["refs"], "meta": meta})
        if n and len(rows) >= n:
            break
    return write_pack(capability, lang, rows, f"""# In-house {capability} ({lang})

Local media and annotations from {ann}. Eval-only and private: never commit or redistribute
(official-upload clips are the rights holders'; archive.org items are public domain per their
pages). Groups: as annotated (default: media file).
""")


@builder("archive_pd_template", capabilities=["B2"], langs=["en", "hi", "ja"],
         license="public domain (archive.org)")
def archive_pd_template(lang: str, n: int | None = None, *, ids: list[str] | None = None,
                        capability: str = "B2") -> Path:
    """Download public-domain animation from archive.org into b_local media and write a B2
    annotation template (empty transitions) to fill by hand."""
    base = local_dir(capability, lang)
    for ident in (ids or ARCHIVE_PD_ANIMATION)[:n or None]:
        url = archive_video_url(ident)
        dst = base / "media" / f"{ident}{Path(url).suffix}"
        if not dst.exists():
            download(url, dst)
    return write_template(capability, lang)
