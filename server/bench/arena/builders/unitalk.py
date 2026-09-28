"""UniTalk-ASD → B1 packs (in-the-wild, multilingual incl. Japanese; AVA-style labels at 25 fps).

Annotations: HF dataset ``plnguyen2908/UniTalk-ASD`` (per-video CSVs: AVA's 8 columns plus
label_id, instance_id). Media: **YouTube**, never redistributed — download the videos yourself
with the authors' ``download_dataset.py`` (github.com/plnguyen2908/UniTalk-ASD-code,
``video_list/*.csv``) into a directory and pass ``media_dir=``; files are matched by video id
(``<id>.mp4|mkv|webm``). The CSV timestamps are taken to be seconds in the source video, as in
AVA (unverified for UniTalk — check one clip in the audit viewer).

Language: UniTalk's language subset (``sub_categories/test_language.csv`` in the code repo) is
not labelled per language in the annotations; pass ``lang_map=`` (a CSV ``video_id,lang``) to
build a hi/ja pack, or ``video_ids=[…]``. Without either, the builder refuses rather than mixing
languages into one pack.

Licence: conflicting — the HF card says MIT, the paper CC BY-NC 4.0; treat as NC, eval-only.
No AVA-Speech layer exists here, so every reference line is on-screen (the off-screen class is
covered by AVA and the in-house set).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from ..packs import pack_dir, write_pack
from . import builder
from ._b_common import (
    ava_clip_item,
    cut_clip,
    find_media,
    need_dir,
    probe,
    read_ava_csv,
    sample,
    source_dir,
)

LICENSE = """# UniTalk-ASD → B1

Source: https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD (annotations);
media downloaded by the user from YouTube (github.com/plnguyen2908/UniTalk-ASD-code).
License: conflicting (HF card MIT, paper CC BY-NC 4.0) — treated as non-commercial, eval-only.
Media are not redistributed. Groups: source video.
"""


@builder("unitalk", capabilities=["B1"], langs=["en", "hi", "ja"],
         license="CC BY-NC 4.0 / MIT (conflicting)")
def unitalk(lang: str, n: int | None = 60, *, media_dir: str | None = None,
            lang_map: str | None = None, video_ids: list[str] | None = None,
            split: str = "val", clip_s: float = 60.0, seed: int = 0,
            capability: str = "B1") -> Path:
    """B1 pack from UniTalk-ASD annotations + user-downloaded YouTube media."""
    media = need_dir(media_dir, "unitalk", "YouTube media fetched with the authors' "
                     "download_dataset.py; files named <video_id>.<ext>")
    wanted = set(video_ids or [])
    if lang_map:
        with Path(lang_map).expanduser().open(newline="") as f:
            wanted |= {r[0] for r in csv.reader(f) if len(r) >= 2 and r[1].strip() == lang}
    if not wanted and lang != "en":
        raise ValueError(f"unitalk: pass lang_map= or video_ids= to select {lang!r} videos")
    from huggingface_hub import snapshot_download

    src = source_dir("unitalk")
    snapshot_download("plnguyen2908/UniTalk-ASD", repo_type="dataset", local_dir=str(src),
                      allow_patterns=["*.csv", "**/*.csv"])
    by_video: dict[str, list[dict]] = {}
    for f in sorted(src.rglob("*.csv")):
        if split and split not in f.as_posix():
            continue
        for r in read_ava_csv(f):
            by_video.setdefault(r["video"], []).append(r)
    ids = [v for v in sorted(by_video) if (not wanted or v in wanted)]
    ids = [v for v in ids if find_media(media, v)]
    if not ids:
        raise FileNotFoundError(f"unitalk: no annotated video found in {media}")
    ids = sample(ids, None, seed)
    per_video = max(1, math.ceil((n or len(ids)) / len(ids)))
    out_dir = pack_dir(capability, lang)
    rows = []
    for vid in ids:
        video = find_media(media, vid)
        ts = sorted(r["t"] for r in by_video[vid])
        lo, hi = ts[0], ts[-1]
        dur = min(clip_s, hi - lo + 1.0)
        span = max(0.0, hi - lo - dur)
        for k in range(per_video):
            start = lo + span * (k + 0.5) / per_video
            rel = f"clips/{vid}_{int(start)}.mp4"
            clip = cut_clip(video, out_dir / rel, start, dur)
            item = ava_clip_item(f"unitalk-{vid}-{int(start)}", vid, rel, by_video[vid], [],
                                 start, dur, {"fps": probe(clip)["fps"], "source": "unitalk",
                                              "video_id": vid})
            if item:
                rows.append(item)
        if n and len(rows) >= n:
            break
    return write_pack(capability, lang, rows[:n] if n else rows, LICENSE)
