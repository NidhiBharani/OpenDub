"""AVA-ActiveSpeaker (val) → B1 packs: 60 s clips with line-level speaker→face references.

Sources (CC BY 4.0, research.google.com/ava/download, checked 2026-09-28):
- ``ava_activespeaker_val_v1.0.tar.bz2`` — per-video CSVs, no header: video_id, frame_timestamp,
  x1, y1, x2, y2 (normalised), label (SPEAKING_AUDIBLE | SPEAKING_NOT_AUDIBLE | NOT_SPEAKING; the
  web page spells them SPEAKING_AND_AUDIBLE …, both are accepted), entity_id.
- ``ava_speech_labels_v1.csv`` — AVA-Speech activity (video_id, start, end, label) over the same
  15:00–30:00 windows; speech with no on-screen speaking face becomes an **off-screen** line.
- Movies from the CVDF mirror ``s3.amazonaws.com/ava-dataset/trainval/<file>``
  (``annotations/ava_speech_file_names_v1.txt`` maps ids to file names). Whole movies are large
  (≈1–3 GB each); only ``videos`` of them are fetched.

Lines: each entity's contiguous SPEAKING_AUDIBLE run (gaps ≤ 0.5 s merged, ≥ 0.4 s long) is an
on-screen line whose reference box is the entity box at the run midpoint; AVA-Speech speech
intervals not covered (> 20%) by any on-screen run are off-screen lines (≤ 5 s pieces). Per-face
AVA labels are kept in ``refs.faces`` for the secondary frame AP.

Language: AVA's movies are mostly English-language but not all; the builder serves ``en`` by
default. For hi/ja pass ``video_ids`` of movies in that language (curate them yourself).
Groups: the movie (clips of one movie are not independent).
"""
from __future__ import annotations

import csv
import math
import tarfile
from pathlib import Path

from ..packs import pack_dir, write_pack
from . import builder
from ._b_common import ava_clip_item, cut_clip, download, probe, read_ava_csv, sample, source_dir

BASE = "https://research.google.com/ava/download"
S3 = "https://s3.amazonaws.com/ava-dataset"
SPEECH_LABELS = {"CLEAN_SPEECH", "SPEECH_WITH_MUSIC", "SPEECH_WITH_NOISE"}

LICENSE = """# AVA-ActiveSpeaker (val) → B1

Source: https://research.google.com/ava/ (AVA-ActiveSpeaker v1.0 val annotations, AVA-Speech
labels v1, movies from the CVDF mirror https://s3.amazonaws.com/ava-dataset/trainval/).
License: CC BY 4.0 (annotations). The movies are copyrighted films distributed by CVDF for
research; clips here are for evaluation only and are never redistributed.
Derived references: line-level speaker→face (on-screen runs of SPEAKING_AUDIBLE; off-screen
lines from AVA-Speech speech not covered by any on-screen speaker) — see
bench/arena/builders/ava_activespeaker.py.
Groups: movie id. Clips: {clip_s:g} s windows inside 15:00–30:00.
"""


@builder("ava_activespeaker", capabilities=["B1"], langs=["en", "hi", "ja"], license="CC BY 4.0")
def ava_activespeaker(lang: str, n: int | None = 60, *, videos: int = 20, clip_s: float = 60.0,
                      video_ids: list[str] | None = None, seed: int = 0,
                      capability: str = "B1") -> Path:
    """B1 pack from AVA-ActiveSpeaker val: ``n`` clips from ``videos`` movies."""
    if lang != "en" and not video_ids:
        raise ValueError("AVA has no language labels: pass video_ids=[…] of movies in "
                         f"{lang!r} (default serves en only)")
    src = source_dir("ava")
    tar = download(f"{BASE}/ava_activespeaker_val_v1.0.tar.bz2",
                   src / "ava_activespeaker_val_v1.0.tar.bz2")
    ann = src / "activespeaker_val"
    if not ann.exists():
        with tarfile.open(tar) as tf:
            tf.extractall(ann, filter="data")
    by_video: dict[str, list[dict]] = {}
    for f in sorted(ann.rglob("*.csv")):
        for r in read_ava_csv(f):
            by_video.setdefault(r["video"], []).append(r)

    speech_csv = download(f"{BASE}/ava_speech_labels_v1.csv", src / "ava_speech_labels_v1.csv")
    speech: dict[str, list[tuple[float, float]]] = {}
    with speech_csv.open(newline="") as f:
        for r in csv.reader(f):
            if len(r) >= 4 and r[3].strip() in SPEECH_LABELS:
                speech.setdefault(r[0], []).append((float(r[1]), float(r[2])))

    names = download(f"{S3}/annotations/ava_speech_file_names_v1.txt",
                     src / "ava_speech_file_names_v1.txt").read_text().split()
    file_of = {Path(nm).stem: nm for nm in names}
    ids = [v for v in (video_ids or sorted(by_video)) if v in by_video and v in file_of]
    ids = sample(ids, None if video_ids else videos, seed)
    per_video = max(1, math.ceil((n or len(ids) * 2) / max(1, len(ids))))

    out_dir = pack_dir(capability, lang)
    rows = []
    for vid in ids:
        movie = download(f"{S3}/trainval/{file_of[vid]}", src / "movies" / file_of[vid])
        ts = sorted(r["t"] for r in by_video[vid])
        lo, hi = ts[0], ts[-1]
        span = max(0.0, hi - lo - clip_s)
        for k in range(per_video):
            start = lo + (span * (k + 0.5) / per_video if per_video > 1 else span / 2)
            rel = f"clips/{vid}_{int(start)}.mp4"
            clip = cut_clip(movie, out_dir / rel, start, clip_s)
            info = probe(clip)
            item = ava_clip_item(f"ava-{vid}-{int(start)}", vid, rel, by_video[vid],
                                 speech.get(vid, []), start, clip_s,
                                 {"fps": info["fps"], "source": "ava_activespeaker",
                                  "video_id": vid})
            if item:
                rows.append(item)
    rows = rows[:n] if n else rows
    return write_pack(capability, lang, rows, LICENSE.format(clip_s=clip_s))
