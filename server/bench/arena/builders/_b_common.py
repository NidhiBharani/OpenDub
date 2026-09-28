"""Shared helpers for the phase-B (source picture) pack builders. Skipped by the builder loader
(leading underscore). Builders never run models: ffmpeg/ffprobe are used only to cut clips and
read durations.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .. import paths


def source_dir(name: str) -> Path:
    d = paths.eval_dir() / "_sources" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, dst: Path, *, force: bool = False) -> Path:
    """Fetch ``url`` to ``dst`` (skipped when it exists); writes to a .part file first."""
    if dst.exists() and not force:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "opendub-arena/1"})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dst)
    return dst


def need_dir(path: str | Path | None, what: str, how: str) -> Path:
    """A user-supplied directory that must exist (media that cannot be fetched by script)."""
    if not path:
        raise FileNotFoundError(f"{what}: pass media_dir=… ({how})")
    p = Path(path).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(f"{what}: {p} is not a directory ({how})")
    return p


def probe(video: str | Path) -> dict[str, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    st = (info.get("streams") or [{}])[0]
    n, _, d = str(st.get("avg_frame_rate") or "25/1").partition("/")
    fps = float(n) / float(d or 1) if float(n or 0) else 25.0
    return {"duration_s": float((info.get("format") or {}).get("duration") or 0.0), "fps": fps,
            "width": float(st.get("width") or 0), "height": float(st.get("height") or 0)}


def cut_clip(src: Path, dst: Path, start: float, dur: float) -> Path:
    """Frame-accurate clip (re-encoded, original frame rate, AAC audio)."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t",
                    f"{dur:.3f}", "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-crf",
                    "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a",
                    "128k", str(dst)], check=True)
    return dst


def link_or_copy(src: Path, dst: Path) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def sample(rows: list[Any], n: int | None, seed: int = 0) -> list[Any]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def find_media(media_dir: Path, stem: str, exts: Iterable[str] = (".mp4", ".mkv", ".webm",
                                                                   ".mov", ".avi")) -> Path | None:
    for ext in exts:
        p = media_dir / f"{stem}{ext}"
        if p.exists():
            return p
    hits = sorted(media_dir.glob(f"{stem}.*"))
    return hits[0] if hits else None


# ------------------------------------------------------------------ AVA-style ASD annotations

SPEAKING = {"SPEAKING_AUDIBLE", "SPEAKING_AND_AUDIBLE"}


def speaking_runs(rows: list[dict[str, Any]], gap: float = 0.5, min_len: float = 0.4
                  ) -> list[dict[str, Any]]:
    """Per-entity runs of audible speech from AVA-style rows ``{t, box, label, entity}`` →
    ``[{start, end, t, box, entity}]`` (``t`` = run midpoint, ``box`` = the entity's box nearest
    to it)."""
    by_ent: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_ent.setdefault(r["entity"], []).append(r)
    runs = []
    for ent, rs in by_ent.items():
        rs.sort(key=lambda r: r["t"])
        cur: list[dict[str, Any]] = []
        for r in rs + [None]:  # type: ignore[list-item]
            if r is not None and r["label"] in SPEAKING and (not cur or r["t"] - cur[-1]["t"]
                                                            <= gap):
                cur.append(r)
                continue
            if cur:
                s, e = cur[0]["t"], cur[-1]["t"] + 0.04
                if e - s >= min_len:
                    mid = (s + e) / 2
                    near = min(cur, key=lambda x: abs(x["t"] - mid))
                    runs.append({"start": round(s, 3), "end": round(e, 3), "t": round(near["t"], 3),
                                 "box": [round(v, 4) for v in near["box"]], "entity": ent})
            cur = [r] if r is not None and r["label"] in SPEAKING else []
    return sorted(runs, key=lambda x: x["start"])


def offscreen_lines(speech: list[tuple[float, float]], onscreen: list[dict[str, Any]],
                    max_len: float = 5.0, min_len: float = 0.8) -> list[dict[str, Any]]:
    """Speech intervals (AVA-Speech) with no on-screen speaking run overlapping them by more
    than 20% → off-screen lines (split into ≤ ``max_len`` pieces)."""
    out = []
    for s, e in speech:
        if e - s < min_len:
            continue
        cover = sum(max(0.0, min(e, r["end"]) - max(s, r["start"])) for r in onscreen)
        if cover > 0.2 * (e - s):
            continue
        k = max(1, int((e - s) // max_len) + (1 if (e - s) % max_len > 0 else 0))
        step = (e - s) / k
        for i in range(k):
            a, b = s + i * step, s + (i + 1) * step
            out.append({"start": round(a, 3), "end": round(b, 3), "t": round((a + b) / 2, 3),
                        "box": None})
    return out


def ava_clip_item(item_id: str, group: str, video_rel: str, rows: list[dict[str, Any]],
                  speech: list[tuple[float, float]], clip_start: float, clip_dur: float,
                  meta: dict[str, Any], face_stride: int = 1) -> dict[str, Any] | None:
    """One B1 item from AVA-style rows (absolute times) restricted to a clip window."""
    lo, hi = clip_start, clip_start + clip_dur
    inside = [dict(r, t=r["t"] - lo) for r in rows if lo <= r["t"] < hi]
    if not inside:
        return None
    runs = speaking_runs(inside)
    sp = [(max(lo, a) - lo, min(hi, b) - lo) for a, b in speech if b > lo and a < hi]
    lines = sorted(runs + offscreen_lines(sp, runs), key=lambda x: x["start"])
    if not lines:
        return None
    ts = sorted({r["t"] for r in inside})
    keep_t = set(ts[::max(1, face_stride)])
    faces = [{"t": round(r["t"], 3), "box": [round(v, 4) for v in r["box"]],
              "speaking": r["label"] in SPEAKING} for r in inside if r["t"] in keep_t]
    return {"id": item_id, "group": group, "split": "test",
            "inputs": {"video": video_rel,
                       "lines": [{"start": ln["start"], "end": ln["end"]} for ln in lines]},
            "refs": {"lines": [{k: ln[k] for k in ("start", "end", "t", "box")} for ln in lines],
                     "faces": faces},
            "meta": {**meta, "clip_start_s": round(lo, 3), "duration_s": round(clip_dur, 3),
                     "n_offscreen": sum(ln["box"] is None for ln in lines)}}


def read_ava_csv(path: Path) -> list[dict[str, Any]]:
    """AVA-ActiveSpeaker CSV (no header): video_id, t, x1, y1, x2, y2, label, entity[, …]."""
    import csv

    rows = []
    with path.open(newline="") as f:
        for r in csv.reader(f):
            if len(r) < 8 or not r[1].replace(".", "", 1).isdigit():
                continue
            rows.append({"video": r[0], "t": float(r[1]),
                         "box": [float(r[2]), float(r[3]), float(r[4]), float(r[5])],
                         "label": r[6].strip(), "entity": r[7].strip()})
    return rows
