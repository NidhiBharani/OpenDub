"""Derived analysis of the source: suggested cut points ("anchors") and the viewer filmstrip.

Anchors merge three sources — shot changes in the picture (`ffmpeg` scene score), silences in the
separated dialogue, and segment boundaries — into one sorted list the timeline draws under its
ruler so the user can pick a whole scene and keep it original. The two expensive detectors are
cached under `analysis/` keyed by the mtime of the file they read; segment edges are merged in
at request time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .. import store
from ..media import ffmpeg
from ..models import Project

AnchorKind = Literal["scene", "silence", "segment"]

SCENE_THRESHOLD = 0.3  # scene score below which frame differences are not cuts
SILENCE_NOISE_DB = -35.0
SILENCE_MIN = 0.3  # seconds
COALESCE = 0.1  # anchors closer than this merge into one
FILMSTRIP_COLS = 20
FILMSTRIP_TILE = (96, 54)


class Anchor(BaseModel):
    t: float
    kind: AnchorKind
    confidence: float = 1.0  # scene score; silence length clamped to 0..1; 1 for segment edges
    start: float = 0.0  # silence anchors carry their extent; others are 0-width
    end: float = 0.0


class AnchorSet(BaseModel):
    version: int = 1
    anchors: list[Anchor] = Field(default_factory=list)


class Filmstrip(BaseModel):
    interval: float
    cols: int
    rows: int
    tile_w: int
    tile_h: int
    count: int
    url: str


_PRIORITY = {"scene": 0, "silence": 1, "segment": 2}


def _cache_path(project: Project, name: str) -> Path:
    return store.resolve(project, f"analysis/{name}")


def _load_cache(path: Path, source: Path) -> dict | None:
    if not path.exists() or not source.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if data.get("source_mtime") != source.stat().st_mtime:
        return None
    return data


def _write_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


async def detect_scenes(project: Project, *, refresh: bool = False) -> list[tuple[float, float]]:
    """(time, score) shot boundaries of playback.mp4, cached in analysis/scenes.json."""
    video = store.resolve(project, "playback.mp4")
    if not video.exists():
        return []
    cache = _cache_path(project, "scenes.json")
    if not refresh:
        data = _load_cache(cache, video)
        if data is not None:
            return [(float(c["t"]), float(c["score"])) for c in data.get("cuts", [])]
    cuts = await ffmpeg.detect_scene_changes(video, threshold=SCENE_THRESHOLD)
    _write_cache(
        cache,
        {
            "version": 1,
            "source_mtime": video.stat().st_mtime,
            "threshold": SCENE_THRESHOLD,
            "cuts": [{"t": t, "score": s} for t, s in cuts],
        },
    )
    return cuts


async def detect_silences(project: Project, *, refresh: bool = False) -> list[tuple[float, float]]:
    """(start, end) silences in the dialogue (vocals.wav, else original.wav), cached in
    analysis/silences.json."""
    audio = store.resolve(project, "audio/vocals.wav")
    if not audio.exists():
        audio = store.resolve(project, "audio/original.wav")
    if not audio.exists():
        return []
    duration = project.media.duration if project.media else 0.0
    cache = _cache_path(project, "silences.json")
    if not refresh:
        data = _load_cache(cache, audio)
        if data is not None:
            return [(float(g["start"]), float(g["end"])) for g in data.get("gaps", [])]
    raw = await ffmpeg.detect_silences(audio, noise_db=SILENCE_NOISE_DB, min_dur=SILENCE_MIN)
    gaps = [(s, e if e is not None else duration) for s, e in raw]
    gaps = [(s, e) for s, e in gaps if e > s]
    _write_cache(
        cache,
        {
            "version": 1,
            "source_mtime": audio.stat().st_mtime,
            "noise_db": SILENCE_NOISE_DB,
            "min_dur": SILENCE_MIN,
            "gaps": [{"start": s, "end": e} for s, e in gaps],
        },
    )
    return gaps


def merge_anchors(
    scenes: list[tuple[float, float]],
    silences: list[tuple[float, float]],
    segment_edges: list[float],
    duration: float,
) -> list[Anchor]:
    """Sort all candidates and coalesce those within COALESCE seconds, keeping the
    highest-priority kind (scene > silence > segment) and the strongest confidence."""
    raw: list[Anchor] = []
    for t, score in scenes:
        raw.append(Anchor(t=t, kind="scene", confidence=score))
    for s, e in silences:
        length = e - s
        raw.append(
            Anchor(t=round((s + e) / 2, 3), kind="silence", confidence=min(1.0, length / 2.0), start=s, end=e)
        )
    for t in segment_edges:
        raw.append(Anchor(t=t, kind="segment", confidence=1.0))

    if duration > 0:
        raw = [a for a in raw if 0.0 <= a.t <= duration]
    raw.sort(key=lambda a: (a.t, _PRIORITY[a.kind]))

    out: list[Anchor] = []
    for a in raw:
        if out and a.t - out[-1].t < COALESCE:
            prev = out[-1]
            if _PRIORITY[a.kind] < _PRIORITY[prev.kind]:
                out[-1] = a.model_copy(update={"confidence": max(a.confidence, prev.confidence if prev.kind == a.kind else a.confidence)})
            elif a.kind == prev.kind:
                out[-1] = prev.model_copy(update={"confidence": max(a.confidence, prev.confidence)})
            continue
        out.append(a)
    return out


async def anchors(project: Project, *, refresh: bool = False) -> AnchorSet:
    duration = project.media.duration if project.media else 0.0
    scenes = await detect_scenes(project, refresh=refresh)
    silences = await detect_silences(project, refresh=refresh)
    edges: list[float] = []
    for seg in project.segments:
        edges.append(round(seg.start, 3))
        edges.append(round(seg.end, 3))
    return AnchorSet(anchors=merge_anchors(scenes, silences, edges, duration))


def filmstrip_interval(duration: float) -> float:
    """One tile every 1 s for short clips, growing so a feature never exceeds ~600 tiles."""
    if duration <= 0:
        return 1.0
    for step in (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0):
        if duration / step <= 600:
            return step
    return 120.0


async def filmstrip(project: Project, *, refresh: bool = False) -> Filmstrip | None:
    """Contact sheet of playback.mp4 under analysis/filmstrip.jpg (+ .json), built on demand."""
    video = store.resolve(project, "playback.mp4")
    if not video.exists() or not project.media:
        return None
    sheet = _cache_path(project, "filmstrip.jpg")
    meta = _cache_path(project, "filmstrip.json")
    if not refresh and sheet.exists():
        data = _load_cache(meta, video)
        if data is not None:
            return Filmstrip(**data["filmstrip"])
    interval = filmstrip_interval(project.media.duration)
    tile_w, tile_h = FILMSTRIP_TILE
    count = await ffmpeg.filmstrip(video, sheet, interval, FILMSTRIP_COLS, tile_w, tile_h)
    rows = (count + FILMSTRIP_COLS - 1) // FILMSTRIP_COLS
    strip = Filmstrip(
        interval=interval,
        cols=FILMSTRIP_COLS,
        rows=rows,
        tile_w=tile_w,
        tile_h=tile_h,
        count=count,
        url=f"/api/media/{project.id}/analysis/filmstrip.jpg?v={int(video.stat().st_mtime)}",
    )
    _write_cache(meta, {"version": 1, "source_mtime": video.stat().st_mtime, "filmstrip": strip.model_dump()})
    return strip
