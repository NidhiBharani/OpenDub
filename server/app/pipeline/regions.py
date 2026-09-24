"""Skip ranges ("keep the original here"): normalization and the derived Segment.skipped flag.

A skip range is a span of the source the user excludes from dubbing. Stages consult
``Segment.skipped`` (never the ranges directly), so this module is the single place that decides
which segments a set of ranges covers.
"""
from __future__ import annotations

from ..models import Project, Segment, TimeRange

MIN_RANGE = 0.1  # ranges shorter than this are dropped


def normalize_ranges(ranges: list[TimeRange], duration: float | None) -> list[TimeRange]:
    """Clamp to the media, drop degenerate spans, sort by start and merge overlapping/touching
    spans (the earlier range keeps its id; labels are joined with " / ")."""
    cleaned: list[TimeRange] = []
    for r in ranges:
        start = max(0.0, float(r.start))
        end = float(r.end)
        if duration is not None and duration > 0:
            start = min(start, duration)
            end = min(end, duration)
        if end - start < MIN_RANGE:
            continue
        cleaned.append(r.model_copy(update={"start": round(start, 3), "end": round(end, 3)}))
    cleaned.sort(key=lambda r: (r.start, r.end))

    merged: list[TimeRange] = []
    for r in cleaned:
        if merged and r.start <= merged[-1].end + 1e-6:
            prev = merged[-1]
            labels = [x for x in (prev.label.strip(), r.label.strip()) if x]
            merged[-1] = prev.model_copy(
                update={"end": max(prev.end, r.end), "label": " / ".join(dict.fromkeys(labels))}
            )
        else:
            merged.append(r)
    return merged


def is_skipped(seg: Segment, ranges: list[TimeRange]) -> bool:
    """A segment is skipped when its midpoint falls inside a range, or a range covers at least
    half of it."""
    if not ranges:
        return False
    mid = (seg.start + seg.end) / 2
    dur = max(seg.end - seg.start, 1e-6)
    for r in ranges:
        if r.start <= mid < r.end:
            return True
        overlap = min(seg.end, r.end) - max(seg.start, r.start)
        if overlap > 0 and overlap >= 0.5 * dur:
            return True
    return False


def apply_skip_ranges(project: Project) -> tuple[set[str], set[str]]:
    """Recompute ``Segment.skipped`` for every segment. Returns the ids that became skipped and
    the ids that stopped being skipped."""
    newly_skipped: set[str] = set()
    newly_unskipped: set[str] = set()
    for seg in project.segments:
        now = is_skipped(seg, project.skip_ranges)
        if now and not seg.skipped:
            newly_skipped.add(seg.id)
        elif seg.skipped and not now:
            newly_unskipped.add(seg.id)
        seg.skipped = now
    return newly_skipped, newly_unskipped


def spans(project: Project) -> list[tuple[float, float]]:
    return [(r.start, r.end) for r in project.skip_ranges]
