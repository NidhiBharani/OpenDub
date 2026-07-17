"""System metrics: per-stage wall time, peak VRAM, and realtime factor.

Timing/VRAM windows are captured by the runner (see runner.py) and passed through
``MetricContext.stage_timing`` as ``{stage: {"seconds": float, "peak_vram_mib": float}}``.
"""
from __future__ import annotations

from app.models import STAGE_ORDER

from .base import Metric, MetricContext


async def score(ctx: MetricContext) -> list[Metric]:
    out: list[Metric] = []
    media_dur = ctx.project.media.duration if ctx.project.media else 0.0
    total = 0.0
    peak_all = 0.0
    for stage in STAGE_ORDER:
        t = ctx.stage_timing.get(stage)
        if not t:
            continue
        secs = t.get("seconds", 0.0)
        vram = t.get("peak_vram_mib", 0.0)
        total += secs
        peak_all = max(peak_all, vram)
        out.append(Metric(f"time.{stage}", round(secs, 1), "s", "free", higher_is_better=False,
                          note=f"peak VRAM {vram:.0f} MiB" if vram else "wall time"))
    if total:
        out.append(Metric("time.total", round(total, 1), "s", "free", higher_is_better=False))
    if media_dur:
        out.append(Metric("time.realtime_factor", round(total / media_dur, 1), "x", "free",
                          higher_is_better=False,
                          note=f"{total:.0f}s to process {media_dur:.0f}s of media"))
    if peak_all:
        out.append(Metric("system.peak_vram", round(peak_all), "MiB", "free",
                          higher_is_better=False, note="max across all stages"))
    return out
