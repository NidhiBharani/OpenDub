"""Lip-sync metrics — INTERFACE ONLY (v1).

The scoring slot is defined so the harness reports lip-sync as a first-class stage, but the
implementation is deferred until there is a live-action benchmark clip (LatentSync's face
detector cannot process 2D anime — see the OpenDub notes).

When implemented, this will wrap LatentSync's own evaluator,
``/home/nidhi/sources/LatentSync/eval/eval_sync_conf.py`` (SyncNet confidence + LSE-C/LSE-D),
run against ``render/lipsync.mp4``. Keep the ``score(ctx)`` signature identical.
"""
from __future__ import annotations

from .base import Metric, MetricContext, missing


async def score(ctx: MetricContext) -> list[Metric]:
    stage = ctx.project.stages.get("lipsync")
    status = stage.status if stage else "pending"
    if status == "skipped":
        return [missing("lipsync.sync_confidence", "lipsync provider is 'none' (stage skipped)",
                        provenance="free")]
    return [
        missing(
            "lipsync.sync_confidence",
            "not implemented in v1 — will wrap LatentSync eval_sync_conf.py (SyncNet) "
            "once a live-action benchmark clip exists",
            provenance="free",
        )
    ]
