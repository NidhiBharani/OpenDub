"""No-op lip sync: copies the source video through unchanged.

The pipeline orchestrator normally short-circuits this provider entirely (marking the `lipsync`
stage `skipped` rather than invoking `sync()`), but this implementation is complete so direct
calls (tests, manual use) still behave correctly.
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from ..base import LipSyncProvider, ProgressFn, ProviderMeta, register


@register
class NoLipSyncProvider(LipSyncProvider):
    meta = ProviderMeta(
        id="lipsync.none",
        kind="lipsync",
        name="No lip sync",
        description="Skip lip sync; the dubbed audio is muxed onto the original video as-is.",
        runtime="local",
        fields=[],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        return True, "ready"

    async def sync(
        self, video: Path, audio: Path, out_video: Path, progress: ProgressFn
    ) -> None:
        progress(0.0, "no lip sync selected, passing video through")
        out_video.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, video, out_video)
        progress(1.0, "done")
