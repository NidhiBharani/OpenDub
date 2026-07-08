"""Trivial diarization: assumes the whole recording is a single speaker."""
from __future__ import annotations

from pathlib import Path

from ...models import ASRSegment
from ..base import DiarizationProvider, ProgressFn, ProviderMeta, register


@register
class SingleSpeakerDiarization(DiarizationProvider):
    meta = ProviderMeta(
        id="diarization.single_speaker",
        kind="diarization",
        name="Single speaker",
        description="Assigns every segment to one speaker. Always available, no dependencies.",
        runtime="local",
        fields=[],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        return True, "ready"

    async def diarize(
        self, audio: Path, segments: list[ASRSegment], progress: ProgressFn
    ) -> list[str]:
        progress(1.0, "single speaker")
        return ["S0"] * len(segments)
