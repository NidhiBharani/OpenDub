"""Zero-dependency "ASR": segments speech-ish regions with ffmpeg's `silencedetect` filter and
fills in placeholder line text. Makes the whole pipeline runnable without any ML dependency
installed — every other ASR provider is optional.
"""
from __future__ import annotations

import math
from pathlib import Path

from ...media import ffmpeg
from ...models import ASRSegment
from ..base import ASRProvider, ProgressFn, ProviderMeta, register

_NOISE_DB = -35.0
_MIN_SILENCE = 0.45
_MIN_REGION = 0.3
_MAX_SEGMENT_DURATION = 12.0
_MAX_SEGMENTS = 500



async def _detect_silences(audio: Path) -> list[tuple[float, float | None]]:
    """(start, end) silence intervals; `end` is None when the file ends inside a silence."""
    return await ffmpeg.detect_silences(audio, noise_db=_NOISE_DB, min_dur=_MIN_SILENCE)


def _merge_short_regions(
    regions: list[tuple[float, float]], min_dur: float = _MIN_REGION
) -> list[tuple[float, float]]:
    """Merge any region shorter than `min_dur` into a neighbor (previous, else next)."""
    if not regions:
        return []
    result: list[list[float]] = []
    pending_start: float | None = None
    for start, end in regions:
        s = pending_start if pending_start is not None else start
        pending_start = None
        if end - s < min_dur:
            if result:
                result[-1][1] = end
            else:
                pending_start = s
        else:
            result.append([s, end])
    if pending_start is not None:
        if result:
            result[-1][1] = max(result[-1][1], regions[-1][1])
        else:
            result.append([pending_start, regions[-1][1]])
    return [(s, e) for s, e in result]


def _split_long_regions(
    regions: list[tuple[float, float]], max_dur: float = _MAX_SEGMENT_DURATION
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in regions:
        dur = end - start
        if dur <= max_dur or dur <= 0:
            out.append((start, end))
            continue
        n_parts = math.ceil(dur / max_dur)
        part_len = dur / n_parts
        for i in range(n_parts):
            p_start = start + i * part_len
            p_end = end if i == n_parts - 1 else start + (i + 1) * part_len
            out.append((p_start, p_end))
    return out


@register
class MockSegmenterASR(ASRProvider):
    meta = ProviderMeta(
        id="asr.mock",
        kind="asr",
        name="Segmenter (mock)",
        description=(
            "Zero-dependency stand-in for real ASR: detects speech-ish regions with ffmpeg "
            "silencedetect and emits placeholder line text. Lets the whole pipeline run without "
            "any ML models installed."
        ),
        runtime="local",
        fields=[],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        return True, "ready"

    async def transcribe(
        self, audio: Path, language: str, progress: ProgressFn
    ) -> list[ASRSegment]:
        from ...media.ffmpeg import wav_duration

        progress(0.02, "probing audio")
        duration = await wav_duration(audio)
        if duration <= 0:
            progress(1.0, "done")
            return []

        progress(0.1, "detecting speech regions")
        raw_silences = await _detect_silences(audio)
        silences = sorted(
            (max(0.0, min(s, duration)), max(0.0, min(e if e is not None else duration, duration)))
            for s, e in raw_silences
        )

        speech_regions: list[tuple[float, float]] = []
        cursor = 0.0
        for s_start, s_end in silences:
            if s_start > cursor:
                speech_regions.append((cursor, s_start))
            cursor = max(cursor, s_end)
        if cursor < duration:
            speech_regions.append((cursor, duration))

        progress(0.4, "merging short regions")
        speech_regions = _merge_short_regions(speech_regions)

        progress(0.6, "splitting long regions")
        final_regions = _split_long_regions(speech_regions)

        if len(final_regions) > _MAX_SEGMENTS:
            final_regions = final_regions[:_MAX_SEGMENTS]

        progress(0.9, "building segments")
        out = [
            ASRSegment(start=round(s, 3), end=round(e, 3), text=f"[line {i + 1}]")
            for i, (s, e) in enumerate(final_regions)
            if e > s
        ]
        progress(1.0, "done")
        return out
