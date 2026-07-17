"""TTS-stage metrics — reference-free.

The most valuable one is degenerate_take_rate: it flags silent/near-empty takes (exactly the
sung-reference F5-TTS failure we hit, a 0.04s -90 dB clip). round_trip_wer re-transcribes each
active take and compares to its translated_text — a proxy for intelligibility that needs no
ground truth. Speaker similarity is left as an optional slot (needs ECAPA/speechbrain).
"""
from __future__ import annotations

import math
import wave

from .base import Metric, MetricContext, missing

_DEGENERATE_MAX_DUR = 0.25  # seconds
_DEGENERATE_MIN_PEAK = -50.0  # dBFS
_RATE_CLAMP = (0.6, 1.6)  # mix-stage atempo clamp; a take pinned to a bound didn't fit its slot


def _take_stats(path) -> tuple[float, float] | None:
    """(duration_s, peak_dbfs) for a wav, or None if unreadable."""
    import audioop

    try:
        with wave.open(str(path)) as w:
            sr, width, n = w.getframerate(), w.getsampwidth(), w.getnframes()
            frames = w.readframes(n)
    except (wave.Error, OSError, EOFError):
        return None
    dur = n / sr if sr else 0.0
    if not frames:
        return dur, -math.inf
    peak = audioop.max(frames, width)
    full = float(1 << (8 * width - 1))
    return dur, (20 * math.log10(peak / full) if peak else -math.inf)


async def score(ctx: MetricContext) -> list[Metric]:
    segs = ctx.project.segments
    with_text = [s for s in segs if s.translated_text.strip()]
    if not with_text:
        return [missing("tts.degenerate_take_rate", "no translated segments to synthesize")]

    degenerate = 0
    counted = 0
    clamped = 0
    for s in with_text:
        take = s.active_take()
        if take is None:
            degenerate += 1  # a line that should speak but has no active take
            counted += 1
            continue
        stats = _take_stats(ctx.path(take.path))
        if stats is None:
            continue
        counted += 1
        dur, peak = stats
        if dur < _DEGENERATE_MAX_DUR or peak < _DEGENERATE_MIN_PEAK:
            degenerate += 1
        if take.rate_factor <= _RATE_CLAMP[0] + 1e-3 or take.rate_factor >= _RATE_CLAMP[1] - 1e-3:
            clamped += 1

    out = [
        Metric(
            "tts.degenerate_take_rate",
            round(degenerate / counted, 3) if counted else None, "ratio", "free",
            higher_is_better=False,
            note=f"{degenerate}/{counted} takes silent (<{_DEGENERATE_MAX_DUR}s or "
                 f"<{_DEGENERATE_MIN_PEAK}dBFS) or missing",
        ),
        Metric(
            "tts.duration_clamp_rate",
            round(clamped / counted, 3) if counted else None, "ratio", "free",
            higher_is_better=False,
            note=f"{clamped}/{counted} takes pinned to the {_RATE_CLAMP} atempo clamp "
                 "(spoken length far from slot)",
        ),
        # round_trip_wer + speaker_similarity are filled by tts_asr.py when faster-whisper /
        # speechbrain are installed; declared here so the report always lists them.
        missing("tts.round_trip_wer", "requires faster-whisper (bench extra)", unit="ratio"),
        missing("tts.speaker_similarity", "requires speechbrain ECAPA (bench extra)", unit="cosine"),
    ]
    return out
