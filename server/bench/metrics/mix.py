"""Mix-stage metrics — all reference-free, all would have caught the two audio bugs we hit:

- integrated loudness far from target  → the "dub buried under music" bug
- high noise floor in the silent lead-in → the dynamic-loudnorm hiss bug
- low dialogue-vs-bed margin in speech slots → dubs inaudible under the background
"""
from __future__ import annotations

import wave

from app.media import ffmpeg

from .base import Metric, MetricContext, missing

_TARGET_LUFS = -16.0


def _peak_dbfs(path, start_s: float, end_s: float) -> float | None:
    """Peak level (dBFS) of a wav window [start_s, end_s). None if the file is unreadable."""
    import audioop  # stdlib; deprecation is fine for an internal bench tool
    import math

    try:
        with wave.open(str(path)) as w:
            sr, width = w.getframerate(), w.getsampwidth()
            w.setpos(min(int(start_s * sr), w.getnframes()))
            n = max(0, int((end_s - start_s) * sr))
            frames = w.readframes(n)
    except (wave.Error, OSError, EOFError):
        return None
    if not frames:
        return -math.inf
    peak = audioop.max(frames, width)
    full = float(1 << (8 * width - 1))
    return 20 * math.log10(peak / full) if peak else -math.inf


async def score(ctx: MetricContext) -> list[Metric]:
    out: list[Metric] = []
    dub_mix = ctx.path("audio/dub_mix.wav")
    if not dub_mix.exists():
        return [missing("mix.integrated_lufs", "dub_mix.wav not found (mix stage did not run)")]

    # 1. Integrated loudness vs the -16 LUFS master target.
    lufs = await ffmpeg.measure_loudness(dub_mix)
    out.append(
        Metric(
            "mix.integrated_lufs", round(lufs, 2), "LUFS", "free",
            note=f"target {_TARGET_LUFS}; |Δ|={abs(lufs - _TARGET_LUFS):.1f}",
        )
    )
    out.append(
        Metric(
            "mix.loudness_error", round(abs(lufs - _TARGET_LUFS), 2), "LU", "free",
            higher_is_better=False, note="distance from -16 LUFS target",
        )
    )

    # 2. Noise floor in the silent lead-in (before the first dubbed line). A clean master sits
    #    well below the speech level here; the dynamic-loudnorm bug pushed it up to ~-47 dBFS.
    segs = sorted(ctx.project.segments, key=lambda s: s.start)
    first_start = segs[0].start if segs else 0.0
    if first_start > 0.4:
        floor = _peak_dbfs(dub_mix, 0.0, min(first_start - 0.1, first_start * 0.8))
        if floor is not None:
            out.append(
                Metric(
                    "mix.lead_in_noise_floor", round(floor, 1), "dBFS", "free",
                    higher_is_better=False, note=f"peak in silent lead-in [0,{first_start:.1f}s)",
                )
            )
    else:
        out.append(missing("mix.lead_in_noise_floor", "no silent lead-in before first segment"))

    # 3. Dialogue-vs-bed margin: dubbed-mix peak inside speech slots minus peak outside them.
    #    Positive → dialogue rides above the background bed; near-zero/negative → buried.
    in_peaks, out_peaks = [], []
    duration = ctx.project.media.duration if ctx.project.media else 0.0
    cursor = 0.0
    for s in segs:
        gap = _peak_dbfs(dub_mix, cursor, s.start) if s.start - cursor > 0.3 else None
        if gap is not None and gap != -float("inf"):
            out_peaks.append(gap)
        sp = _peak_dbfs(dub_mix, s.start, s.end)
        if sp is not None and sp != -float("inf"):
            in_peaks.append(sp)
        cursor = s.end
    if duration - cursor > 0.3:
        tail = _peak_dbfs(dub_mix, cursor, duration)
        if tail is not None and tail != -float("inf"):
            out_peaks.append(tail)

    if in_peaks and out_peaks:
        margin = sum(in_peaks) / len(in_peaks) - sum(out_peaks) / len(out_peaks)
        out.append(
            Metric(
                "mix.dialogue_bed_margin", round(margin, 1), "dB", "free", higher_is_better=True,
                note="mean speech-slot peak minus mean non-speech peak",
            )
        )
    else:
        out.append(missing("mix.dialogue_bed_margin", "not enough speech/non-speech regions"))

    return out
