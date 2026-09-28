"""Mix loudness and same-window stem measurements.

Lead-ins can contain music/effects, so their level is reported as signal, never noise.
"""
from __future__ import annotations

import json
import math

from app.media import ffmpeg
from app.pipeline.quality import audio_stats, window_rms

from .base import Metric, MetricContext, missing

_TARGET_LUFS = -16.0


def _gain_metadata(ctx: MetricContext) -> dict | None:
    """Read persisted mix gains when a pipeline version reports both contributions."""
    path = ctx.path('quality/mix_levels.json')
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return None
        float(data['vocal_gain_db'])
        float(data['background_gain_db'])
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


async def score(ctx: MetricContext) -> list[Metric]:
    dub_mix = ctx.path('audio/dub_mix.wav')
    stats = audio_stats(dub_mix)
    if stats['status'] != 'ok':
        reason = 'not found' if stats['status'] == 'missing' else 'unreadable'
        return [missing('mix.integrated_lufs', f"dub_mix.wav {reason}: {stats['reason']}")]

    out = [Metric('mix.sample_peak', round(stats['peak_dbfs'], 2), 'dBFS', 'free',
                  higher_is_better=False,
                  note='PCM sample peak; inter-sample true peak requires oversampled measurement'),
           Metric('mix.clipping_fraction', round(stats['clipping_fraction'], 6), 'ratio', 'free',
                  higher_is_better=False, note='samples at or above -0.1 dBFS')]
    try:
        true_peak = await ffmpeg.measure_true_peak(dub_mix)
    except Exception:  # noqa: BLE001 - optional meter failure is explicitly unavailable
        true_peak = None
    out.append(Metric('mix.true_peak', true_peak, 'dBFS', 'free', higher_is_better=False,
                      note='ffmpeg ebur128 oversampled peak' if true_peak is not None else
                           'ffmpeg true-peak measurement unavailable'))
    gains = _gain_metadata(ctx)
    try:
        lufs = await ffmpeg.measure_loudness(dub_mix)
        target = float((gains or {}).get('target_lufs', _TARGET_LUFS))
        out.extend([Metric('mix.integrated_lufs', round(lufs, 2), 'LUFS', 'free',
                           note=f'target {target} LUFS'),
                    Metric('mix.loudness_error', round(abs(lufs - target), 2), 'LU',
                           'free', higher_is_better=False)])
    except Exception as exc:  # noqa: BLE001 - ffmpeg failure is a missing measurement
        out.extend([missing('mix.integrated_lufs', f'loudness measurement failed: {exc}'),
                    missing('mix.loudness_error', f'loudness measurement failed: {exc}')])

    segs = sorted((s for s in ctx.project.segments if not s.skipped), key=lambda s: s.start)
    first_start = segs[0].start if segs else 0.0
    if first_start > 0.4:
        lead_rms = window_rms(dub_mix, 0, min(first_start - 0.1, first_start * 0.8))
        out.append(Metric('mix.lead_in_level', round(20 * math.log10(lead_rms), 2)
                          if lead_rms and lead_rms > 0 else -120.0 if lead_rms == 0 else None,
                          'dBFS', 'free', note='RMS of lead-in; may be intended music/effects'))
    else:
        out.append(missing('mix.lead_in_level', 'no lead-in before first dub', unit='dBFS'))
    out.append(missing('mix.lead_in_noise_floor',
                       'lead-in may contain intended music/effects; no isolated noise reference',
                       unit='dBFS'))

    vocal = ctx.path('audio/dub_vocals.wav')
    background = ctx.path('audio/background.wav')
    margins = []
    for segment in segs:
        if segment.end <= segment.start:
            continue
        vr = window_rms(vocal, segment.start, segment.end)
        br = window_rms(background, segment.start, segment.end)
        if vr is None or br is None or vr <= 0 or br <= 0:
            continue
        margin = 20 * math.log10(vr / br)
        if gains:
            margin += float(gains['vocal_gain_db']) - float(gains['background_gain_db'])
        margins.append(margin)
    if margins:
        name = 'mix.dialogue_bed_margin' if gains else 'mix.raw_stem_margin'
        out.append(Metric(name, round(sum(margins) / len(margins), 2), 'dB', 'free',
                          higher_is_better=True if gains else None,
                          note=('same-window static-gain estimate; limiting/ducking may alter balance' if gains else
                                'same-window raw stem RMS; final applied gains unavailable')))
    else:
        out.append(missing('mix.dialogue_bed_margin',
                           'no comparable dubbed vocal/background samples in speech windows', unit='dB'))
    if not gains:
        out.append(missing('mix.dialogue_bed_margin',
                           'applied stem gains unavailable; raw_stem_margin is descriptive', unit='dB'))
    return out
