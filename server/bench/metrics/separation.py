"""Separation diagnostics and optional reference SI-SDR.

Relative stem loudness is descriptive only: a quiet vocal may be correctly separated.
"""
from __future__ import annotations

from app.media import ffmpeg
from app.pipeline.quality import audio_stats

from .base import Metric, MetricContext, missing


async def score(ctx: MetricContext) -> list[Metric]:
    vocals = ctx.path('audio/vocals.wav')
    background = ctx.path('audio/background.wav')
    vs, bs = audio_stats(vocals), audio_stats(background)
    out = []
    if vs['status'] != 'ok' or bs['status'] != 'ok':
        reason = f"vocals: {vs['status']} ({vs['reason']}); background: {bs['status']} ({bs['reason']})"
        out.extend([missing('separation.stem_duration_mismatch', reason, unit='s'),
                    missing('separation.vocal_bg_lufs_gap', reason, unit='LU'),
                    missing('separation.si_sdr', reason, provenance='gt', unit='dB')])
        return out
    out.append(Metric('separation.stem_duration_mismatch',
                      round(abs(vs['duration_s'] - bs['duration_s']), 3), 's', 'free',
                      higher_is_better=False))
    try:
        v, b = await ffmpeg.measure_loudness(vocals), await ffmpeg.measure_loudness(background)
        out.append(Metric('separation.vocal_bg_lufs_gap', round(v - b, 1), 'LU', 'free',
                          note='descriptive stem loudness difference; no separation quality implied'))
    except Exception as exc:  # noqa: BLE001 - ffmpeg failure is a missing measurement
        out.append(missing('separation.vocal_bg_lufs_gap', f'loudness measurement failed: {exc}',
                           unit='LU'))
    out.append(await _si_sdr(ctx, vocals))
    return out


async def _si_sdr(ctx: MetricContext, vocals) -> Metric:
    if not ctx.case.has('ref_vocals'):
        return missing('separation.si_sdr', 'no ref_vocals stem', provenance='gt', unit='dB')
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return missing('separation.si_sdr', 'numpy/soundfile not installed (bench extra)',
                       provenance='gt', unit='dB')
    try:
        est, sr1 = sf.read(str(vocals))
        ref, sr2 = sf.read(str(ctx.case.ref_vocals))
    except (OSError, RuntimeError, ValueError) as exc:
        return missing('separation.si_sdr', f'unreadable audio: {exc}', provenance='gt', unit='dB')
    est, ref = _mono(est), _mono(ref)
    n = min(len(est), len(ref))
    if n == 0 or sr1 != sr2:
        return missing('separation.si_sdr', 'empty or sample-rate-mismatched audio',
                       provenance='gt', unit='dB')
    est, ref = est[:n], ref[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
    proj = alpha * ref
    noise = est - proj
    si_sdr = 10 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(noise, noise) + 1e-12))
    return Metric('separation.si_sdr', round(float(si_sdr), 2), 'dB', 'gt', higher_is_better=True,
                  note='scale-invariant SDR of vocals stem vs ref_vocals')


def _mono(x):
    return x.mean(axis=1) if getattr(x, 'ndim', 1) > 1 else x
