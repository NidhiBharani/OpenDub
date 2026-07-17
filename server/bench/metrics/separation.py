"""Separation-stage metrics.

Reference-free: vocal-vs-background energy ratio (a healthy separation leaves the background
much quieter than the vocals in the vocal stem). Ground truth (when ref_vocals exists):
SI-SDR of the produced vocals stem against the reference vocal stem.
"""
from __future__ import annotations

from app.media import ffmpeg

from .base import Metric, MetricContext, missing


async def score(ctx: MetricContext) -> list[Metric]:
    vocals = ctx.path("audio/vocals.wav")
    background = ctx.path("audio/background.wav")
    out: list[Metric] = []

    if not vocals.exists():
        return [missing("separation.vocal_bg_ratio", "vocals.wav not found (separate did not run)")]

    # --- reference-free: loudness gap between the two stems ---
    if background.exists():
        v = await ffmpeg.measure_loudness(vocals)
        b = await ffmpeg.measure_loudness(background)
        out.append(
            Metric("separation.vocal_bg_lufs_gap", round(v - b, 1), "LU", "free",
                   higher_is_better=True,
                   note=f"vocals {v:.1f} vs background {b:.1f} LUFS; ~0 (passthrough) means no "
                        "real separation happened")
        )
    else:
        out.append(missing("separation.vocal_bg_lufs_gap", "background.wav not found"))

    # --- ground truth: SI-SDR ---
    out.append(await _si_sdr(ctx, vocals))
    return out


async def _si_sdr(ctx: MetricContext, vocals) -> Metric:
    if not ctx.case.has("ref_vocals"):
        return missing("separation.si_sdr", "no ref_vocals stem", provenance="gt", unit="dB")
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return missing("separation.si_sdr", "numpy/soundfile not installed (bench extra)",
                       provenance="gt", unit="dB")
    est, sr1 = sf.read(str(vocals))
    ref, sr2 = sf.read(str(ctx.case.ref_vocals))
    est, ref = _mono(est), _mono(ref)
    n = min(len(est), len(ref))
    if n == 0 or sr1 != sr2:
        return missing("separation.si_sdr", "empty or sample-rate-mismatched audio",
                       provenance="gt", unit="dB")
    est, ref = est[:n], ref[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
    proj = alpha * ref
    noise = est - proj
    si_sdr = 10 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(noise, noise) + 1e-12))
    return Metric("separation.si_sdr", round(float(si_sdr), 2), "dB", "gt", higher_is_better=True,
                  note="scale-invariant SDR of vocals stem vs ref_vocals")


def _mono(x):
    return x.mean(axis=1) if getattr(x, "ndim", 1) > 1 else x
