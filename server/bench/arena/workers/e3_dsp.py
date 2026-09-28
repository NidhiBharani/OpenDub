"""E3 DSP workers (server env, CPU): ``passthrough`` (what OpenDub does today: the dry take is
mixed as is) and ``blind_parametric`` — a classical, model-free matcher.

``blind_parametric``: estimate T60 from the reverberant reference by the free-decay method (band
energy envelopes 250 Hz–4 kHz, linear fits over monotone decays of ≥ 10 dB after offsets; the
fastest decays are bounded by the room, so a low percentile of the implied T60s is taken — after
Ratnam et al. / Löllmann et al.), set DRR from a distance prior (longer rooms → lower DRR), render
an exponential-decay noise RIR (three bands, high band decaying faster) and convolve the dry
line, RMS-matched to the reference. The judge's synthetic rooms share this exponential-tail
family, which flatters this baseline slightly (noted in E3.yaml).

params: mode (passthrough | blind_parametric), t60_percentile (20), drr_at_03s (8 dB),
drr_slope (6 dB per doubling of T60).
"""
from __future__ import annotations

import math
from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    import numpy as np
    import soundfile as sf

    return {"np": np, "sf": sf, "p": params}


def estimate_t60(np, x, sr: int, pct: float) -> float:
    from scipy.signal import butter, sosfilt

    sos = butter(4, [250, min(4000, sr / 2 - 100)], "bandpass", fs=sr, output="sos")
    y = sosfilt(sos, x)
    hop, win = int(0.01 * sr), int(0.03 * sr)
    n = 1 + max(0, (len(y) - win) // hop)
    env = np.array([10 * np.log10(np.mean(y[i * hop:i * hop + win] ** 2) + 1e-12)
                    for i in range(n)])
    t60s, i = [], 0
    while i < n - 1:
        j = i
        while j + 1 < n and env[j + 1] < env[j] - 0.05:
            j += 1
        drop = env[i] - env[j]
        if drop >= 10 and (j - i) * hop / sr >= 0.08:
            slope = np.polyfit(np.arange(i, j + 1) * hop / sr, env[i:j + 1], 1)[0]
            if slope < 0:
                t60s.append(-60.0 / slope)
        i = max(j, i + 1)
    if not t60s:
        return 0.4
    return float(min(3.0, max(0.1, np.percentile(t60s, pct))))


def render_rir(np, t60: float, drr_db: float, sr: int, seed: int = 0):
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    n = int(min(2.0, 1.5 * t60 + 0.1) * sr)
    t = np.arange(n) / sr
    tail = np.zeros(n)
    for band, scale in (((None, 500.0), 1.15), ((500.0, 4000.0), 1.0), ((4000.0, None), 0.7)):
        lo, hi = band
        if lo is None:
            sos = butter(4, hi, "lowpass", fs=sr, output="sos")
        elif hi is None:
            sos = butter(4, min(lo, sr / 2 - 100), "highpass", fs=sr, output="sos")
        else:
            sos = butter(4, [lo, min(hi, sr / 2 - 100)], "bandpass", fs=sr, output="sos")
        tail += sosfilt(sos, rng.normal(0, 1, n)) * np.exp(-6.9078 * t / (t60 * scale))
    tail *= np.clip((t - 0.002) / 0.004, 0, 1)
    d0 = int(0.002 * sr)
    h = tail * math.sqrt(10 ** (-drr_db / 10) / max(float(np.sum(tail ** 2)), 1e-20))
    h[d0] += 1.0
    return h


def run(state: dict, item: dict, out: Path) -> dict:
    np, sf, p = state["np"], state["sf"], state["p"]
    x, sr = sf.read(item["inputs"]["audio"], dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    wav = out.with_suffix(".wav")
    if p.get("mode", "blind_parametric") == "passthrough":
        sf.write(wav, x, sr, subtype="PCM_16")
        return {"files": {"audio": str(wav)}, "params": {"mode": "passthrough"}}
    from scipy.signal import fftconvolve, resample_poly

    ref, rsr = sf.read(item["inputs"]["reference"], dtype="float64", always_2d=True)
    ref = ref.mean(axis=1)
    if rsr != sr:
        g = math.gcd(int(rsr), int(sr))
        ref = resample_poly(ref, sr // g, rsr // g)
    t60 = estimate_t60(np, ref, sr, float(p.get("t60_percentile", 20)))
    drr = float(p.get("drr_at_03s", 8.0)) - float(p.get("drr_slope", 6.0)) * math.log2(t60 / 0.3)
    drr = max(-3.0, min(15.0, drr))
    h = render_rir(np, t60, drr, sr)
    y = fftconvolve(x, h)[: len(x) + int(0.3 * sr)]
    y *= math.sqrt(float(np.mean(ref ** 2)) / max(float(np.mean(y ** 2)), 1e-20))
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y *= 0.99 / peak
    sf.write(wav, y, sr, subtype="PCM_16")
    rir = out.with_suffix(".rir.wav")
    sf.write(rir, h / np.max(np.abs(h)), sr, subtype="FLOAT")
    return {"files": {"audio": str(wav), "rir": str(rir)},
            "params": {"mode": "blind_parametric", "t60_s": round(t60, 3),
                       "drr_db": round(drr, 2)}}


if __name__ == "__main__":
    serve(load, run)
