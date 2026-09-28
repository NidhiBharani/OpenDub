"""E3 — acoustic scene / room matching (parked in the plan; spec and packs ready).

Items (``e3-synth-rooms``): a dry line and a *reference* — another utterance convolved with a
synthetic room impulse response (RIR) whose T60/DRR/C50 are measured from the RIR itself::

    inputs: {audio: dry line, reference: reverberant reference (the "original dialogue stem")}
    refs:   {rir, t60_s, drr_db, c50_db, text}
    payload: {files: {audio: the line rendered into the reference's room, rir?}, params?}

The judge needs no model and no candidate cooperation: the dry input is known, so the room the
candidate actually applied is recovered by regularised (Wiener) deconvolution of output by input,
and T60 (Schroeder backward integral with noise compensation, T20 → T10 fallback), DRR (±2.5 ms
around the direct peak) and C50 are measured on that estimate exactly as on the reference RIR.

Primary: ``room_err_jnd`` = mean of |ΔT60|, |ΔDRR|, |ΔC50| in just-noticeable differences
(5 % of T60 — ISO 3382-1; 2 dB DRR — Larsen et al. 2008; 1 dB C50 — ISO 3382-1 clarity), each
capped at 20 JND so a dry passthrough is maximally wrong but finite.
"""
from __future__ import annotations

import math
from typing import Any

from ..judgelib import asr_roundtrip
from ..judges import ModelJudge, Row, Spec, cost, mean_of, output_file, speed
from ..packs import Item
from .e1 import delta_vs_input, on_input

JND = {"t60": 0.05, "drr": 2.0, "c50": 1.0}   # t60 is relative
CAP_JND = 20.0
DIRECT_S = 0.0025


def _mono(path: str, sr: int | None = None):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr and fs != sr:
        g = math.gcd(int(fs), int(sr))
        x = resample_poly(x, int(sr) // g, int(fs) // g)
        fs = sr
    return np.asarray(x), int(fs)


def estimate_rir(dry, wet, sr: int, max_s: float = 2.5, reg: float = 1e-3):
    """Wiener deconvolution: the linear filter h with wet ≈ dry * h, from 5 ms before the direct
    peak to ``max_s`` after it."""
    import numpy as np

    n = 1 << int(np.ceil(np.log2(len(dry) + len(wet))))
    X, Y = np.fft.rfft(dry, n), np.fft.rfft(wet, n)
    pw = np.abs(X) ** 2
    h = np.fft.irfft(Y * np.conj(X) / (pw + reg * float(np.mean(pw)) + 1e-20), n)
    peak = int(np.argmax(np.abs(h[: len(wet)])))
    start = max(0, peak - int(0.005 * sr))
    return h[start:start + int(max_s * sr)]


def rir_params(h, sr: int) -> dict[str, float]:
    """T60 (s), DRR (dB), C50 (dB) of an impulse response."""
    import numpy as np

    h = np.asarray(h, dtype=np.float64)
    peak = int(np.argmax(np.abs(h)))
    e = h[peak:] ** 2
    if not len(e) or e.sum() <= 0:
        return {"t60_s": 0.0, "drr_db": 30.0, "c50_db": 30.0}
    d = int(DIRECT_S * sr)
    direct = float(np.sum(h[max(0, peak - d):peak + d + 1] ** 2))
    rev = float(np.sum(h[peak + d + 1:] ** 2))
    drr = 10 * math.log10(direct / rev) if rev > 0 else 30.0
    k50 = int(0.05 * sr)
    late = float(e[k50:].sum())
    c50 = 10 * math.log10(float(e[:k50].sum()) / late) if late > 0 else 30.0
    # Noise-compensated Schroeder integral: subtract the floor seen in the last 10 %.
    tail = e[int(0.9 * len(e)):]
    floor = float(np.mean(tail)) if len(tail) > 10 else 0.0
    ec = np.maximum(e - floor, 0.0)
    edc = np.cumsum(ec[::-1])[::-1]
    t60 = 0.0
    if edc[0] > 0:
        edc_db = 10 * np.log10(np.maximum(edc / edc[0], 1e-12))
        t = np.arange(len(edc_db)) / sr
        for lo, hi in ((-5.0, -25.0), (-5.0, -15.0)):
            sel = (edc_db <= lo) & (edc_db >= hi)
            if sel.sum() >= max(8, int(0.005 * sr)) and edc_db.min() < hi:
                slope = np.polyfit(t[sel], edc_db[sel], 1)[0]
                if slope < 0:
                    t60 = float(-60.0 / slope)
                    break
    return {"t60_s": t60, "drr_db": max(-30.0, min(30.0, drr)),
            "c50_db": max(-30.0, min(30.0, c50))}


def room_errors(est: dict[str, float], ref: dict[str, float]) -> list[Row]:
    t60_ref = max(float(ref["t60_s"]), 0.05)
    d_t60 = abs(est["t60_s"] - t60_ref)
    d_drr = abs(est["drr_db"] - float(ref["drr_db"]))
    d_c50 = abs(est["c50_db"] - float(ref["c50_db"]))
    jnd = [min(CAP_JND, d_t60 / (JND["t60"] * t60_ref)), min(CAP_JND, d_drr / JND["drr"]),
           min(CAP_JND, d_c50 / JND["c50"])]
    return [("room_err_jnd", sum(jnd) / 3, 1.0), ("t60_err_s", d_t60, 1.0),
            ("t60_rel_err", d_t60 / t60_ref, 1.0), ("drr_err_db", d_drr, 1.0),
            ("c50_err_db", d_c50, 1.0)]


def room_match(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    audio, dry = output_file(out), item.inputs.get("audio")
    if not audio or not dry or item.refs.get("t60_s") is None:
        return []
    x, sr = _mono(dry)
    y, _ = _mono(audio, sr)
    est = rir_params(estimate_rir(x, y, sr), sr)
    return room_errors(est, item.refs)


def _text(item: Item, out: dict[str, Any]) -> str | None:
    return item.refs.get("text")


def _model_judges(lang: str) -> list[ModelJudge]:
    asr = asr_roundtrip(lang, text_of=_text)
    return [*asr, *[on_input(j) for j in asr]]


SPEC = Spec(
    id="E3",
    title="Acoustic scene / room matching",
    judges={"room@1": room_match, "speed@1": speed, "cost@1": cost},
    primary={"*": "room_err_jnd"},
    higher_is_better={"room_err_jnd": False, "t60_err_s": False, "t60_rel_err": False,
                      "drr_err_db": False, "c50_err_db": False, "rt_cer": False,
                      "d_rt_cer": False, "rtfx": True, "cost_usd": False},
    threshold={"room_err_jnd": 0.5},
    secondary=["t60_rel_err", "drr_err_db", "c50_err_db", "d_rt_cer", "rtfx", "cost_usd"],
    model_judges=_model_judges,
    derived={"rt_cer": mean_of("rt_", "_cer"), "d_rt_cer": delta_vs_input("rt_", "_cer")},
    packs=["e3-synth-rooms"],
    io="""item.inputs: {audio: dry line, reference: reverberant reference line}.
item.refs: {rir, t60_s, drr_db, c50_db, text}. payload: {files: {audio, rir?}, params?:
{t60_s, drr_db, c50_db}}. The output must stay time-aligned with the dry input (a pure delay is
tolerated; the judge locks onto the direct peak).""",
)
