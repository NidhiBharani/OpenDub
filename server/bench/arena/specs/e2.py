"""E2 — bandwidth extension / 48 kHz restoration of TTS takes.

Two item kinds share one pack:

- **simulated** (``e2-vctk-sim``): clean 48 kHz VCTK test speech (CC BY 4.0, the 8 test speakers
  every VCTK-trained BWE model holds out) low-passed by resampling to 8/16/24 kHz;
  ``refs.clean`` is the 48 kHz original, so ViSQOL (audio mode, 48 kHz) and log-spectral distance
  are defined.
- **real** (``e2-arena-takes``): D-phase TTS outputs at their native rate; no clean reference, so
  only effective bandwidth, Audiobox Aesthetics PQ and the Δ gates apply.

::

    inputs: {audio: low-rate input}          meta: {input_sr, cutoff_hz}
    refs:   {clean?: 48 kHz original, ref_audio?: other clean clip of the speaker, text}
    payload: {files: {audio: 48 kHz output}}

Primary: ViSQOL MOS-LQO (audio mode) for simulated items. Gates (appendix): round-trip CER may rise
by at most 0.5 points (0.005) and speaker similarity may drop by at most 0.01 against the
unprocessed input — BWE must not invent phonemes or drift timbre. LSD-HF (above the input's
Nyquist) is diagnostic.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..hardware import Requires
from ..judgelib import asr_roundtrip, naturalness, speaker_similarity
from ..judges import ModelJudge, Row, Spec, cost, mean_of, output_file, speed
from ..packs import Item
from .e1 import delta_vs_input, on_input

TARGET_SR = 48000


def _load_at(path: str, sr: int):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        g = math.gcd(int(fs), sr)
        x = resample_poly(x, sr // g, int(fs) // g)
    return np.asarray(x)


def _power_spec(x, n_fft: int = 2048, hop: int = 512):
    import numpy as np

    win = np.hanning(n_fft)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (frames, bins)


def lsd(ref, est, sr: int = TARGET_SR, f_lo: float = 0.0, n_fft: int = 2048) -> float:
    """Log-spectral distance (NU-Wave / AP-BWE convention: log10 power, RMS over frequency,
    mean over frames) on bins ≥ ``f_lo``."""
    import numpy as np

    n = min(len(ref), len(est))
    pr, pe = _power_spec(ref[:n], n_fft), _power_spec(est[:n], n_fft)
    lo = int(np.ceil(f_lo / (sr / n_fft)))
    d = np.log10(pr[:, lo:] + 1e-10) - np.log10(pe[:, lo:] + 1e-10)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=1))))


def effective_bandwidth_hz(x, sr: int = TARGET_SR, drop_db: float = 50.0) -> float:
    """Highest frequency whose long-term spectrum stays within ``drop_db`` of the 1–4 kHz level
    (smoothed over 1/6 octave-ish 16-bin windows). Resampled 16 kHz speech reads ≈ 8 kHz;
    full-band speech reads well above 16 kHz."""
    import numpy as np

    p = _power_spec(x).mean(axis=0)
    freqs = np.fft.rfftfreq(2048, 1 / sr)
    db = 10 * np.log10(np.convolve(p, np.ones(16) / 16, mode="same") + 1e-20)
    band = (freqs >= 1000) & (freqs <= 4000)
    ref = float(np.median(db[band]))
    above = np.nonzero(db >= ref - drop_db)[0]
    return float(freqs[above[-1]]) if len(above) else 0.0


def spectral(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    audio = output_file(out)
    if not audio:
        return []
    y = _load_at(audio, TARGET_SR)
    rows: list[Row] = [("eff_bw_khz", effective_bandwidth_hz(y) / 1000, 1.0)]
    clean = item.refs.get("clean")
    if clean and Path(clean).exists():
        x = _load_at(clean, TARGET_SR)
        cutoff = float(item.meta.get("cutoff_hz") or (item.meta.get("input_sr", 16000) / 2))
        rows += [("lsd", lsd(x, y), 1.0), ("lsd_hf", lsd(x, y, f_lo=cutoff), 1.0)]
    return rows


def metric_rows(payload: dict[str, Any]) -> list[Row]:
    """Judge-worker payload ``{metrics: {name: value}, weights?: {...}}`` → rows."""
    w = payload.get("weights") or {}
    return [(k, float(v), float(w.get(k, 1.0))) for k, v in (payload.get("metrics") or {}).items()
            if v is not None]


def visqol(version: int = 1) -> ModelJudge:
    """ViSQOL v3 audio mode (48 kHz) via visqol-python (pure-Python port, 12/12 conformance tests
    against the C++ reference per its README). Needs ``refs.clean``."""
    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        audio, clean = output_file(out), item.refs.get("clean")
        return {"audio": audio, "reference": clean} if audio and clean else None

    return ModelJudge(id=f"visqol.audio@{version}", worker="e2_judge_visqol", env="e2_visqol",
                      params={"mode": "audio", "sr": TARGET_SR}, requires=Requires(),
                      lang_param=False, inputs=inputs,
                      to_rows=lambda item, payload, out: metric_rows(payload))


def _text(item: Item, out: dict[str, Any]) -> str | None:
    return item.refs.get("text") or item.inputs.get("text")


def _model_judges(lang: str) -> list[ModelJudge]:
    gated = [*asr_roundtrip(lang, text_of=_text), speaker_similarity("wavlm-sv")]
    return [visqol(), *gated, naturalness("audiobox"), *[on_input(j) for j in gated]]


SPEC = Spec(
    id="E2",
    title="Bandwidth extension / 48 kHz restoration",
    judges={"spectral@1": spectral, "speed@1": speed, "cost@1": cost},
    # hi/ja have no public 48 kHz multi-speaker set wired yet (only real TTS takes): they rank
    # on Audiobox PQ until a simulated pack exists (``e2-vctk-sim --local-dir`` takes any corpus).
    primary={"*": "visqol", "hi": "aes_pq", "ja": "aes_pq"},
    higher_is_better={"visqol": True, "lsd": False, "lsd_hf": False, "eff_bw_khz": True,
                      "aes_pq": True, "aes_ce": True, "aes_cu": True, "aes_pc": True,
                      "rt_cer": False, "d_rt_cer": False, "d_sim": True, "rtfx": True,
                      "cost_usd": False},
    threshold={"visqol": 0.05, "aes_pq": 0.1},
    secondary=["lsd", "lsd_hf", "eff_bw_khz", "aes_pq", "d_rt_cer", "d_sim", "rtfx", "cost_usd"],
    model_judges=_model_judges,
    derived={"rt_cer": mean_of("rt_", "_cer"), "d_rt_cer": delta_vs_input("rt_", "_cer"),
             "d_sim": delta_vs_input("sim_")},
    gates={"d_rt_cer": ("<=", 0.005), "d_sim": (">=", -0.01)},
    packs=["e2-vctk-sim", "e2-arena-takes"],
    io="""item.inputs: {audio: low-rate input (native rate)}; item.meta: {input_sr, cutoff_hz}.
item.refs: {clean?: 48 kHz original, ref_audio?: another clean clip of the speaker, text}.
payload: {files: {audio: 48 kHz output, same length as the input}}.""",
)
