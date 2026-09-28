"""A2 — reference-clip enhancement (denoise / dereverb / declip of cloning references).

Packs are clean speech with simulated degradations (``builders/a2_degrade.py``): the input is the
degraded clip, ``refs.clean`` the original. Ranked by **speaker-similarity preservation** against
the clean clip, from encoders disjoint from the A7 candidates' training recipe where possible
(WavLM-SV + ReDimNet2 panel, mean = ``spk_sim``). Naturalness (UTMOSv2, DNSMOS) and SI-SDR vs the
clean clip are secondary; generative enhancers are not sample-aligned, so SI-SDR undersells
them and never ranks. The gate is an ASR round-trip on the enhanced clip: an enhancer that
invents or drops phonemes (the appendix's SpeechBERTScore/LPS check) fails it.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..judgelib import asr_roundtrip, naturalness, speaker_similarity
from ..judges import ModelJudge, Row, Spec, cost, mean_of, output_file, speed
from ..packs import Item
from .a1 import load_mono, si_sdr


def _signal_rows(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    clean_path, est_path = item.refs.get("clean"), output_file(out, "audio")
    if not clean_path or not Path(str(clean_path)).exists() or not est_path:
        return [("missing_audio", 1.0, 1.0)] if not est_path else []
    import numpy as np

    clean, sr = load_mono(clean_path)
    est, _ = load_mono(est_path, sr)
    noisy, _ = load_mono(item.inputs["audio"], sr)
    rows: list[Row] = [("missing_audio", 0.0, 1.0)]
    val, base = si_sdr(clean, est), si_sdr(clean, noisy)
    if not math.isnan(val):
        rows.append(("sisdr", val, 1.0))
        if not math.isnan(base):
            rows.append(("sisdri", val - base, 1.0))
    # Log-spectral distance (dB) at the clean clip's rate: a coarse spectral-fidelity check
    # that, unlike SI-SDR, tolerates small phase / sample offsets.
    n = min(len(clean), len(est))
    if n > 2048:
        def spec(x):
            frames = np.lib.stride_tricks.sliding_window_view(x[:n], 1024)[::256]
            return np.log10(np.abs(np.fft.rfft(frames * np.hanning(1024), axis=1)) ** 2 + 1e-10)
        lsd = np.sqrt(((10 * (spec(clean) - spec(est))) ** 2).mean(axis=1)).mean()
        rows.append(("lsd_db", float(lsd), 1.0))
    # Level drift: enhancers that renormalise are fine, clipping is not.
    rows.append(("clipped_frac", float(np.mean(np.abs(est) >= 0.999)), 1.0))
    return rows


def _model_judges(lang: str) -> list[ModelJudge]:
    return [speaker_similarity("wavlm-sv", ref_key="clean"),
            speaker_similarity("redimnet2", ref_key="clean"),
            naturalness("utmosv2"), naturalness("dnsmos"),
            *asr_roundtrip(lang, text_of=lambda it, out: it.refs.get("text"))]


SPEC = Spec(
    id="A2",
    title="Reference-clip enhancement",
    judges={"signal@1": _signal_rows, "speed@1": speed, "cost@1": cost},
    model_judges=_model_judges,
    derived={"spk_sim": mean_of("sim_", ""), "rt_cer": mean_of("rt_", "_cer")},
    primary={"*": "spk_sim"},
    higher_is_better={"spk_sim": True, "sim_wavlm-sv": True, "sim_redimnet2": True,
                      "mos_utmosv2": True, "mos_dnsmos": True, "sisdr": True, "sisdri": True,
                      "lsd_db": False, "clipped_frac": False, "missing_audio": False,
                      "rt_cer": False, "rtfx": True, "cost_usd": False},
    threshold={"spk_sim": 0.01, "sisdri": 0.5},
    # Catastrophic content change (invented / dropped phonemes). Loose on purpose: absolute CERs
    # differ per language; the ranking itself never uses it.
    gates={"rt_cer": ("<=", 0.3)},
    secondary=["mos_utmosv2", "mos_dnsmos", "rt_cer", "sisdri", "lsd_db", "clipped_frac",
               "rtfx", "cost_usd"],
    packs=["a2-degrade"],
    io="""item.inputs: {audio: degraded clip}; item.refs: {clean: clean clip path, text};
item.meta.degradations: what was applied. payload: {files: {audio}} at the model's native rate
(16 kHz models are not upsampled; judges resample).""",
)
