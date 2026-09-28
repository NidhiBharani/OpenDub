"""E4 — dialogue levelling + delivery loudness.

Items (``e4-fleurs-programmes`` / ``e4-arena-programmes``): one programme × one delivery preset::

    inputs: {dialogue: dub dialogue stem, background: M&E bed, reference_dialogue?: original
             dialogue stem, audio: premix (dialogue + bed at unity), speech: [[s, e], ...],
             delivery: {name, target_lufs, true_peak_dbtp, gate: programme|dialogue,
                        tolerance_lu}}
    refs:   {speech, lines: [[s, e, rel_lu], ...], speech_bed_db, delivery}
    payload: {files: {audio: final mix, dialogue?, background?}, params: {...}}

Metrics come from an in-house ITU-R BS.1770-5 meter (K-weighting with the libebur128 coefficient
formulas, 400 ms blocks at 75 % overlap, −70 LUFS absolute + −10 LU relative gate), 4× oversampled
true peak and EBU Tech 3342 LRA. Dialogue-gated loudness (Netflix −27 LKFS) keeps the blocks that
lie ≥ 50 % inside the known dialogue intervals and **replaces** the relative gate (the reason the
Netflix spec cites BS.1770-1), keeping only the −70 LUFS absolute gate. The meter is validated in
``tests/test_arena_e.py`` against EBU Tech 3341 reference signals.

Primary: |measured − target| in the delivery's gating. Gate: exact conformance (loudness within
the preset's tolerance and true peak ≤ its ceiling) on ≥ 99 % of items.
"""
from __future__ import annotations

import math
from typing import Any

from ..judges import Row, Spec, cost, output_file, speed
from ..packs import Item

# Delivery presets (E4 capability params in app/capabilities.py; research: compute-tiers E4).
DELIVERIES: dict[str, dict[str, Any]] = {
    "web": {"name": "web -16", "target_lufs": -16.0, "true_peak_dbtp": -1.0,
            "gate": "programme", "tolerance_lu": 1.0},
    "ebu": {"name": "EBU R128 -23", "target_lufs": -23.0, "true_peak_dbtp": -1.0,
            "gate": "programme", "tolerance_lu": 0.5},
    "atsc": {"name": "ATSC A/85 -24", "target_lufs": -24.0, "true_peak_dbtp": -2.0,
             "gate": "programme", "tolerance_lu": 2.0},
    "netflix": {"name": "Netflix -27", "target_lufs": -27.0, "true_peak_dbtp": -2.0,
                "gate": "dialogue", "tolerance_lu": 2.0},
}

ABS_GATE = -70.0
REL_GATE = -10.0


# ------------------------------------------------------------------ BS.1770-5 meter (numpy/scipy)

def k_weighting(sr: int):
    """(b, a) pairs of the two K-weighting biquads for any sample rate (libebur128 formulas)."""
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10 ** (g / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = ([(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
             [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    hp = ([1.0, -2.0, 1.0], [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    return [shelf, hp]


def channel_weights(n_ch: int) -> list[float]:
    """L, R, C = 1.0; Ls, Rs = 1.41; LFE excluded (5.1 order L R C LFE Ls Rs)."""
    if n_ch == 6:
        return [1.0, 1.0, 1.0, 0.0, 1.41, 1.41]
    if n_ch == 5:
        return [1.0, 1.0, 1.0, 1.41, 1.41]
    return [1.0] * n_ch


def _weighted(x, sr: int):
    import numpy as np
    from scipy.signal import lfilter

    x = np.atleast_2d(np.asarray(x, dtype=np.float64).T).T  # (n, ch)
    y = x.copy()
    for b, a in k_weighting(sr):
        y = lfilter(b, a, y, axis=0)
    return y


def block_powers(x, sr: int, block_s: float = 0.4, step_s: float = 0.1):
    """Per-block channel-weighted mean square z (after K-weighting) and block start times."""
    import numpy as np

    y = _weighted(x, sr)
    g = np.array(channel_weights(y.shape[1]))
    blk, hop = round(block_s * sr), round(step_s * sr)
    if len(y) < blk:
        return np.zeros(0), np.zeros(0)
    n = 1 + (len(y) - blk) // hop
    sq = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y ** 2, axis=0)])
    starts = hop * np.arange(n)
    ms = (sq[starts + blk] - sq[starts]) / blk            # (n, ch)
    return ms @ g, starts / sr


def _lufs(z) -> float:
    import numpy as np

    m = float(np.mean(z)) if len(z) else 0.0
    return -0.691 + 10 * math.log10(m) if m > 0 else -math.inf


def integrated_loudness(x, sr: int, *, relative_gate: bool = True,
                        keep=None) -> float:
    """BS.1770 gated integrated loudness (LUFS). ``keep`` optionally masks blocks (bool array)."""
    import numpy as np

    z, _ = block_powers(x, sr)
    if keep is not None:
        z = z[np.asarray(keep, bool)]
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10 * np.log10(np.maximum(z, 1e-20))
    z = z[lk > ABS_GATE]
    if not len(z):
        return -math.inf
    if relative_gate:
        rel = _lufs(z) + REL_GATE
        with np.errstate(divide="ignore"):
            z = z[-0.691 + 10 * np.log10(np.maximum(z, 1e-20)) > rel]
    return _lufs(z)


def dialogue_block_mask(sr: int, n_blocks: int, starts, speech: list[list[float]],
                        block_s: float = 0.4, min_frac: float = 0.5):
    import numpy as np

    keep = np.zeros(n_blocks, bool)
    for i, t0 in enumerate(starts):
        inside = sum(max(0.0, min(t0 + block_s, e) - max(t0, s)) for s, e in speech)
        keep[i] = inside >= min_frac * block_s
    return keep


def dialogue_gated_loudness(x, sr: int, speech: list[list[float]]) -> float:
    """Blocks ≥ 50 % inside dialogue, absolute gate only (the dialogue gate replaces the
    relative gate, as in the Netflix / Dolby Dialogue Intelligence practice)."""
    z, starts = block_powers(x, sr)
    keep = dialogue_block_mask(sr, len(z), starts, speech)
    return integrated_loudness(x, sr, relative_gate=False, keep=keep)


def true_peak_dbtp(x, sr: int) -> float:
    """BS.1770 Annex 2: 4× oversampling below 96 kHz (2× up to 192 kHz), max |sample|."""
    import numpy as np
    from scipy.signal import resample_poly

    x = np.atleast_2d(np.asarray(x, dtype=np.float64).T).T
    up = 4 if sr < 96000 else 2 if sr < 192000 else 1
    y = resample_poly(x, up, 1, axis=0) if up > 1 else x
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return 20 * math.log10(peak) if peak > 0 else -120.0


def loudness_range(x, sr: int) -> float:
    """EBU Tech 3342 LRA: short-term (3 s, 10 Hz) loudness, −70 abs / −20 LU rel gate,
    95th − 10th percentile."""
    import numpy as np

    z, _ = block_powers(x, sr, block_s=3.0, step_s=0.1)
    if not len(z):
        return 0.0
    with np.errstate(divide="ignore"):
        st = -0.691 + 10 * np.log10(np.maximum(z, 1e-20))
    st = st[st > ABS_GATE]
    if not len(st):
        return 0.0
    rel = -0.691 + 10 * math.log10(float(np.mean(10 ** ((st + 0.691) / 10)))) - 20.0
    st = st[st > rel]
    return float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) else 0.0


def window_loudness(x, sr: int, s: float, e: float) -> float:
    """Ungated K-weighted loudness of one window (line level)."""
    import numpy as np

    seg = np.atleast_2d(np.asarray(x).T).T[max(0, int(s * sr)):int(e * sr)]
    if len(seg) < int(0.1 * sr):
        return -math.inf
    y = _weighted(seg, sr)
    m = float(np.mean(y ** 2, axis=0) @ np.array(channel_weights(y.shape[1])))
    return -0.691 + 10 * math.log10(m) if m > 0 else -math.inf


def complement(speech: list[list[float]], total: float, pad: float = 0.25) -> list[list[float]]:
    """Non-dialogue windows (bed only), keeping ``pad`` seconds away from every line."""
    out, t = [], 0.0
    for s, e in sorted(speech):
        if s - pad > t + pad:
            out.append([t + pad, s - pad])
        t = max(t, e)
    if total - pad > t + pad:
        out.append([t + pad, total - pad])
    return out


def speech_bed_db(x, sr: int, speech: list[list[float]]) -> float | None:
    """Loudness inside dialogue windows minus loudness of the bed-only windows (a DBR proxy
    that needs no stems)."""
    import numpy as np

    total = np.shape(x)[0] / sr

    def energy(ivs):
        vals = [(window_loudness(x, sr, s, e), e - s) for s, e in ivs]
        vals = [(v, d) for v, d in vals if math.isfinite(v)]
        if not vals:
            return None
        return 10 * math.log10(sum(10 ** (v / 10) * d for v, d in vals) / sum(d for _, d in vals))

    sp, bed = energy(speech), energy(complement(speech, total))
    return None if sp is None or bed is None else sp - bed


def measure(x, sr: int, speech: list[list[float]] | None = None) -> dict[str, float]:
    out = {"integrated_lufs": integrated_loudness(x, sr), "true_peak_dbtp": true_peak_dbtp(x, sr),
           "lra_lu": loudness_range(x, sr)}
    if speech:
        out["dialogue_lufs"] = dialogue_gated_loudness(x, sr, speech)
    return out


def _load(path: str):
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return x, int(sr)


# ------------------------------------------------------------------ function judges

def conformance(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    audio = output_file(out)
    d = item.inputs.get("delivery") or item.refs.get("delivery")
    if not audio or not d:
        return []
    x, sr = _load(audio)
    speech = item.refs.get("speech") or item.inputs.get("speech") or []
    m = measure(x, sr, speech)
    level = m["dialogue_lufs"] if d.get("gate") == "dialogue" and speech else m["integrated_lufs"]
    if not math.isfinite(level):
        return [("loudness_err", 99.0, 1.0), ("conform", 0.0, 1.0)]
    err = abs(level - float(d["target_lufs"]))
    over = max(0.0, m["true_peak_dbtp"] - float(d["true_peak_dbtp"]))
    ok = err <= float(d.get("tolerance_lu", 1.0)) + 1e-9 and over <= 1e-9
    rows: list[Row] = [("loudness_err", err, 1.0), ("conform", float(ok), 1.0),
                       ("true_peak_over_db", over, 1.0), ("lra_lu", m["lra_lu"], 1.0)]
    if math.isfinite(m.get("dialogue_lufs", -math.inf)) and math.isfinite(m["integrated_lufs"]):
        # How far the programme-gated number sits from the dialogue-gated one: large gaps are
        # where a programme-gated chain misses a dialogue-gated spec.
        rows.append(("gating_gap_lu", abs(m["integrated_lufs"] - m["dialogue_lufs"]), 1.0))
    return rows


def balance(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    """Line-to-line level and dialogue-over-bed balance against the original programme."""
    audio = output_file(out)
    lines = item.refs.get("lines")
    if not audio or not lines:
        return []
    x, sr = _load(audio)
    rows: list[Row] = []
    got = [(window_loudness(x, sr, s, e), rel) for s, e, rel in lines]
    got = [(g, r) for g, r in got if math.isfinite(g) and r is not None]
    if len(got) >= 2:
        mg = sum(g for g, _ in got) / len(got)
        mr = sum(r for _, r in got) / len(got)
        errs = [abs((g - mg) - (r - mr)) for g, r in got]
        rows.append(("line_lu_err", sum(errs) / len(errs), float(len(errs))))
    ref_sb = item.refs.get("speech_bed_db")
    speech = item.refs.get("speech") or []
    if ref_sb is not None and speech:
        sb = speech_bed_db(x, sr, speech)
        if sb is not None:
            rows.append(("speech_bed_err_db", abs(sb - float(ref_sb)), 1.0))
    return rows


SPEC = Spec(
    id="E4",
    title="Dialogue levelling + delivery loudness",
    judges={"conformance@1": conformance, "balance@1": balance, "speed@1": speed,
            "cost@1": cost},
    primary={"*": "loudness_err"},
    higher_is_better={"loudness_err": False, "conform": True, "true_peak_over_db": False,
                      "lra_lu": False, "gating_gap_lu": False, "line_lu_err": False,
                      "speech_bed_err_db": False, "rtfx": True, "cost_usd": False},
    threshold={"loudness_err": 0.1},
    secondary=["conform", "true_peak_over_db", "line_lu_err", "speech_bed_err_db", "lra_lu",
               "gating_gap_lu", "rtfx", "cost_usd"],
    gates={"conform": (">=", 0.99)},
    packs=["e4-fleurs-programmes", "e4-arena-programmes"],
    io="""item.inputs: {dialogue, background, reference_dialogue?, audio (premix), speech:
[[s,e]], delivery: {name, target_lufs, true_peak_dbtp, gate: programme|dialogue, tolerance_lu}}.
item.refs: {speech, lines: [[s, e, rel_lu]], speech_bed_db, delivery}.
payload: {files: {audio: final mix, dialogue?, background?}, params?}. Keep 48 kHz.""",
)
