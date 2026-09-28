"""E4 worker: BS.1770 two-pass static chain with delivery presets and dialogue gating — the
research's primary pick, implemented in-house around pyloudnorm (MIT).

1. (``levelling: true``) per-line levelling: each dub line is moved toward the original line's
   level relative to the original dialogue mean (from ``reference_dialogue`` over the known line
   windows), capped at ±``max_line_gain`` dB, with 20 ms ramps between windows;
2. dialogue gain to the original dialogue's integrated loudness (the original dialogue-over-bed
   balance; bed at unity), else dialogue 10 LU over the bed;
3. measure the premix in the delivery's gating — programme: pyloudnorm integrated loudness;
   dialogue: blocks ≥ 50 % inside the known dialogue windows, absolute gate only (the dialogue
   gate replaces the relative gate, Netflix / BS.1770-1 practice) — and apply one static gain;
4. 4× oversampled look-ahead true-peak limiter at the preset ceiling − ``tp_margin``; re-measure
   and correct the static gain (second pass), up to ``passes`` times.

params: levelling (true), max_line_gain (9), tp_margin (0.3), passes (3), lookahead_ms (5).
Output: 48 kHz 24-bit stereo mix + processed stems.
"""
from __future__ import annotations

import math
from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    import numpy as np
    import pyloudnorm
    import soundfile as sf

    return {"np": np, "sf": sf, "pyln": pyloudnorm, "p": params, "meters": {}}


def _meter(state: dict, sr: int):
    if sr not in state["meters"]:
        state["meters"][sr] = state["pyln"].Meter(sr)
    return state["meters"][sr]


def _kweight(state: dict, x, sr: int):
    """K-weighted copy via pyloudnorm's own filter stages."""
    y = state["np"].array(x, dtype=float, copy=True)
    for stage in _meter(state, sr)._filters.values():
        for ch in range(y.shape[1]):
            y[:, ch] = stage.apply_filter(y[:, ch])
    return y


def dialogue_lufs(state: dict, x, sr: int, speech: list) -> float:
    np = state["np"]
    y = _kweight(state, x, sr)
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    zs = []
    for k in range(0, max(0, len(y) - blk) + 1, hop):
        t0 = k / sr
        inside = sum(max(0.0, min(t0 + 0.4, e) - max(t0, s)) for s, e in speech)
        if inside >= 0.2:
            z = float(np.mean(y[k:k + blk] ** 2, axis=0).sum())
            if z > 0 and -0.691 + 10 * math.log10(z) > -70:
                zs.append(z)
    return -0.691 + 10 * math.log10(sum(zs) / len(zs)) if zs else -math.inf


def loud(state: dict, x, sr: int, gate: str, speech: list) -> float:
    if gate == "dialogue" and speech:
        return dialogue_lufs(state, x, sr, speech)
    return float(_meter(state, sr).integrated_loudness(x))


def window_lufs(state: dict, x, sr: int, s: float, e: float) -> float:
    np = state["np"]
    seg = x[int(s * sr):int(e * sr)]
    if len(seg) < int(0.1 * sr):
        return -math.inf
    z = float(np.mean(_kweight(state, seg, sr) ** 2, axis=0).sum())
    return -0.691 + 10 * math.log10(z) if z > 0 else -math.inf


def tp_limit(state: dict, y, sr: int, ceiling_db: float, lookahead_ms: float):
    np = state["np"]
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    from scipy.signal import resample_poly

    c = 10 ** (ceiling_db / 20)
    for _ in range(4):
        up = np.max(np.abs(resample_poly(y, 4, 1, axis=0)), axis=1)
        if up.max() <= c:
            break
        need = up[: len(y) * 4].reshape(len(y), 4).max(axis=1)  # resample_poly: exactly 4n
        gr = np.maximum(0.0, 20 * np.log10(np.maximum(need, 1e-12) / c))
        w = max(1, int(lookahead_ms / 1000 * sr))
        gr = uniform_filter1d(maximum_filter1d(gr, 2 * w + 1), w)
        y = y * (10 ** (-gr / 20))[:, None]
    return y


def run(state: dict, item: dict, out: Path) -> dict:
    np, sf, p = state["np"], state["sf"], state["p"]
    inp = item["inputs"]
    d = inp["delivery"]
    speech = inp.get("speech") or []
    dia, sr = sf.read(inp["dialogue"], dtype="float64", always_2d=True)
    bed, _ = sf.read(inp["background"], dtype="float64", always_2d=True)
    n = min(len(dia), len(bed))
    dia, bed = dia[:n], bed[:n]
    ref = None
    if inp.get("reference_dialogue"):
        ref, _ = sf.read(inp["reference_dialogue"], dtype="float64", always_2d=True)
        ref = ref[:n]
    line_gains = []
    if p.get("levelling", True) and ref is not None and len(speech) >= 2:
        lr = [window_lufs(state, ref, sr, s, e) for s, e in speech]
        ld = [window_lufs(state, dia, sr, s, e) for s, e in speech]
        ok = [i for i in range(len(speech)) if math.isfinite(lr[i]) and math.isfinite(ld[i])]
        if len(ok) >= 2:
            mr = sum(lr[i] for i in ok) / len(ok)
            md = sum(ld[i] for i in ok) / len(ok)
            cap = float(p.get("max_line_gain", 9.0))
            pts_t, pts_g = [0.0], [0.0]
            for i in ok:
                g = max(-cap, min(cap, (lr[i] - mr) - (ld[i] - md)))
                line_gains.append(round(g, 2))
                s, e = speech[i]
                pts_t += [max(0.0, s - 0.02), s, e, e + 0.02]
                pts_g += [pts_g[-1], g, g, g]
            order = np.argsort(pts_t, kind="stable")
            env = np.interp(np.arange(n) / sr, np.array(pts_t)[order], np.array(pts_g)[order])
            dia = dia * (10 ** (env / 20))[:, None]
    ld_int = float(_meter(state, sr).integrated_loudness(dia))
    if ref is not None and math.isfinite(ld_int):
        target_d = float(_meter(state, sr).integrated_loudness(ref))
    else:
        target_d = float(_meter(state, sr).integrated_loudness(bed)) + 10.0
    dgain = max(-30.0, min(30.0, target_d - ld_int)) if math.isfinite(ld_int) else 0.0
    dia = dia * 10 ** (dgain / 20)
    mix = dia + bed
    gate, target = d.get("gate", "programme"), float(d["target_lufs"])
    ceiling = float(d["true_peak_dbtp"]) - float(p.get("tp_margin", 0.3))
    gain, y = 0.0, mix
    for _ in range(int(p.get("passes", 3))):
        level = loud(state, y, sr, gate, speech)
        if not math.isfinite(level):
            break
        gain += target - level
        y = tp_limit(state, mix * 10 ** (gain / 20), sr, ceiling,
                     float(p.get("lookahead_ms", 5.0)))
        if abs(loud(state, y, sr, gate, speech) - target) < 0.05:
            break
    wav = out.with_suffix(".wav")
    sf.write(wav, y, sr, subtype="PCM_24")
    stems = {}
    for name, arr in (("dialogue", dia), ("background", bed)):
        path = out.with_suffix(f".{name}.wav")
        sf.write(path, np.clip(arr * 10 ** (gain / 20), -1, 1), sr, subtype="PCM_24")
        stems[name] = str(path)
    return {"files": {"audio": str(wav), **stems},
            "params": {"dialogue_gain_db": round(dgain, 2), "master_gain_db": round(gain, 2),
                       "line_gains_db": line_gains, "gate": gate, "ceiling_dbtp": ceiling}}


def describe(state: dict) -> dict:
    return {"pyloudnorm": getattr(state["pyln"], "__version__", "?")}


if __name__ == "__main__":
    serve(load, run, describe)
