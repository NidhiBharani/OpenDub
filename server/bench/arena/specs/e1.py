"""E1 — timing fit + pause mapping: land a take in its slot with pauses where the source pauses.

Items (``e1-fleurs-synth`` / ``e1-arena-takes`` builders)::

    inputs: {audio: take, target_s: slot length, src_pauses: [[s, e], ...] (slot timeline),
             src_speech: [[s, e], ...]}
    refs:   {text, pauses: [[s, e], ...], speech: [[s, e], ...], target_s}
    payload: {files: {audio}, time_map: [[t_in, t_out], ...]?, params: {...}}

``src_pauses`` are what the pipeline knows from the source line's word timings (A4); the ranking
asks whether the fitted take pauses there too. Primary: pause-alignment F1 (a pause ≥ 150 ms in the
output matches a reference pause when both its edges are within ±100 ms), aggregated as a corpus
micro-F1 (per-item weight = 2·TP + FP + FN). Gates: the stretch must not cost intelligibility or
naturalness relative to the unfitted take (Δ round-trip CER, Δ UTMOSv2, both judged on the input
take too), and the fitted take must not overflow its slot.

The speech/pause detector here is shared by the judges and the builders (so the reference pauses
are measured the same way as the outputs); candidates carry their own.
"""
from __future__ import annotations

import dataclasses
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..judgelib import asr_roundtrip, naturalness
from ..judges import ModelJudge, Row, Spec, cost, mean_of, output_file, speed
from ..packs import Item

MIN_PAUSE_S = 0.15     # pauses shorter than this are not pauses (appendix: pauses ≥ 150 ms)
PAUSE_TOL_S = 0.10     # ±100 ms on each pause edge
OVERFLOW_TOL_S = 0.02  # the app tolerates encoder rounding of ~2 ms; 20 ms is audibly nothing


# ------------------------------------------------------------------ shared DSP (numpy)

def load_mono(path: str | Path):
    """(float64 mono samples, sample rate)."""
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return np.asarray(x.mean(axis=1)), int(sr)


def speech_intervals(x, sr: int, *, hop_s: float = 0.01, win_s: float = 0.02,
                     min_pause: float = MIN_PAUSE_S, min_speech: float = 0.05,
                     rel_db: float = 30.0, floor_db: float = -65.0) -> list[list[float]]:
    """Energy speech mask → [[start, end], ...] seconds.

    Frame level vs an adaptive threshold (max of: 95th percentile − ``rel_db``, 10th percentile +
    12 dB, ``floor_db``); gaps shorter than ``min_pause`` are bridged, islands shorter than
    ``min_speech`` dropped. Deterministic and model-free: good enough for read speech and TTS.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    hop, win = max(1, round(hop_s * sr)), max(1, round(win_s * sr))
    if len(x) < win:
        return []
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    db = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    thr = max(float(np.percentile(db, 95)) - rel_db, float(np.percentile(db, 10)) + 12.0,
              floor_db)
    active = db > thr
    runs: list[list[float]] = []
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            runs.append([(i * hop + (win - hop) / 2) / sr, (j * hop + (win + hop) / 2) / sr])
            i = j + 1
        else:
            i += 1
    merged: list[list[float]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < min_pause:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [[round(s, 4), round(e, 4)] for s, e in merged if e - s >= min_speech]


def internal_pauses(speech: list[list[float]], min_pause: float = MIN_PAUSE_S) -> list[list[float]]:
    """Gaps between speech runs (leading/trailing silence is not a pause)."""
    return [[a[1], b[0]] for a, b in pairwise(speech)
            if b[0] - a[1] >= min_pause]


def match_pauses(ref: list[list[float]], hyp: list[list[float]],
                 tol: float = PAUSE_TOL_S) -> int:
    """True positives: one-to-one greedy matching, both edges within ``tol``."""
    pairs = sorted((max(abs(r[0] - h[0]), abs(r[1] - h[1])), i, j)
                   for i, r in enumerate(ref) for j, h in enumerate(hyp))
    used_r: set[int] = set()
    used_h: set[int] = set()
    tp = 0
    for err, i, j in pairs:
        if err > tol:
            break
        if i not in used_r and j not in used_h:
            used_r.add(i)
            used_h.add(j)
            tp += 1
    return tp


def mask_iou(a: list[list[float]], b: list[list[float]], total: float, step: float = 0.01) -> float:
    import numpy as np

    n = max(1, int(np.ceil(total / step)))
    ma, mb = np.zeros(n, bool), np.zeros(n, bool)
    for m, ivs in ((ma, a), (mb, b)):
        for s, e in ivs:
            m[max(0, int(s / step)):max(0, int(np.ceil(e / step)))] = True
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 1.0


def _overlap(a0: float, a1: float, ivs: list[list[float]]) -> float:
    return sum(max(0.0, min(a1, e) - max(a0, s)) for s, e in ivs)


def stretch_profile(time_map: list[list[float]] | None, in_speech: list[list[float]],
                    out_speech: list[list[float]]) -> tuple[float, float] | None:
    """(max |rate−1|, speech-weighted mean |rate−1|) over speech-bearing time-map segments.

    rate = input seconds / output seconds (>1 = sped up). Without a time map: one global rate,
    total input speech over total output speech."""
    segs: list[tuple[float, float]] = []  # (|rate-1|, speech weight)
    if time_map and len(time_map) >= 2:
        for (i0, o0), (i1, o1) in pairwise(time_map):
            if o1 - o0 <= 1e-4 or i1 - i0 <= 1e-4:
                continue
            w = _overlap(i0, i1, in_speech)
            if w >= 0.02:
                segs.append((abs((i1 - i0) / (o1 - o0) - 1.0), w))
    if not segs:
        tin = sum(e - s for s, e in in_speech)
        tout = sum(e - s for s, e in out_speech)
        if tin <= 0 or tout <= 0:
            return None
        segs = [(abs(tin / tout - 1.0), tin)]
    total = sum(w for _, w in segs)
    return max(d for d, _ in segs), sum(d * w for d, w in segs) / total


# ------------------------------------------------------------------ function judges

def pause_alignment(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    audio = output_file(out)
    ref = item.refs.get("pauses")
    if not audio or ref is None:
        return []
    x, sr = load_mono(audio)
    hyp = internal_pauses(speech_intervals(x, sr))
    tp = match_pauses(ref, hyp)
    denom = len(ref) + len(hyp)
    if denom == 0:
        return []
    return [("pause_f1", 2 * tp / denom, float(denom)),
            ("pause_recall", tp / len(ref) if ref else 1.0, float(len(ref) or 1)),
            ("pause_precision", tp / len(hyp) if hyp else 1.0, float(len(hyp) or 1))]


def timing(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    audio = output_file(out)
    target = item.refs.get("target_s", item.inputs.get("target_s"))
    if not audio or not target:
        return []
    x, sr = load_mono(audio)
    dur = len(x) / sr
    speech = speech_intervals(x, sr)
    rows: list[Row] = [("dur_err_ms", abs(dur - float(target)) * 1000, 1.0),
                       ("overflow", float(dur > float(target) + OVERFLOW_TOL_S), 1.0)]
    ref_speech = item.refs.get("speech")
    if ref_speech and speech:
        rows.append(("onset_err_ms", abs(speech[0][0] - ref_speech[0][0]) * 1000, 1.0))
        rows.append(("speech_overlap", mask_iou(speech, ref_speech, max(dur, float(target))),
                     1.0))
    take = item.inputs.get("audio")
    if take and Path(take).exists():
        tx, tsr = load_mono(take)
        prof = stretch_profile(out.get("time_map"), speech_intervals(tx, tsr), speech)
        if prof is not None:
            rows += [("max_stretch", prof[0], 1.0), ("mean_stretch", prof[1], 1.0)]
    return rows


# ------------------------------------------------------------------ model judges on the input

def on_input(judge: ModelJudge, input_key: str = "audio") -> ModelJudge:
    """The same model judge applied to the item's *input* audio, metrics prefixed ``in_``.

    Lets a spec gate on "no worse than before processing" (Δ = output − input) without a core
    change. The input is re-judged once per candidate output (cached like any judge output)."""
    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        src = item.inputs.get(input_key)
        if not src or not Path(src).exists():
            return None
        return judge.inputs(item, {"files": {"audio": src}}, prefix)

    def to_rows(item: Item, payload: dict[str, Any], out: dict[str, Any]) -> list[Row]:
        return [(f"in_{m}", v, w) for m, v, w in judge.to_rows(item, payload, out)]

    return dataclasses.replace(judge, id=f"input.{judge.id}", inputs=inputs, to_rows=to_rows)


def delta_vs_input(prefix: str, suffix: str = ""):
    """Derived: mean over judges of (output metric − the same metric on the input)."""
    def fn(values: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
        diffs = [(v - values[f"in_{k}"][0], w) for k, (v, w) in values.items()
                 if k.startswith(prefix) and k.endswith(suffix) and f"in_{k}" in values]
        if not diffs:
            return None
        return (sum(d for d, _ in diffs) / len(diffs), sum(w for _, w in diffs) / len(diffs))
    return fn


def _text(item: Item, out: dict[str, Any]) -> str | None:
    return item.refs.get("text") or item.inputs.get("text")


def _model_judges(lang: str) -> list[ModelJudge]:
    base = [*asr_roundtrip(lang, text_of=_text), naturalness("utmosv2")]
    return base + [on_input(j) for j in base]


SPEC = Spec(
    id="E1",
    title="Timing fit + pause mapping",
    judges={"pauses@1": pause_alignment, "timing@1": timing, "speed@1": speed, "cost@1": cost},
    primary={"*": "pause_f1"},
    higher_is_better={"pause_f1": True, "pause_recall": True, "pause_precision": True,
                      "dur_err_ms": False, "overflow": False, "onset_err_ms": False,
                      "speech_overlap": True, "max_stretch": False, "mean_stretch": False,
                      "rt_cer": False, "d_rt_cer": False, "d_mos": True, "rtfx": True,
                      "cost_usd": False},
    threshold={"pause_f1": 0.02},
    secondary=["d_rt_cer", "d_mos", "overflow", "max_stretch", "mean_stretch", "speech_overlap",
               "onset_err_ms", "dur_err_ms", "rtfx"],
    model_judges=_model_judges,
    derived={"rt_cer": mean_of("rt_", "_cer"), "d_rt_cer": delta_vs_input("rt_", "_cer"),
             "d_mos": delta_vs_input("mos_")},
    gates={"d_rt_cer": ("<=", 0.01), "d_mos": (">=", -0.10), "overflow": ("<=", 0.01)},
    packs=["e1-fleurs-synth", "e1-arena-takes"],
    io="""item.inputs: {audio: take, target_s, src_pauses: [[s,e]], src_speech: [[s,e]]} (slot
timeline, seconds from slot start). item.refs: {text, pauses, speech, target_s}.
payload: {files: {audio}, time_map?: [[t_in, t_out], ...] monotone anchors, params?: {...}}.
Output keeps the take's native rate; it must not exceed target_s (+20 ms).""",
)
