"""Shared pieces of the phase-D (voice) specs: function judges, extra model judges and derived
metrics. Not a spec module itself (leading underscore: the spec loader skips it).

Metric vocabulary (every D spec uses the same names so leaderboards line up):

| metric            | from                                    | meaning                                   |
|-------------------|-----------------------------------------|-------------------------------------------|
| rt_<asr>_cer/wer  | judgelib.asr_roundtrip                  | round-trip error vs the script            |
| rt_cer            | derived: mean of rt_*_cer               | two-family (+CTC) intelligibility         |
| hf_<asr>_cer      | :func:`human_floor_asr`                 | same ASR on the HUMAN recording of the    |
|                   |                                         | same sentence (refs.human_audio)          |
| rt_cer_ratio      | derived: (rt_cer+ε)/(hf_cer+ε), ε=0.02  | the plan's WER_ratio vs the human floor   |
| rt_cer_excess     | derived: rt_cer − hf_cer                | same, additive (aggregates as corpus Δ)   |
| sim_<enc>         | judgelib.speaker_similarity             | cosine to the reference voice             |
| sim               | derived: mean of sim_* (two encoders)   | D1/D6 primary                             |
| simdiff_<enc>     | :func:`sim_to_human`                    | SIM(human other-speaker recording, ref):  |
|                   |                                         | the different-speaker floor per item      |
| sim_margin        | derived: sim − mean simdiff_*           | SIM above the different-speaker floor     |
| leak_<enc>, leak  | :func:`source_leak`                     | D6: SIM of the output to the SOURCE voice |
| mos_utmosv2       | judgelib.naturalness                    | within-language naturalness               |
| dur_s, speech_frac, degenerate | :func:`audio_stats`        | output sanity                             |
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..hardware import Requires
from ..judgelib import (
    ASR_JUDGES,
    ASR_PANEL,
    SIM_ENCODERS,
    asr_roundtrip,
    naturalness,
    speaker_similarity,
    spoken_text,
)
from ..judges import ModelJudge, Row, cost, error_rows, mean_of, output_file, speed
from ..packs import Item

SIM_PANEL = ("wavlm-sv", "eres2netv2")  # two disjoint encoder lineages (WavLM vs 3D-Speaker)
RATIO_EPS = 0.02
TAG = re.compile(r"\[[^\]]*\]|<\|[^|]*\|>")


# ------------------------------------------------------------------ audio helpers

def _mono(path: str):
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.asarray(x.mean(axis=1)), int(sr)


def speech_fraction(x, sr: int, frame_s: float = 0.02, floor_db: float = -40.0) -> float:
    """Share of 20 ms frames within ``floor_db`` of the clip's loud frames (energy VAD): silence
    padding or long dead air lowers it. Model-free on purpose (D3 gate)."""
    import numpy as np

    n = max(1, int(sr * frame_s))
    frames = len(x) // n
    if frames == 0:
        return 0.0
    e = (x[:frames * n].reshape(frames, n) ** 2).mean(axis=1) + 1e-12
    db = 10 * np.log10(e)
    ref = np.percentile(db, 95)
    return float((db > ref + floor_db).mean())


def audio_stats(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    """dur_s, speech_frac, rms_dbfs and a degenerate flag (missing/near-empty/silent audio, or an
    absurd length for the text: > 1 s per character or < 25 ms per character)."""
    import numpy as np

    path = output_file(out)
    text = TAG.sub("", str(item.inputs.get("text") or item.refs.get("text") or ""))
    chars = len(re.sub(r"\s+", "", text))
    if path is None:
        return [("degenerate", 1.0, 1.0)]
    x, sr = _mono(path)
    dur = len(x) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    rms_db = 20 * np.log10(rms + 1e-9)
    bad = dur < 0.3 or rms_db < -50 or (chars and (dur > chars * 1.0 + 3 or dur < chars * 0.025))
    return [("dur_s", dur, 1.0), ("speech_frac", speech_fraction(x, sr), 1.0),
            ("rms_dbfs", float(rms_db), 1.0), ("degenerate", float(bool(bad)), 1.0)]


def duration_fit(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    """D3: output duration vs ``inputs.target_s`` — relative error, ratio and in-window flags at
    ε = 5 % and 10 %."""
    tgt = item.inputs.get("target_s")
    path = output_file(out)
    if not tgt or path is None:
        return []
    import soundfile as sf

    info = sf.info(path)
    dur = info.frames / float(info.samplerate)
    err = abs(dur - float(tgt)) / float(tgt)
    return [("dur_err", err, 1.0), ("dur_ratio", dur / float(tgt), 1.0),
            ("in_window_5", float(err <= 0.05), 1.0), ("in_window_10", float(err <= 0.10), 1.0)]


def accept_at_n(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    """D3 accept-rate@N and wall seconds per accepted take. Best-of-N payloads carry ``takes``;
    a single-take candidate's accept@1 is its in-window (10 %) flag."""
    rows: list[Row] = []
    takes = out.get("takes")
    if takes:
        acc = sum(bool(t.get("accepted")) for t in takes)
        rows += [("accept_rate", acc / len(takes), float(len(takes))),
                 ("n_takes", float(len(takes)), 1.0)]
    else:
        fit = {m: v for m, v, _ in duration_fit(item, out, row)}
        if "in_window_10" not in fit:
            return []
        acc = int(fit["in_window_10"])
        rows.append(("accept_rate", float(acc), 1.0))
    secs = row["seconds"] if row is not None and "seconds" in row.keys() else None  # noqa: SIM118 - sqlite3.Row
    if secs and acc:
        rows.append(("sec_per_accept", float(secs) / acc, 1.0))
    return rows


# ------------------------------------------------------------------ chrF (D8)

def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, Popović 2015; spaces removed like sacreBLEU), 0–100."""
    h, r = re.sub(r"\s+", "", hyp), re.sub(r"\s+", "", ref)
    if not r:
        return 100.0 if not h else 0.0
    precs, recs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not rc:
            continue
        match = sum((hc & rc).values())
        precs.append(match / max(1, sum(hc.values())))
        recs.append(match / sum(rc.values()))
    if not recs:
        return 0.0
    p, rr = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + rr == 0:
        return 0.0
    return 100 * (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


def text_chrf(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    """D8: chrF of the system's own translation text (when it emits one) vs the reference."""
    ref, hyp = item.refs.get("text"), out.get("text")
    if not ref or not hyp:
        return []
    from ..judges import normalize_text

    return [("text_chrf", chrf(normalize_text(hyp), normalize_text(ref)), 1.0)]


# ------------------------------------------------------------------ extra model judges

def _rename_rows(payload: dict[str, Any], old: str, new: str) -> list[Row]:
    weights = payload.get("weights") or {}
    return [(k.replace(old, new, 1), float(v), float(weights.get(k, 1.0)))
            for k, v in (payload.get("metrics") or {}).items() if v is not None]


def human_floor_asr(lang: str, version: int = 1) -> list[ModelJudge]:
    """The ASR panel on ``refs.human_audio`` (a real recording of the same target sentence):
    per-item floor for rt_cer_ratio. Skipped for items without a human recording."""
    judges = []
    for name in ASR_PANEL.get(lang, ASR_PANEL["*"]):
        cfg = ASR_JUDGES[name]

        def inputs(item: Item, out: dict[str, Any], prefix: Path):
            human = item.refs.get("human_audio")
            return {"audio": human} if human and Path(human).exists() and \
                spoken_text(item, out) else None

        def to_rows(item: Item, payload: dict[str, Any], out: dict[str, Any], _n=name):
            return [r for r in error_rows(item.lang, spoken_text(item, out) or "",
                                          payload.get("text", ""), prefix=f"hf_{_n}_")
                    if r[0].endswith(("_cer", "_wer"))]

        judges.append(ModelJudge(
            id=f"asr_human_floor.{name}@{version}", worker=cfg["worker"], env=cfg["env"],
            params=cfg["params"], requires=Requires.parse(cfg["requires"]),
            languages=cfg["languages"], inputs=inputs, to_rows=to_rows))
    return [j for j in judges if j.supports(lang)]


def _sim_judge(encoder: str, jid: str, audio_of, ref_of, old: str, new: str,
               version: int = 1) -> ModelJudge:
    env, vram = SIM_ENCODERS[encoder]

    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        audio, ref = audio_of(item, out), ref_of(item)
        return {"audio": audio, "ref_audio": ref} if audio and ref else None

    return ModelJudge(id=f"{jid}.{encoder}@{version}", worker="judge_speaker_sim", env=env,
                      params={"encoder": encoder},
                      requires=Requires.parse({"gpu": True, "vram_gb": vram}), lang_param=False,
                      inputs=inputs,
                      to_rows=lambda item, payload, out: _rename_rows(payload, old, new))


def sim_to_human(encoder: str) -> ModelJudge:
    """SIM(refs.human_audio, reference voice): a different speaker saying the target sentence —
    the per-item different-speaker floor (SIM_diff) that sim_margin subtracts."""
    return _sim_judge(encoder, "speaker_sim_human_floor",
                      lambda item, out: item.refs.get("human_audio"),
                      lambda item: item.inputs.get("ref_audio"), "sim_", "simdiff_")


def source_leak(encoder: str) -> ModelJudge:
    """D6: SIM of the converted output to the SOURCE speaker (inputs.audio): source leakage."""
    return _sim_judge(encoder, "speaker_leak", lambda item, out: output_file(out),
                      lambda item: item.inputs.get("audio"), "sim_", "leak_")


def asr_chrf(lang: str, version: int = 1) -> list[ModelJudge]:
    """D8 ASR-chrF: transcribe the translated speech with the panel, chrF vs refs.text."""
    from ..judges import normalize_text

    judges = []
    for name in ASR_PANEL.get(lang, ASR_PANEL["*"]):
        cfg = ASR_JUDGES[name]

        def inputs(item: Item, out: dict[str, Any], prefix: Path):
            audio = output_file(out)
            return {"audio": audio} if audio and item.refs.get("text") else None

        def to_rows(item: Item, payload: dict[str, Any], out: dict[str, Any], _n=name):
            return [(f"asr_chrf_{_n}", chrf(normalize_text(payload.get("text", "")),
                                            normalize_text(item.refs["text"])), 1.0)]

        judges.append(ModelJudge(
            id=f"asr_chrf.{name}@{version}", worker=cfg["worker"], env=cfg["env"],
            params=cfg["params"], requires=Requires.parse(cfg["requires"]),
            languages=cfg["languages"], inputs=inputs, to_rows=to_rows))
    return [j for j in judges if j.supports(lang)]


def untagged_text(item: Item, out: dict[str, Any]) -> str | None:
    """The script without inline tags ([laugh], <|emotion:x|>): what ASR should hear (D2/D7)."""
    t = spoken_text(item, out)
    return " ".join(TAG.sub(" ", t).split()) if t else t


# ------------------------------------------------------------------ derived metrics

rt_cer = mean_of("rt_", "_cer")
hf_cer = mean_of("hf_", "_cer")
sim_mean = mean_of("sim_", "")
simdiff_mean = mean_of("simdiff_", "")
leak_mean = mean_of("leak_", "")


def rt_cer_ratio(values: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    rt, hf = rt_cer(values), hf_cer(values)
    if rt is None or hf is None:
        return None
    return ((rt[0] + RATIO_EPS) / (hf[0] + RATIO_EPS), rt[1])


def rt_cer_excess(values: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    rt, hf = rt_cer(values), hf_cer(values)
    return None if rt is None or hf is None else (rt[0] - hf[0], rt[1])


def sim_margin(values: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    s, d = sim_mean(values), simdiff_mean(values)
    return None if s is None or d is None else (s[0] - d[0], 1.0)


def sim_minus_leak(values: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    s, lk = sim_mean(values), leak_mean(values)
    return None if s is None or lk is None else (s[0] - lk[0], 1.0)


INTELLIGIBILITY = {"rt_cer": rt_cer, "hf_cer": hf_cer, "rt_cer_ratio": rt_cer_ratio,
                   "rt_cer_excess": rt_cer_excess}
SIMILARITY = {"sim": sim_mean, "simdiff": simdiff_mean, "sim_margin": sim_margin}

BASE_JUDGES = {"audio_stats@1": audio_stats, "speed@1": speed, "cost@1": cost}

# Direction of every metric the D specs can emit.
HIB = {"rt_cer": False, "hf_cer": False, "rt_cer_ratio": False, "rt_cer_excess": False,
       "sim": True, "simdiff": False, "sim_margin": True, "leak": False, "sim_minus_leak": True,
       "mos_utmosv2": True, "dur_s": True, "speech_frac": True, "rms_dbfs": True,
       "degenerate": False, "rtfx": True, "cost_usd": False, "dur_err": False,
       "dur_ratio": True, "in_window_5": True, "in_window_10": True, "accept_rate": True,
       "n_takes": True, "sec_per_accept": False, "emo_sim_emotion2vec": True,
       "emo_post_sim_emotion2vec": True, "emo_label_match": True, "emo_avd_dist": False,
       "emo_arousal_d": False, "emo_valence_d": False, "emo_dominance_d": False,
       "nv_f1": True, "nv_f1_timed": True, "nv_precision": True, "nv_recall": True,
       "nv_ref_events": True, "nv_out_events": True, "asr_chrf": True, "text_chrf": True}


def voice_judges(lang: str, *, sim: bool = True, human_floor: bool = True, mos: bool = True,
                 text_of=spoken_text) -> list[ModelJudge]:
    """The standard D1–D5 panel: ASR round-trip (+ human floor), two SIM encoders (+ human
    different-speaker floor), UTMOSv2."""
    js = list(asr_roundtrip(lang, text_of=text_of))
    if human_floor:
        js += human_floor_asr(lang)
    if sim:
        js += [speaker_similarity(e) for e in SIM_PANEL] + [sim_to_human(e) for e in SIM_PANEL]
    if mos:
        js.append(naturalness("utmosv2"))
    return js
