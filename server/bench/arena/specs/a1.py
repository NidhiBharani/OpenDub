"""A1 — dialogue / music-and-effects separation.

Ranked by **bed ghost-dialogue**: the dialogue that leaks into the estimated M&E bed, measured as
the share of reference dialogue words an ASR panel can still recover from ``files.background``
(lower is better; docs/plans/model-ranking.md appendix A1). SI-SDR(i) of dialogue and bed on
synthetic mixtures with reference stems (``item.refs``) is reported alongside, and a bed that is
worse than the untouched mixture (``sisdri_background < 0``, e.g. a silent bed that trivially has no
ghost dialogue) fails the gate.

Aggregation note: the arena takes weighted means, so SI-SDR is a per-item mean (the plan's median
is visible in the audit viewer's per-item rows).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..hardware import Requires
from ..judgelib import ASR_JUDGES, ASR_PANEL, asr_roundtrip
from ..judges import ModelJudge, Row, Spec, cost, mean_of, normalize_for, output_file, speed
from ..packs import Item

EPS = 1e-9


# ------------------------------------------------------------------ audio helpers (numpy)

def load_mono(path: str | Path, sr: int | None = None):
    """Mono float64 signal and its rate; resampled to ``sr`` when given (polyphase)."""
    import numpy as np
    import soundfile as sf

    x, rate = sf.read(str(path), always_2d=True, dtype="float64")
    x = x.mean(axis=1)
    if sr and rate != sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), int(rate))
        x = resample_poly(x, sr // g, rate // g)
        rate = sr
    return np.asarray(x, dtype=np.float64), rate


def si_sdr(ref, est) -> float:
    """Scale-invariant SDR in dB (Le Roux et al. 2019), both signals zero-meaned."""
    import numpy as np

    n = min(len(ref), len(est))
    if n == 0:
        return float("nan")
    r = ref[:n] - ref[:n].mean()
    e = est[:n] - est[:n].mean()
    if not np.any(r) or not np.any(e):
        return float("nan")
    alpha = np.dot(e, r) / (np.dot(r, r) + EPS)
    target = alpha * r
    noise = e - target
    return float(10 * np.log10((np.dot(target, target) + EPS) / (np.dot(noise, noise) + EPS)))


def _stem_rows(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    mix_path = item.inputs.get("audio")
    if not mix_path:
        return []
    rows: list[Row] = []
    mix = rate = None
    for stem in ("dialogue", "background", "music", "effects"):
        ref_path, est_path = item.refs.get(stem), output_file(out, stem)
        if not ref_path or not Path(str(ref_path)).exists():
            continue
        ref, rate = load_mono(ref_path)
        if mix is None:
            mix, _ = load_mono(mix_path, rate)
        base = si_sdr(ref, mix)
        if est_path is None:
            if stem in ("dialogue", "background"):  # a required stem is missing: worst case
                rows.append((f"missing_{stem}", 1.0, 1.0))
            continue
        est, _ = load_mono(est_path, rate)
        val = si_sdr(ref, est)
        if math.isnan(val):  # silent estimate
            val = -50.0
        rows.append((f"sisdr_{stem}", val, 1.0))
        if not math.isnan(base):
            rows.append((f"sisdri_{stem}", val - base, 1.0))
    # Reconstruction: does dialogue + bed add back up to the mix? (bed = mix − dialogue does.)
    d, b = output_file(out, "dialogue"), output_file(out, "background")
    if d and b and rate:
        dd, _ = load_mono(d, rate)
        bb, _ = load_mono(b, rate)
        n = min(len(dd), len(bb), len(mix))
        rows.append(("recon_sdr", min(60.0, si_sdr(mix[:n], dd[:n] + bb[:n])), 1.0))
    return rows


# ------------------------------------------------------------------ ghost dialogue in the bed

def word_recall(lang: str, reference: str, hypothesis: str) -> tuple[float, float] | None:
    """Share of reference words (characters for unspaced languages) present in the hypothesis,
    as a multiset: (recall, reference length). Order-free on purpose: a ghost line is a leak
    whether or not the ASR gets the word order right."""
    from collections import Counter

    from ..judges import UNSPACED

    ref, hyp = normalize_for(lang, reference), normalize_for(lang, hypothesis)
    if lang in UNSPACED:
        rt, ht = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    else:
        rt, ht = ref.split(), hyp.split()
    if not rt:
        return None
    hits = sum((Counter(rt) & Counter(ht)).values())
    return hits / len(rt), float(len(rt))


def dialogue_text(item: Item, out: dict[str, Any]) -> str | None:
    return item.refs.get("text")


def ghost_dialogue(lang: str, *, panel: list[str] | None = None,
                   version: int = 1) -> list[ModelJudge]:
    """ASR panel over ``files.background``; ``ghost_<asr>_recall`` = reference dialogue words
    recovered from the bed (↓). Worker/env/params are the judgelib ASR judges."""
    judges = []
    for name in panel or ASR_PANEL.get(lang, ASR_PANEL["*"]):
        cfg = ASR_JUDGES[name]

        def inputs(item: Item, out: dict[str, Any], prefix: Path):
            bed = output_file(out, "background")
            return {"audio": bed} if bed and item.refs.get("text") else None

        def to_rows(item: Item, payload: dict[str, Any], out: dict[str, Any], _n=name):
            res = word_recall(item.lang, item.refs.get("text", ""), payload.get("text", ""))
            return [(f"ghost_{_n}_recall", res[0], res[1])] if res else []

        judges.append(ModelJudge(
            id=f"ghost_dialogue.{name}@{version}", worker=cfg["worker"], env=cfg["env"],
            params=cfg["params"], requires=Requires.parse(cfg["requires"]),
            languages=cfg["languages"], inputs=inputs, to_rows=to_rows))
    return [j for j in judges if j.supports(lang)]


def _model_judges(lang: str) -> list[ModelJudge]:
    # Ghost dialogue in the bed (primary) + intelligibility of the dialogue stem (secondary).
    return [*ghost_dialogue(lang),
            *asr_roundtrip(lang, text_of=dialogue_text, audio_key="dialogue")]


SPEC = Spec(
    id="A1",
    title="Dialogue / music-and-effects separation",
    judges={"stems@1": _stem_rows, "speed@1": speed, "cost@1": cost},
    model_judges=_model_judges,
    derived={"ghost_recall": mean_of("ghost_", "_recall"),
             "rt_cer": mean_of("rt_", "_cer")},
    primary={"*": "ghost_recall"},
    higher_is_better={"ghost_recall": False, "sisdr_dialogue": True, "sisdri_dialogue": True,
                      "sisdr_background": True, "sisdri_background": True, "sisdr_music": True,
                      "sisdri_music": True, "sisdr_effects": True, "sisdri_effects": True,
                      "recon_sdr": True, "missing_dialogue": False, "missing_background": False,
                      "rt_cer": False, "rtfx": True, "cost_usd": False},
    threshold={"ghost_recall": 0.02, "sisdri_dialogue": 0.5, "sisdri_background": 0.5},
    gates={"sisdri_background": (">=", 0.0)},
    secondary=["sisdri_dialogue", "sisdri_background", "sisdr_dialogue", "sisdr_background",
               "sisdri_music", "sisdri_effects", "recon_sdr", "rt_cer", "rtfx", "cost_usd"],
    packs=["a1-dnr-remix", "a1-dnr-v3"],
    io="""item.inputs: {audio: the mixture}; item.refs: {dialogue, background (music+effects),
music?, effects? (reference stem paths), text (dialogue transcript)}.
payload: {files: {dialogue, background, music?, effects?}} at the model's native rate.""",
)
