"""C2 — translation and dubbing adaptation. Also home of the C-phase text helpers the other C
specs import (direction strings, the speaking-rate model, chrF).

Pack language is a *direction*: ``data/eval/C2/ja-en/``, ``ja-hi``, ``en-hi``, ``hi-en``,
``en-ja`` …; ``item.meta`` carries ``src_lang``/``tgt_lang``. Everything that normalises or
counts text uses the target language, never ``item.lang``.

Ranking (docs/plans/model-ranking.md appendix, row C2):

- **Gate — duration compliance.** The translation's predicted spoken duration (per-language
  speaking-rate model: syllables for en/hi, morae for ja; characters are not comparable across
  scripts) must fall within [0.9, 1.1]× the source slot. The slot is the measured line duration
  when the pack has one (``inputs.slot_s``, e.g. SCENE), else the predicted duration of the
  source text. A candidate is ranked only if ≥ :data:`COMPLIANCE_GATE` of its lines comply —
  a starting value until the rate model is calibrated (``ref_dur_compliance`` shows how often
  the human reference itself complies under the same model).
- **Primary — MetricX-24-Hybrid-XL** via ``judgelib.mt_qe`` (reference-based where the item has
  a reference, QE otherwise; lower is better; practical threshold 1 MetricX point).
- **Secondary** — xCOMET-XL and CometKiwi-XL (eval-only licences), chrF, a GEMBA-ESA LLM judge
  from a model family that is not a candidate (DeepSeek; see :data:`GEMBA_JUDGE`), MetricX on
  compliant lines only ("quality at fixed compliance"), $/line.

Caveat recorded for the user: TranslateGemma was trained with MetricX-QE in its RL loop, so its
MetricX rank is optimistic (§4.7); read its xCOMET/GEMBA columns.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import unicodedata
from collections import Counter
from functools import cache
from typing import Any

from .. import judgelib, paths
from ..hardware import Requires
from ..judges import ModelJudge, Row, Spec, cost
from ..packs import Item

# ------------------------------------------------------------------ shared C-phase helpers


@cache
def worker_module(name: str):
    """Import a stdlib-only module from ``bench/arena/workers`` (e.g. the speaking-rate model)
    so specs and workers share one implementation."""
    wdir = str(paths.WORKERS_DIR)
    added = wdir not in sys.path
    if added:
        sys.path.insert(0, wdir)
    try:
        spec = importlib.util.spec_from_file_location(f"_arena_worker_{name}",
                                                      paths.WORKERS_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        if added:
            sys.path.remove(wdir)


rate = worker_module("c_speech_rate")


def _env_rates() -> dict[str, float]:
    """``OPENDUB_ARENA_RATES="en=5.9,ja=7.4,hi=5.4"`` overrides the default units/second (bump
    the judge versions below when you change it, so outputs are re-scored)."""
    out = {}
    for part in os.environ.get("OPENDUB_ARENA_RATES", "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


RATES = _env_rates()


def langs_of(item: Item) -> tuple[str, str]:
    """(src, tgt) of a text item: meta first, else the direction string ``"ja-en"``."""
    if "-" in item.lang:
        src, tgt = item.lang.split("-", 1)
    else:
        src = tgt = rate.base_lang(item.lang)
    return (rate.base_lang(item.meta.get("src_lang") or src),
            rate.base_lang(item.meta.get("tgt_lang") or tgt))


def tgt_lang(item: Item) -> str:
    return langs_of(item)[1]


def src_lang(item: Item) -> str:
    return langs_of(item)[0]


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text or "").split())


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Sentence chrF (Popović 2015; sacreBLEU defaults: char 6-grams, β = 2, whitespace
    removed), 0–100. Own implementation so the server venv needs no sacrebleu."""
    h, r = normalize(hyp).replace(" ", ""), normalize(ref).replace(" ", "")
    if not r:
        return 0.0
    precs, recs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not rc:
            continue
        match = sum((hc & rc).values())
        precs.append(match / sum(hc.values()) if hc else 0.0)
        recs.append(match / sum(rc.values()))
    if not recs:
        return 0.0
    p, rr = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + rr == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * rr / (b2 * p + rr)


def slot_seconds(item: Item) -> float:
    slot = item.inputs.get("slot_s")
    if slot:
        return float(slot)
    return rate.predicted_seconds(str(item.inputs.get("text", "")), src_lang(item), RATES)


def duration_rows(text: str, item: Item, prefix: str = "") -> list[Row]:
    slot = slot_seconds(item)
    r = rate.duration_ratio(text, tgt_lang(item), slot, RATES)
    if r is None or r <= 0:
        return []
    return [(f"{prefix}dur_ratio", r, 1.0),
            (f"{prefix}dur_compliance", float(LOW <= r <= HIGH), 1.0),
            (f"{prefix}dur_abs_log_err", abs(math.log(r)), 1.0)]


# ------------------------------------------------------------------ C2 judges

LOW, HIGH = 0.9, 1.1
COMPLIANCE_GATE = 0.5


def duration(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    rows = duration_rows(str(out.get("text", "")), item)
    ref = item.refs.get("text")
    if ref:
        rows += [(m.replace("dur_", "ref_dur_"), v, w) for m, v, w in duration_rows(ref, item)
                 if m == "dur_compliance"]
    return rows


def text_checks(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    hyp = normalize(str(out.get("text", "")))
    src = normalize(str(item.inputs.get("text", "")))
    rows: list[Row] = [("empty_output", float(not hyp), 1.0),
                       ("untranslated", float(bool(hyp) and hyp == src), 1.0)]
    ref = item.refs.get("text")
    if ref:
        rows.append(("chrf", chrf(hyp, ref), 1.0))
    return rows


# GEMBA-ESA judge: the plan requires a family different from every candidate. None of the C2
# candidates is a DeepSeek model (tests enforce it), so DeepSeek's OpenAI-compatible API judges.
GEMBA_JUDGE = {"backend": "external", "base_url": "https://api.deepseek.com",
               "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY", "task": "gemba",
               "use_reference": False, "temperature": 0.0}
GEMBA_FAMILY = "deepseek"


def gemba_judge(version: int = 1) -> ModelJudge:
    def inputs(item: Item, out: dict[str, Any], prefix) -> dict[str, Any] | None:
        src, hyp = item.inputs.get("text"), out.get("text")
        if not src or not hyp:
            return None
        s, t = langs_of(item)
        return {"source": src, "hypothesis": hyp, "src_lang": s, "tgt_lang": t}

    def to_rows(item: Item, payload: dict[str, Any], out: dict[str, Any]) -> list[Row]:
        m = payload.get("metrics") or {}
        return [(k, float(v), 1.0) for k, v in m.items() if v is not None]

    return ModelJudge(id=f"gemba_esa.{GEMBA_FAMILY}@{version}", worker="c2_openai_compat",
                      env="server", params=GEMBA_JUDGE, lang_param=False,
                      requires=Requires.parse({"api_keys": ["DEEPSEEK_API_KEY"]}),
                      inputs=inputs, to_rows=to_rows)


METRICX = "metricx-24-hybrid-xl"


def _metricx(values: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """Reference-based MetricX where the item had a reference, QE otherwise."""
    return values.get(f"hyb_{METRICX}") or values.get(f"qe_{METRICX}")


def _metricx_compliant(values: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    mx, ok = _metricx(values), values.get("dur_compliance")
    return mx if mx and ok and ok[0] >= 1.0 else None


def model_judges(lang: str) -> list[ModelJudge]:
    return [judgelib.mt_qe(METRICX, use_reference=True),
            judgelib.mt_qe("xcomet-xl", use_reference=True),
            judgelib.mt_qe("cometkiwi-da-xl"),
            gemba_judge()]


SPEC = Spec(
    id="C2",
    title="Translation and dubbing adaptation",
    judges={"duration@1": duration, "text_checks@1": text_checks, "cost@1": cost},
    primary={"*": "metricx"},
    higher_is_better={
        "metricx": False, "metricx_compliant": False,
        f"hyb_{METRICX}": False, f"qe_{METRICX}": False,
        "hyb_xcomet-xl": True, "qe_xcomet-xl": True, "qe_cometkiwi-da-xl": True,
        "gemba_esa": True, "gemba_major": False, "gemba_minor": False,
        "dur_ratio": False, "dur_compliance": True, "dur_abs_log_err": False,
        "ref_dur_compliance": True, "chrf": True, "empty_output": False,
        "untranslated": False, "cost_usd": False},
    threshold={"metricx": 1.0, "metricx_compliant": 1.0, "chrf": 1.0, "gemba_esa": 2.0},
    secondary=["dur_compliance", "dur_abs_log_err", "metricx_compliant", "hyb_xcomet-xl",
               "qe_xcomet-xl", "qe_cometkiwi-da-xl", "gemba_esa", "chrf", "empty_output",
               "cost_usd", "ref_dur_compliance"],
    model_judges=model_judges,
    gates={"dur_compliance": (">=", COMPLIANCE_GATE), "empty_output": ("<=", 0.01)},
    derived={"metricx": _metricx, "metricx_compliant": _metricx_compliant},
    packs=["wmt24pp", "in22_conv", "flores_plus", "bsd", "scene_c2"],
    io="""pack lang = direction (ja-en, ja-hi, en-hi, hi-en, en-ja).
item.inputs: {text (source line), src_lang, tgt_lang, slot_s? (measured seconds),
context_before?: [..], context_after?: [..], speaker?, emotion?};
item.refs: {text? (reference translation)}; item.meta: {src_lang, tgt_lang, source, domain?}.
payload: {text, candidates? (N-best), src_lang, tgt_lang, usage?}.
Worker params.task = translate; params.prompt selects the prompt style (c_common).""",
)
