"""C6 — subtitle condensation: make the translated cues readable in their time (CPS), line
length (CPL) and line count, without losing meaning.

- **Gates** (IWSLT 2026 subtitling track limits, per target language, :data:`LIMITS`): share of
  cues within CPS, within CPL, within the line count.
- **Primary — SubER** (Wilken et al. 2022; the IWSLT subtitling primary metric) against the
  reference subtitles, from the ``c6_suber_judge`` worker (``subtitle-edit-rate`` in its own
  env). Lower is better; practical threshold 1 point.
- Secondary: chrF of the cue text against the reference text, compression ratio (output chars
  / input chars), CPS/CPL compliance.

Limits: IWSLT 2026 (iwslt.org/2026/subtitling) uses 21 CPS / 42 CPL / 2 lines for ar, de, es;
9/16/2 for zh; 4/13/2 for ja with half-width characters counting 0.5. en and hi are not IWSLT
targets; they use the same 21/42/2 as the European languages (Netflix-style). The ja 4 CPS
profile is known to be broken (IWSLT's own ja references are only ~45% compliant at 4 CPS and
~96% at 6), so ja gates at 6 CPS and records the nominal 4 in ``nominal_cps``.
"""
from __future__ import annotations

import unicodedata
from typing import Any

from ..hardware import Requires
from ..judges import ModelJudge, Row, Spec, cost
from ..packs import Item
from .c2 import chrf, tgt_lang

LIMITS = {"en": {"cps": 21, "cpl": 42, "lines": 2},
          "hi": {"cps": 21, "cpl": 42, "lines": 2},
          "ja": {"cps": 6, "cpl": 13, "lines": 2, "nominal_cps": 4, "halfwidth": 0.5},
          "zh": {"cps": 9, "cpl": 16, "lines": 2},
          "de": {"cps": 21, "cpl": 42, "lines": 2}, "es": {"cps": 21, "cpl": 42, "lines": 2}}


def char_len(line: str, lang: str) -> float:
    """Characters as IWSLT counts them: in ja, half-width characters count 0.5."""
    half = LIMITS.get(lang, {}).get("halfwidth")
    if not half:
        return float(len(line))
    return sum(half if unicodedata.east_asian_width(c) in ("H", "Na") else 1.0 for c in line)


def cue_checks(cues: list[dict], lang: str) -> dict[str, float]:
    lim = LIMITS.get(lang, LIMITS["en"])
    n = len(cues)
    if not n:
        return {}
    cps = cpl = lines = 0
    for c in cues:
        ls = [ln for ln in str(c.get("text", "")).split("\n")]
        dur = max(1e-3, float(c["end"]) - float(c["start"]))
        chars = sum(char_len(ln, lang) for ln in ls)
        cps += chars / dur <= lim["cps"]
        cpl += max((char_len(ln, lang) for ln in ls), default=0) <= lim["cpl"]
        lines += len(ls) <= lim["lines"]
    return {"cps_ok": cps / n, "cpl_ok": cpl / n, "lines_ok": lines / n}


def subtitle_checks(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    lang = tgt_lang(item)
    cues = out.get("cues") or []
    n = float(max(1, len(cues)))
    rows: list[Row] = [(k, v, n) for k, v in cue_checks(cues, lang).items()]
    src_chars = sum(char_len(str(c.get("text", "")).replace("\n", ""), lang)
                    for c in item.inputs.get("cues") or [])
    out_chars = sum(char_len(str(c.get("text", "")).replace("\n", ""), lang) for c in cues)
    if src_chars:
        rows.append(("compression", out_chars / src_chars, 1.0))
    rows.append(("cue_count_kept", float(len(cues) == len(item.inputs.get("cues") or [])), 1.0))
    ref = item.refs.get("cues")
    if ref:
        rows.append(("chrf", chrf(" ".join(str(c.get("text", "")) for c in cues),
                                  " ".join(str(c["text"]) for c in ref)), 1.0))
    return rows


def suber_judge(version: int = 1) -> ModelJudge:
    def inputs(item: Item, out: dict[str, Any], prefix) -> dict[str, Any] | None:
        ref, hyp = item.refs.get("cues"), out.get("cues")
        return {"hyp_cues": hyp, "ref_cues": ref} if ref and hyp else None

    return ModelJudge(
        id=f"suber@{version}", worker="c6_suber_judge", env="c6_suber", lang_param=False,
        requires=Requires(), inputs=inputs,
        to_rows=lambda item, payload, out: [(k, float(v), 1.0) for k, v in
                                            (payload.get("metrics") or {}).items()])


SPEC = Spec(
    id="C6",
    title="Subtitle condensation",
    judges={"subtitle_checks@1": subtitle_checks, "cost@1": cost},
    primary={"*": "suber"},
    higher_is_better={"suber": False, "suber_cased": False, "cps_ok": True, "cpl_ok": True,
                      "lines_ok": True, "compression": False, "cue_count_kept": True,
                      "chrf": True, "cost_usd": False},
    threshold={"suber": 1.0, "chrf": 1.0},
    secondary=["cps_ok", "cpl_ok", "lines_ok", "chrf", "compression", "suber_cased",
               "cost_usd"],
    model_judges=lambda lang: [suber_judge()],
    gates={"cps_ok": (">=", 0.85), "cpl_ok": (">=", 0.95), "lines_ok": (">=", 0.99)},
    packs=["iwslt_subtitling", "c6_srt_pairs"],
    io="""pack lang = direction (en-ja, ja-en, ja-hi, en-hi …).
item.inputs: {cues: [{start, end, text (uncondensed target), source?}], limits: {cps, cpl,
lines}, src_lang, tgt_lang}; item.refs: {cues: [{start, end, text}] (reference subtitles)};
group = document/episode. payload: {cues: [{start, end, text}], text}.
LLM workers run with params.task = condense.""",
)
