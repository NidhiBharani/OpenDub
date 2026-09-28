"""D7 — non-verbal vocalisations (laughs, sighs, breaths, cries, screams …).

Ranked by non-verbal event F1: the output is tagged with CED (``mispeech/ced-base``, AudioSet;
worker ``d7_event_tagger``) and its events are matched per group against the reference events —
the inline tags of the script (``[laugh]`` …) or, when the pack carries it, the tagged source line
(``refs.nv_audio``; then onsets within ±200 ms also give ``nv_f1_timed``, meaningful for
time-aligned outputs such as the splice method). CED is a judge lineage none of the candidates is
trained against. Gate: the spoken words still survive (``rt_cer`` on the untagged script ≤ 0.2).
Packs: ``scene_voice`` (lines whose transcripts carry non-verbal tags). NV-Bench / NonverbalTTS
(NC) are not built.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..hardware import Requires
from ..judgelib import asr_roundtrip, naturalness
from ..judges import ModelJudge, Spec, output_file
from ..packs import Item
from ._dvoice import BASE_JUDGES, HIB, INTELLIGIBILITY, untagged_text

TAGS = re.compile(r"\[([^\]]+)\]")


def event_tagger(model: str = "mispeech/ced-base", version: int = 1) -> ModelJudge:
    """judgelib-style factory kept in this module (no core edit): CED event F1 vs the script's
    tags / the source vocalisation."""
    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        audio = output_file(out)
        if not audio:
            return None
        d: dict[str, Any] = {"audio": audio}
        nv = item.refs.get("nv_audio")
        if nv:
            d["ref_audio"] = nv
        else:
            d["expected"] = TAGS.findall(str(item.inputs.get("text") or ""))
        return d

    def to_rows(item: Item, payload: dict[str, Any], out: dict[str, Any]):
        weights = payload.get("weights") or {}
        return [(k, float(v), float(weights.get(k, 1.0)))
                for k, v in (payload.get("metrics") or {}).items() if v is not None]

    return ModelJudge(id=f"nv_events.ced@{version}", worker="d7_event_tagger", env="d7_tagger",
                      params={"model": model, "threshold": 0.3, "tol_s": 0.2},
                      requires=Requires.parse({"gpu": True, "vram_gb": 1.5}), lang_param=False,
                      inputs=inputs, to_rows=to_rows)


SPEC = Spec(
    id="D7",
    title="Non-verbal vocalizations",
    judges=dict(BASE_JUDGES),
    primary={"*": "nv_f1"},
    higher_is_better=HIB,
    threshold={"nv_f1": 0.02},
    secondary=["nv_f1_timed", "nv_precision", "nv_recall", "rt_cer", "mos_utmosv2", "degenerate",
               "rtfx", "cost_usd"],
    model_judges=lambda lang: [*asr_roundtrip(lang, text_of=untagged_text), event_tagger(),
                               naturalness("utmosv2")],
    gates={"rt_cer": ("<=", 0.2)},
    derived=dict(INTELLIGIBILITY),
    packs=["scene_voice"],
    io="""item.inputs: {text with inline tags e.g. "[laugh] そうなの?", ref_audio, ref_text,
events?: [tag, …], event_audio_<j>? (source vocalisation clips for the splice method)}
item.refs: {text, nv_audio? (source line with the vocalisation)}. payload: {files: {audio}}""",
)
