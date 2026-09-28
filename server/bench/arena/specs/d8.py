"""D8 — direct speech-to-speech translation (evaluate-only comparator).

Ranked by ASR-chrF: the translated speech is transcribed by the ASR panel and scored with chrF
against the reference translation (``refs.text``), so systems that emit no text are comparable.
The system's own text output (when present) gets ``text_chrf``; voice preservation is SIM to
the source clip (``sim``) and UTMOSv2 is reported within a language. No gate: D8 never becomes a
pipeline stage; it is compared against the cascade on the same clips (the ladder/cascade view).
Packs: ``fleurs_voice`` (capability D8: a ja/en FLEURS recording → the en/hi reference text of the
same FLoRes sentence) and ``scene_voice`` (anime lines → official dub transcript).
"""
from __future__ import annotations

from ..judgelib import naturalness, speaker_similarity
from ..judges import Spec, mean_of
from ._dvoice import BASE_JUDGES, HIB, SIM_PANEL, asr_chrf, sim_mean, text_chrf

SPEC = Spec(
    id="D8",
    title="Direct speech-to-speech translation (evaluate only)",
    judges={**BASE_JUDGES, "text_chrf@1": text_chrf},
    primary={"*": "asr_chrf"},
    higher_is_better=HIB,
    threshold={"asr_chrf": 1.0},
    secondary=["text_chrf", "sim", "mos_utmosv2", "degenerate", "rtfx", "cost_usd"],
    model_judges=lambda lang: [*asr_chrf(lang),
                               *[speaker_similarity(e, ref_key="audio") for e in SIM_PANEL],
                               naturalness("utmosv2")],
    derived={"asr_chrf": mean_of("asr_chrf_", ""), "sim": sim_mean},
    packs=["fleurs_voice", "scene_voice"],
    io="""item.inputs: {audio (source-language speech)}; item.meta: {src_lang}
item.refs: {text (reference translation)}. payload: {text?, files: {audio}}""",
)
