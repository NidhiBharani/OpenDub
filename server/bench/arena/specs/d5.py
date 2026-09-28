"""D5 — licensed non-cloning (stock / designed) voices.

No reference voice: candidates speak with documented stock voices (or a voice designed from a
text brief), picked per language and by the line's speaker gender (``meta.gender``). The plan's
primary ("fits the character", native pairwise) is the user's audit; the automatic primary is
UTMOSv2 within a language (en, ja) and round-trip CER for Hindi (UTMOS is not validated for
Indic). Gate: intelligibility. Licence/provenance is a registry field (``license``, ``ship_ok``),
not a metric. Packs: ``fleurs_voice`` (capability D5 drops the reference voice).
"""
from __future__ import annotations

from ..judges import Spec
from ._dvoice import BASE_JUDGES, HIB, INTELLIGIBILITY, voice_judges

SPEC = Spec(
    id="D5",
    title="Licensed non-cloning voices",
    judges=dict(BASE_JUDGES),
    primary={"*": "mos_utmosv2", "hi": "rt_cer"},
    higher_is_better=HIB,
    threshold={"mos_utmosv2": 0.05, "rt_cer": 0.01},
    secondary=["rt_cer", "rt_cer_ratio", "mos_utmosv2", "degenerate", "rtfx", "cost_usd"],
    model_judges=lambda lang: voice_judges(lang, sim=False),
    gates={"rt_cer": ("<=", 0.2), "degenerate": ("<=", 0.02)},
    derived=dict(INTELLIGIBILITY),
    packs=["fleurs_voice"],
    io="""item.inputs: {text, voice_brief?}; item.meta: {gender?}; item.refs: {text, human_audio?}
payload: {files: {audio}, voice}""",
)
