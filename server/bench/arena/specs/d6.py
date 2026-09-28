"""D6 — voice conversion.

Ranked by speaker similarity of the converted speech to the TARGET reference (``sim``, two
encoders) behind a content gate (``rt_cer`` of the output vs the source transcript ≤ 0.2).
Source leakage (``leak``: similarity of the output to the SOURCE speaker) and ``sim_minus_leak``
are secondary. Packs: ``fleurs_voice`` (capability D6: FLEURS en/hi/ja utterance as source, a
Japanese speaker as target) and ``scene_voice`` (official dub actor line → original actor
voice, the appendix's IH pairs).
"""
from __future__ import annotations

from ..judgelib import asr_roundtrip, naturalness, speaker_similarity
from ..judges import Spec
from ._dvoice import (
    BASE_JUDGES,
    HIB,
    INTELLIGIBILITY,
    SIM_PANEL,
    leak_mean,
    sim_mean,
    sim_minus_leak,
    source_leak,
)

SPEC = Spec(
    id="D6",
    title="Voice conversion",
    judges=dict(BASE_JUDGES),
    primary={"*": "sim"},
    higher_is_better=HIB,
    threshold={"sim": 0.01, "leak": 0.01},
    secondary=["leak", "sim_minus_leak", "rt_cer", "mos_utmosv2", "degenerate", "rtfx",
               "cost_usd"],
    model_judges=lambda lang: [*asr_roundtrip(lang), *[speaker_similarity(e) for e in SIM_PANEL],
                               *[source_leak(e) for e in SIM_PANEL], naturalness("utmosv2")],
    gates={"rt_cer": ("<=", 0.2), "degenerate": ("<=", 0.02)},
    derived={**INTELLIGIBILITY, "sim": sim_mean, "leak": leak_mean,
             "sim_minus_leak": sim_minus_leak},
    packs=["fleurs_voice", "scene_voice"],
    io="""item.inputs: {audio (source speech), ref_audio (target voice), text? (source transcript)}
item.refs: {text (source transcript)}. payload: {files: {audio}}""",
)
