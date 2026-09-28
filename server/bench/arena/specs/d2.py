"""D2 — expressive / style-transfer synthesis.

Ranked by emotion consistency with the source line (``emo_sim_emotion2vec``: emotion2vec
utterance-embedding cosine between output and the source clip ``inputs.ref_audio``) behind the
D1 gates. The plan also subtracts a neutral-take baseline; that needs a paired neutral run and is
left to the audit (TODO once a neutral-control split exists). A/V/D distances from the judge are
secondary diagnostics. Packs: ``cv3_eval`` (emotion_zeroshot en), ``scene_voice`` (anime lines:
the headline expressive material), ``jvnv_emotion`` is not built (CC BY-SA, ja; future).
"""
from __future__ import annotations

from ..judgelib import emotion_consistency
from ..judges import Spec
from ._dvoice import BASE_JUDGES, HIB, INTELLIGIBILITY, SIMILARITY, untagged_text, voice_judges

SPEC = Spec(
    id="D2",
    title="Expressive and style-transfer synthesis",
    judges=dict(BASE_JUDGES),
    primary={"*": "emo_sim_emotion2vec"},
    higher_is_better=HIB,
    threshold={"emo_sim_emotion2vec": 0.01, "sim": 0.01},
    secondary=["emo_post_sim_emotion2vec", "emo_label_match", "emo_avd_dist", "sim", "rt_cer",
               "rt_cer_ratio", "mos_utmosv2", "degenerate", "rtfx", "cost_usd"],
    model_judges=lambda lang: [*voice_judges(lang, text_of=untagged_text),
                               emotion_consistency("emotion2vec")],
    gates={"rt_cer": ("<=", 0.2), "degenerate": ("<=", 0.02)},
    derived={**INTELLIGIBILITY, **SIMILARITY},
    packs=["cv3_eval", "scene_voice"],
    io="""item.inputs: {text, ref_audio (the emotional source line), ref_text, emotion? (label or
description)}. payload: {files: {audio}}""",
)
