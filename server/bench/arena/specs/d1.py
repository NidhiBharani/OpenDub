"""D1 — zero-shot cross-lingual voice cloning.

Ranked by cross-lingual speaker similarity (``sim`` = mean of two disjoint encoders, WavLM-SV and
3D-Speaker ERes2NetV2, to the source-language reference voice) after an intelligibility gate
(``rt_cer`` ≤ 0.2 on the ASR panel; the plan's goal is ``rt_cer_ratio`` ≤ 1.5 against the same
ASR on the human recording of the sentence, computed wherever the pack has
``refs.human_audio``). UTMOSv2 is secondary and only comparable within a language.

Packs: ``fleurs_voice`` (ja voice → en/hi text, en → hi, same-language control split),
``seedtts_eval`` (en), ``cv3_eval`` (cross-lingual → en/ja, zero-shot ja/en), ``minimax_ml``
(24-language set incl. en/hi/ja), ``indicvoices_r`` (hi) and ``scene_voice`` (SPY×FAMILY lines).
Judge lineage: CosyVoice/Step-Audio-EditX embed CAM++ speaker vectors, so no CAM++ encoder is on
the SIM panel; the best-of-N verifier (large-v3-turbo) is not the reporting ASR.
"""
from __future__ import annotations

from ..judges import Spec
from ._dvoice import BASE_JUDGES, HIB, INTELLIGIBILITY, SIMILARITY, voice_judges

SPEC = Spec(
    id="D1",
    title="Zero-shot cross-lingual voice cloning",
    judges=dict(BASE_JUDGES),
    primary={"*": "sim"},
    higher_is_better=HIB,
    threshold={"sim": 0.01, "rt_cer": 0.01, "rt_cer_ratio": 0.1, "mos_utmosv2": 0.05},
    secondary=["rt_cer", "rt_cer_ratio", "rt_cer_excess", "sim_margin", "mos_utmosv2",
               "degenerate", "speech_frac", "rtfx", "cost_usd"],
    model_judges=lambda lang: voice_judges(lang),
    gates={"rt_cer": ("<=", 0.2), "degenerate": ("<=", 0.02)},
    derived={**INTELLIGIBILITY, **SIMILARITY},
    packs=["fleurs_voice", "seedtts_eval", "cv3_eval", "minimax_ml", "indicvoices_r",
           "scene_voice"],
    io="""item.inputs: {text (target language), ref_audio (source-language voice), ref_text}
item.refs: {text, human_audio? (human recording of the target sentence: ASR floor)}
item.meta: {src_lang, pair, duration_s, gender?}; split test = cross-lingual, control = same
language. payload: {files: {audio}, sample_rate, duration_s, seed}""",
)
