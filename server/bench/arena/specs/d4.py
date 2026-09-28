"""D4 — regional and low-resource language voices (Hindi in the current scope).

The plan ranks by native-listener CMOS; with no raters (decision 2026-09-27) the automatic
primary is intelligibility: ``rt_cer`` on the ASR panel (Whisper + Qwen3-ASR + MMS CTC), with
``rt_cer_ratio`` against the human recording as the calibrated view and the user's audit
overruling. UTMOSv2 is NOT validated for Indic languages: reported, not ranked. SIM is secondary
(a D4 item may carry a reference voice). Packs: ``indicvoices_r`` (hi, CC BY 4.0, gated),
``fleurs_voice`` (hi), ``minimax_ml`` (hi).
"""
from __future__ import annotations

from ..judges import Spec
from ._dvoice import BASE_JUDGES, HIB, INTELLIGIBILITY, SIMILARITY, voice_judges

SPEC = Spec(
    id="D4",
    title="Regional and low-resource language voices",
    judges=dict(BASE_JUDGES),
    primary={"*": "rt_cer"},
    higher_is_better=HIB,
    threshold={"rt_cer": 0.01, "rt_cer_ratio": 0.1},
    secondary=["rt_cer_ratio", "rt_cer_excess", "sim", "sim_margin", "mos_utmosv2", "degenerate",
               "rtfx", "cost_usd"],
    model_judges=lambda lang: voice_judges(lang),
    gates={"rt_cer": ("<=", 0.2), "degenerate": ("<=", 0.02)},
    derived={**INTELLIGIBILITY, **SIMILARITY},
    packs=["indicvoices_r", "fleurs_voice", "minimax_ml"],
    io="""item.inputs: {text, ref_audio?, ref_text?}; item.refs: {text, human_audio?}
payload: {files: {audio}}""",
)
