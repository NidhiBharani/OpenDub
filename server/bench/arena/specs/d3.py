"""D3 — duration-controlled synthesis.

Ranked by the in-window rate at ε = 10 % (``in_window_10``: |dur − target_s| / target_s ≤ 0.10;
ε = 5 % secondary) behind two gates: intelligibility (``rt_cer`` ≤ 0.2 — truncation or rushing
shows up here) and speech fraction (energy VAD ≥ 0.6 — no silence padding to hit the window).
``accept_rate`` is accept-rate@N from best-of-N candidates (``takes`` in the payload; a single-take
candidate reports accept@1 = its in-window flag) and ``sec_per_accept`` the wall time per
accepted take, so the cost of rejection sampling is visible.
Packs: ``fleurs_voice`` with ``target_s`` = the human recording's duration (optionally scaled
0.8–1.2 via the builder's ``compress``), ``scene_voice`` with ``target_s`` = the source line.
"""
from __future__ import annotations

from ..judges import Spec
from ._dvoice import (
    BASE_JUDGES,
    HIB,
    INTELLIGIBILITY,
    SIMILARITY,
    accept_at_n,
    duration_fit,
    voice_judges,
)

SPEC = Spec(
    id="D3",
    title="Duration-controlled synthesis",
    judges={**BASE_JUDGES, "duration_fit@1": duration_fit, "accept_at_n@1": accept_at_n},
    primary={"*": "in_window_10"},
    higher_is_better=HIB,
    threshold={"in_window_10": 0.02, "in_window_5": 0.02, "dur_err": 0.005},
    secondary=["in_window_5", "dur_err", "accept_rate", "sec_per_accept", "rt_cer",
               "rt_cer_ratio", "sim", "mos_utmosv2", "speech_frac", "rtfx", "cost_usd"],
    model_judges=lambda lang: voice_judges(lang),
    gates={"rt_cer": ("<=", 0.2), "speech_frac": (">=", 0.6), "degenerate": ("<=", 0.02)},
    derived={**INTELLIGIBILITY, **SIMILARITY},
    packs=["fleurs_voice", "scene_voice"],
    io="""item.inputs: {text, ref_audio?, ref_text?, target_s (seconds the take must fill)}
payload: {files: {audio}, takes?: [{seed, duration_s, cer, in_window, accepted}], accept_rate?}""",
)
