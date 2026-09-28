"""G2 — speaker-similarity judge, meta-evaluated against human similarity ratings.

Candidates are speaker encoders (``judge_speaker_sim``: WavLM-SV, ERes2NetV2, ReDimNet2, ECAPA,
WeSpeaker, Resemblyzer) and LLM same-speaker raters. Each returns ``score`` (cosine / rating) for
(take, reference); the leaderboard ranks judges by Spearman correlation with human similarity
(VoxSim, VCC2018, VMC2026 T3 via ``user_ratings``). A better verifier (lower EER) is not
necessarily a better perceptual judge — that is exactly what this ranks.
"""
from __future__ import annotations

from ..judges import CorpusMetric, Spec, cost, speed
from ._gmeta import ref_value, score_judge

SPEC = Spec(
    id="G2",
    title="Speaker-similarity judge — agreement with human similarity ratings",
    judges={"pred@1": score_judge(), "speed@1": speed, "cost@1": cost},
    primary={"*": "srcc"},
    higher_is_better={"srcc": True, "lcc": True, "kendall": True, "pred": True, "rtfx": True,
                      "cost_usd": False},
    threshold={"srcc": 0.02},
    secondary=["lcc", "kendall", "rtfx", "cost_usd"],
    corpus={"srcc": CorpusMetric("pred", ref_value("sim"), "spearman"),
            "lcc": CorpusMetric("pred", ref_value("sim"), "pearson"),
            "kendall": CorpusMetric("pred", ref_value("sim"), "kendall")},
    packs=["voxsim", "vcc2018_sim", "user_ratings"],
    io="""item.inputs: {audio (take), ref_audio (reference voice; may be a list)};
item.refs: {sim: human similarity, higher = more similar}.
payload: {metrics: {sim_<encoder>: cos}, score: cos}.""",
)
