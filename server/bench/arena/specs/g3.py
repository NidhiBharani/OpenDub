"""G3 — naturalness (MOS) predictor, meta-evaluated against human MOS.

Candidates are MOS predictors (``judge_mos``: UTMOSv2, Distill-MOS, Audiobox Aesthetics, NISQA,
NISQA-TTS, DNSMOS, XLS-R-SQA) and LLM MOS raters. Ranked by utterance-level Spearman correlation
with human MOS (BVCC, SOMOS, TTSDS2, LIMMITS/VMC via ``user_ratings``, and the user's ok/bad
audit marks as a binary target). Compare within a language only (plan appendix G3). Groups are
synthesis systems, so the CI reflects system-level variability; a system-level SRCC column is
not computed yet (CorpusMetric sees items, not systems).
"""
from __future__ import annotations

from ..judges import CorpusMetric, Spec, cost, speed
from ._gmeta import ref_value, score_judge

TARGET = ref_value("mos", "ok")

SPEC = Spec(
    id="G3",
    title="Naturalness (MOS) predictor — agreement with human MOS",
    judges={"pred@1": score_judge(), "speed@1": speed, "cost@1": cost},
    primary={"*": "srcc"},
    higher_is_better={"srcc": True, "lcc": True, "kendall": True, "pairwise_acc": True,
                      "pred": True, "rtfx": True, "cost_usd": False},
    threshold={"srcc": 0.02},
    secondary=["lcc", "kendall", "pairwise_acc", "rtfx", "cost_usd"],
    corpus={"srcc": CorpusMetric("pred", TARGET, "spearman"),
            "lcc": CorpusMetric("pred", TARGET, "pearson"),
            "kendall": CorpusMetric("pred", TARGET, "kendall"),
            "pairwise_acc": CorpusMetric("pred", TARGET, "pairwise_acc")},
    packs=["bvcc", "somos", "ttsds2_ratings", "arena_audit", "user_ratings"],
    io="""item.inputs: {audio}; item.refs: {mos} (or {ok: 0|1} from audit marks).
payload: {metrics: {mos_<model>: …}, score: <primary sub-score>}.""",
)
