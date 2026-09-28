"""G4 — emotion-consistency judge, meta-evaluated against human "same feeling" labels.

Candidates: emotion2vec+ cosine, Odyssey-2024 WavLM A/V/D distance (``judge_emotion``) and LLM
"same feeling" raters (both A/B orders averaged). Ranked by Spearman correlation of ``score``
with refs.emo — EMOS from VoiceMOS 2026 T2 (``user_ratings``) or 1/0 same-emotion labels on
ESD/JVNV pairs (``emotion_pairs``; different speakers and texts, half cross-lingual for ESD).
``auc`` is the same-emotion detection AUC on the binary items.
"""
from __future__ import annotations

from ..judges import CorpusMetric, Spec, cost, speed
from ._gmeta import auc, binary_ref, ref_value, score_judge

SPEC = Spec(
    id="G4",
    title="Emotion-consistency judge — agreement with human emotion labels",
    judges={"pred@1": score_judge(), "speed@1": speed, "cost@1": cost},
    primary={"*": "srcc"},
    higher_is_better={"srcc": True, "auc": True, "pred": True, "rtfx": True, "cost_usd": False},
    threshold={"srcc": 0.02},
    secondary=["auc", "rtfx", "cost_usd"],
    corpus={"srcc": CorpusMetric("pred", ref_value("emo"), "spearman"),
            "auc": CorpusMetric("pred", binary_ref("emo"), auc)},
    packs=["emotion_pairs", "user_ratings"],
    io="""item.inputs: {audio (take / dub), ref_audio (source line)};
item.refs: {emo: EMOS or 1 same emotion / 0 different}.
payload: {metrics: {...}, score: similarity (higher = same feeling)}.""",
)
