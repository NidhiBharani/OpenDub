"""A7 — speaker embeddings and voice-bank matching.

Two item kinds share one contract (``inputs: {audio, bank_00, bank_01, …}``: the enrolment clips
as separate keys so the pack loader resolves each relative path; workers also accept a
``bank: [...]`` list of absolute paths):

- **trials** (VoxCeleb1-O style, ``refs.label`` 1 = same speaker, 0 = different, one bank
  entry): the worker's cosine ``scores[0]`` becomes ``trial_score``; **EER** (primary) and
  **minDCF** (p_target 0.01, C_miss = C_fa = 1) are corpus metrics over all trials, bootstrapped
  over speaker groups.
- **bank matching** (``refs.match`` = index of the right bank entry, −1 = not in the bank):
  ``bank_top1`` accuracy; for −1 items a correct abstention is the worker returning
  ``match: null`` (score below its threshold) — workers never abstain unless ``params.abstain``.
"""
from __future__ import annotations

from typing import Any

from ..judges import CorpusMetric, Row, Spec, cost, speed
from ..packs import Item


def _det(scores: list[float], labels: list[float]):
    """Miss and false-alarm rates at every threshold (scores sorted ascending)."""
    import numpy as np

    s = np.asarray(scores, float)
    y = np.asarray(labels, float) > 0.5
    order = np.argsort(s, kind="mergesort")
    y = y[order]
    n_t, n_n = y.sum(), (~y).sum()
    if n_t == 0 or n_n == 0:
        return None
    # threshold just above the k-th lowest score: targets below it are misses
    miss = np.concatenate([[0], np.cumsum(y)]) / n_t
    fa = 1 - np.concatenate([[0], np.cumsum(~y)]) / n_n
    return miss, fa


def eer(scores: list[float], labels: list[float]) -> float:
    d = _det(scores, labels)
    if d is None:
        return float("nan")
    miss, fa = d
    import numpy as np

    k = int(np.argmin(np.abs(miss - fa)))
    return float((miss[k] + fa[k]) / 2)


def min_dcf(scores: list[float], labels: list[float], p_target: float = 0.01,
            c_miss: float = 1.0, c_fa: float = 1.0) -> float:
    """Normalised minimum detection cost (NIST SRE / VoxSRC)."""
    d = _det(scores, labels)
    if d is None:
        return float("nan")
    miss, fa = d
    dcf = c_miss * miss * p_target + c_fa * fa * (1 - p_target)
    return float(dcf.min() / min(c_miss * p_target, c_fa * (1 - p_target)))


def embedding_rows(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    scores = out.get("scores") or []
    rows: list[Row] = []
    label = item.refs.get("label")
    if label is not None and scores:
        rows.append(("trial_score", float(scores[0]), 1.0))
    target = item.refs.get("match")
    if target is not None:
        match = out.get("match")
        rows.append(("bank_top1", float((match if match is not None else -1) == int(target)),
                     1.0))
    return rows


def _label(it: Item) -> float | None:
    v = it.refs.get("label")
    return None if v is None else float(v)


SPEC = Spec(
    id="A7",
    title="Speaker embeddings / voice-bank matching",
    judges={"embedding@1": embedding_rows, "speed@1": speed, "cost@1": cost},
    corpus={"eer": CorpusMetric(pred="trial_score", target=_label, fn=eer),
            "min_dcf": CorpusMetric(pred="trial_score", target=_label, fn=min_dcf)},
    primary={"*": "eer"},
    higher_is_better={"eer": False, "min_dcf": False, "trial_score": True, "bank_top1": True,
                      "rtfx": True, "cost_usd": False},
    threshold={"eer": 0.0025, "min_dcf": 0.01, "bank_top1": 0.01},
    secondary=["min_dcf", "bank_top1", "rtfx", "cost_usd"],
    packs=["a7-voxceleb1", "a7-cv-speakers"],
    io="""item.inputs: {audio: test clip, bank_00, bank_01 …: enrolment clips (or bank: [abs
paths])}; item.refs: {label: 1|0} (trial, one bank entry) or {match: bank index | -1}. payload: {scores: [cosine per bank entry],
match: argmax index or null, embedding_file: .npy of the test embedding}.""",
)
