"""Paired statistics over per-item scores: weighted means, group bootstrap, Holm, winner sets.

Only items scored for *every* compared candidate count (a paired design), and resampling is over
``group`` (speaker / file / title), never individual frames or sentences of one recording.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetricTable:
    """metric -> cand_key -> group -> [sum(value*weight), sum(weight)] over the paired items."""

    groups: list[str]
    sums: dict[str, np.ndarray]     # cand_key -> (n_groups, 2)
    n_items: int

    def mean(self, cand: str) -> float:
        s = self.sums[cand].sum(axis=0)
        return float(s[0] / s[1]) if s[1] else float("nan")


def build_table(rows: Iterable, metric: str, cand_keys: list[str],
                judges: Iterable[str] | None = None) -> MetricTable | None:
    judges = set(judges) if judges is not None else None
    per: dict[str, dict[str, tuple[str, float, float]]] = defaultdict(dict)
    for r in rows:
        if r["metric"] != metric or r["cand_key"] not in cand_keys:
            continue
        if judges is not None and r["judge"] not in judges:
            continue
        per[r["cand_key"]][r["item"]] = (r["grp"], r["value"], r["weight"])
    if any(c not in per for c in cand_keys):
        cand_keys = [c for c in cand_keys if c in per]
    if not cand_keys:
        return None
    common = set.intersection(*(set(per[c]) for c in cand_keys))
    if not common:
        return None
    group_of = {i: per[cand_keys[0]][i][0] for i in common}
    groups = sorted(set(group_of.values()))
    gidx = {g: k for k, g in enumerate(groups)}
    sums = {}
    for c in cand_keys:
        arr = np.zeros((len(groups), 2))
        for i in common:
            _, v, w = per[c][i]
            arr[gidx[group_of[i]], 0] += v * w
            arr[gidx[group_of[i]], 1] += w
        sums[c] = arr
    return MetricTable(groups=groups, sums=sums, n_items=len(common))


@dataclass
class Ranked:
    cand_key: str
    mean: float
    ci: tuple[float, float]           # 95% CI of the candidate's own mean
    diff_vs_best: float = 0.0         # oriented: > 0 means worse than the best
    diff_ci: tuple[float, float] = (0.0, 0.0)
    p_adj: float | None = None        # Holm-adjusted one-sided p that it is worse than the best
    in_winner_set: bool = True
    notes: list[str] = field(default_factory=list)


def rank(table: MetricTable, higher_is_better: bool, threshold: float = 0.0, *,
         n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> list[Ranked]:
    cands = list(table.sums)
    sign = -1.0 if higher_is_better else 1.0  # oriented score: lower is better
    rng = np.random.default_rng(seed)
    g = len(table.groups)
    idx = rng.integers(0, g, size=(n_boot, g))
    boot = {}
    for c in cands:
        s = table.sums[c][idx]                       # (n_boot, g, 2)
        tot = s.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[c] = tot[:, 0] / tot[:, 1]
    means = {c: table.mean(c) for c in cands}
    best = min(cands, key=lambda c: sign * means[c])

    out: list[Ranked] = []
    pvals: dict[str, float] = {}
    for c in cands:
        lo, hi = np.nanpercentile(boot[c], [2.5, 97.5])
        r = Ranked(c, means[c], (float(lo), float(hi)))
        if c != best:
            d = sign * (boot[c] - boot[best])
            r.diff_vs_best = sign * (means[c] - means[best])
            dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
            r.diff_ci = (float(dlo), float(dhi))
            # one-sided: probability mass where c is not worse than best
            pvals[c] = float(min(1.0, (np.sum(d <= 0) + 1) / (n_boot + 1)))
        out.append(r)

    # Holm step-down over the "worse than best" hypotheses.
    order = sorted(pvals, key=pvals.get)
    m = len(order)
    running = 0.0
    adj: dict[str, float] = {}
    for k, c in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[c]))
        adj[c] = running
    for r in out:
        if r.cand_key == best:
            continue
        r.p_adj = adj[r.cand_key]
        worse = r.p_adj < alpha
        practical = r.diff_vs_best > threshold
        r.in_winner_set = not (worse and practical)
        if worse and not practical:
            r.notes.append("significant but below practical threshold")
    out.sort(key=lambda r: sign * r.mean)
    return out


# ---------------------------------------------------------------- corpus-level metrics

def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def corpus_value(fn, preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) < 3:
        return float("nan")
    if callable(fn):
        return float(fn(list(preds), list(targets)))
    if fn in ("pearson", "spearman") and (np.ptp(preds) == 0 or np.ptp(targets) == 0):
        return float("nan")  # a constant judge carries no ranking information
    if fn == "pearson":
        return float(np.corrcoef(preds, targets)[0, 1])
    if fn == "spearman":
        return float(np.corrcoef(_rankdata(preds), _rankdata(targets))[0, 1])
    if fn in ("kendall", "pairwise_acc"):
        i, j = np.triu_indices(len(preds), 1)
        dp, dt = np.sign(preds[i] - preds[j]), np.sign(targets[i] - targets[j])
        keep = dt != 0
        if not keep.any():
            return float("nan")
        agree = (dp[keep] == dt[keep]).mean()
        return float(agree if fn == "pairwise_acc" else 2 * agree - 1)
    raise ValueError(f"unknown corpus metric {fn!r}")


def rank_corpus(per_cand: dict[str, list[tuple[str, float, float]]], fn, *,
                higher_is_better: bool = True, threshold: float = 0.0, n_boot: int = 2000,
                alpha: float = 0.05, seed: int = 0) -> list[Ranked]:
    """``per_cand[cand] = [(group, pred, target)]`` on the items every candidate scored.
    Group bootstrap (fewer resamples than item means: each one recomputes the statistic)."""
    cands = list(per_cand)
    groups = sorted({g for c in cands for g, _, _ in per_cand[c]})
    gidx = {g: k for k, g in enumerate(groups)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    sign = -1.0 if higher_is_better else 1.0
    arrays = {c: (np.array([gidx[g] for g, _, _ in per_cand[c]]),
                  np.array([p for _, p, _ in per_cand[c]], float),
                  np.array([t for _, _, t in per_cand[c]], float)) for c in cands}
    point = {c: corpus_value(fn, a[1], a[2]) for c, a in arrays.items()}
    boot = {c: np.empty(n_boot) for c in cands}
    for b in range(n_boot):
        counts = np.bincount(draws[b], minlength=len(groups))
        for c, (gi, pr, tg) in arrays.items():
            w = counts[gi]
            sel = np.repeat(np.arange(len(gi)), w)
            boot[c][b] = corpus_value(fn, pr[sel], tg[sel])
    # A judge with no defined statistic (constant output) ranks last instead of poisoning min().
    point = {c: (v if not np.isnan(v) else np.inf * sign) for c, v in point.items()}
    best = min(cands, key=lambda c: sign * point[c])
    out, pvals = [], {}
    def pct(x: np.ndarray) -> tuple[float, float]:
        if np.isnan(x).all():
            return (float("nan"), float("nan"))
        lo, hi = np.nanpercentile(x, [2.5, 97.5])
        return (float(lo), float(hi))

    for c in cands:
        r = Ranked(c, point[c], pct(boot[c]))
        if c != best:
            d = sign * (boot[c] - boot[best])
            r.diff_vs_best = sign * (point[c] - point[best])
            r.diff_ci = pct(d)
            pvals[c] = float(min(1.0, (np.sum(d <= 0) + 1) / (n_boot + 1)))
        out.append(r)
    order = sorted(pvals, key=pvals.get)
    running, adj = 0.0, {}
    for k, c in enumerate(order):
        running = max(running, min(1.0, (len(order) - k) * pvals[c]))
        adj[c] = running
    for r in out:
        if r.cand_key != best:
            r.p_adj = adj[r.cand_key]
            r.in_winner_set = not (r.p_adj < alpha and r.diff_vs_best > threshold)
    out.sort(key=lambda r: sign * r.mean)
    return out
