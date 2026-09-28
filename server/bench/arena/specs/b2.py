"""B2 — shot boundary detection.

Ranked by **transition F1**: hard cuts match within ±2 frames, gradual transitions (dissolves,
fades, wipes) match when the predicted span overlaps the reference span (also with ±2 frames of
slack), one-to-one, greedily by centre distance. Secondary: precision, recall, false positives
per hour, and **false cuts inside dialogue lines** — a spurious cut mid-line splits a subtitle and
resets a lip-sync face track, which is the failure a dub actually sees (flashes, title cards,
archival damage, anime impact frames).

Transitions are frame intervals ``[a, b]`` with ``a = round(start·fps)``, ``b = round(end·fps)``;
a hard cut at time ``t`` (first frame of the new shot) is ``[round(t·fps), round(t·fps)]``.
"""
from __future__ import annotations

from itertools import pairwise
from typing import Any

from ..judges import Spec, cost, speed
from ..packs import Item
from ._bpic import f_row, greedy_match, ratio_row

CUT_TOL_FRAMES = 2
GRADUAL_MIN_FRAMES = 2   # a reference span longer than this many frames is a gradual transition


def transitions_of(obj: dict[str, Any], fps: float) -> list[tuple[int, int]]:
    """Frame intervals from ``transitions`` [{start, end}], ``cuts`` [t] or ``shots`` [{start,
    end}] (boundary between consecutive shots; a gap longer than 1.5 frames is a gradual span)."""
    if obj.get("transitions") is not None:
        out = []
        for tr in obj["transitions"]:
            s = float(tr["start"]) if isinstance(tr, dict) else float(tr)
            e = float(tr.get("end", s)) if isinstance(tr, dict) else s
            out.append((round(min(s, e) * fps), round(max(s, e) * fps)))
        return sorted(out)
    if obj.get("cuts") is not None:
        return sorted((round(float(t) * fps),) * 2 for t in obj["cuts"])
    shots = sorted((float(s["start"]), float(s["end"])) for s in obj.get("shots") or [])
    out = []
    for (_s0, e0), (s1, _e1) in pairwise(shots):
        if s1 - e0 > 1.5 / fps:
            out.append((round(e0 * fps), round(s1 * fps)))
        else:
            out.append((round(s1 * fps),) * 2)
    return out


def _is_gradual(tr: tuple[int, int]) -> bool:
    return tr[1] - tr[0] > GRADUAL_MIN_FRAMES


def match_transitions(pred: list[tuple[int, int]], ref: list[tuple[int, int]],
                      tol: int = CUT_TOL_FRAMES) -> list[tuple[int, int]]:
    def score(i: int, j: int) -> float | None:
        p, r = pred[i], ref[j]
        if p[1] + tol < r[0] or r[1] + tol < p[0]:
            return None
        return abs((p[0] + p[1]) / 2 - (r[0] + r[1]) / 2)
    return greedy_match(len(pred), len(ref), score, higher_is_better=False)


def transition_scores(item: Item, out: dict[str, Any], row: Any) -> list[tuple[str, float, float]]:
    if "transitions" not in item.refs and "cuts" not in item.refs and "shots" not in item.refs:
        return []
    fps = float(item.meta.get("fps") or out.get("fps") or 25.0)
    ref = transitions_of(item.refs, fps)
    pred = transitions_of(out, fps)
    pairs = match_transitions(pred, ref)
    tp = len(pairs)
    rows = f_row("f1", tp, len(pred), len(ref))
    rows += ratio_row("precision", tp, len(pred))
    rows += ratio_row("recall", tp, len(ref))
    matched_ref = {j for _, j in pairs}
    cuts = [j for j, r in enumerate(ref) if not _is_gradual(r)]
    grads = [j for j, r in enumerate(ref) if _is_gradual(r)]
    rows += ratio_row("cut_recall", sum(j in matched_ref for j in cuts), len(cuts))
    rows += ratio_row("gradual_recall", sum(j in matched_ref for j in grads), len(grads))

    matched_pred = {i for i, _ in pairs}
    false = [pred[i] for i in range(len(pred)) if i not in matched_pred]
    dur = float(item.meta.get("duration_s") or out.get("duration_s") or 0.0)
    if dur > 0:
        hours = dur / 3600.0
        rows.append(("fp_per_hour", len(false) / hours, hours))
    lines = item.refs.get("lines") or []
    if lines:
        margin = 0.1  # s: a cut within 100 ms of a line edge is at the edge, not inside
        inside = 0
        for a, b in false:
            t = (a + b) / 2 / fps
            inside += any(float(ln["start"]) + margin < t < float(ln["end"]) - margin
                          for ln in lines)
        rows += ratio_row("false_cuts_per_line", inside, len(lines))
    return rows


SPEC = Spec(
    id="B2",
    title="Shot boundary detection",
    judges={"transitions@1": transition_scores, "speed@1": speed, "cost@1": cost},
    primary={"*": "f1"},
    higher_is_better={"f1": True, "precision": True, "recall": True, "cut_recall": True,
                      "gradual_recall": True, "fp_per_hour": False,
                      "false_cuts_per_line": False, "rtfx": True, "cost_usd": False},
    threshold={"f1": 0.01},
    secondary=["recall", "precision", "false_cuts_per_line", "fp_per_hour", "cut_recall",
               "gradual_recall", "rtfx", "cost_usd"],
    packs=["clipshots", "autoshot_shot", "bbc_planet_earth", "b_local"],
    io="""item.inputs: {video}; item.meta: {fps, duration_s, n_frames?}.
item.refs: {transitions: [{start, end, type: cut|dissolve|fade|wipe|other}] in seconds (a cut has
start == end == first frame time of the new shot), lines?: [{start, end}] dialogue lines}.
payload: {transitions: [{start, end, type?, score?}] and/or shots: [{start, end}] (seconds), fps,
duration_s}""",
)
