"""B1 — active speaker detection and face tracking.

Ranked by **line-level speaker→face accuracy with an off-screen class**: for every dialogue line
of the reference, did the model point at the face that speaks it, or correctly say that nobody on
screen speaks it (voice-over, off-screen character)? That is the decision a dub needs — where to
apply lip sync, and which face to crop. AVA-style frame AP is a secondary diagnostic (AVA is
saturated at 94–96 mAP and does not separate candidates; docs/plans/model-ranking.md appendix).

Line assignment from a payload, in order of preference:

1. ``payload.lines`` (VLM workers and ASD workers that decide per line): the predicted line with
   the largest time overlap supplies ``box`` (normalised) or ``track`` (id into ``tracks``); a line
   with neither, or ``offscreen: true``, is an off-screen prediction.
2. Otherwise from ``payload.tracks``: the track with the highest mean speaking probability inside
   the line window, if that mean is ≥ ``payload.threshold`` (default 0.5) and the track covers at
   least ``MIN_COVER`` of the window; else off-screen.

A predicted face is correct when it is the same face as the reference box at the reference time
``t`` (``_bpic.same_face``: IoU ≥ 0.3 or mutual centre containment — detectors disagree on how
much chin and forehead a face box includes).
"""
from __future__ import annotations

from typing import Any

from ..judges import Spec, cost, speed
from ..packs import Item
from ._bpic import Box, as_box, f_row, nearest, overlap, ratio_row, same_face

MIN_COVER = 0.3        # a track must be visible for 30% of a line to be its speaker
TIME_TOL = 0.25        # s: a track box counts as "at t" within this distance


def _track_frames(track: dict[str, Any]) -> list[tuple[float, Box, float]]:
    """``[(t, box, p_speaking)]`` from ``{"boxes": [[t, x1, y1, x2, y2], ...], "speaking": [...]}``
    (``speaking`` optional → 0) or ``{"frames": [{"t", "box", "speaking"}]}``."""
    out = []
    if track.get("frames"):
        for f in track["frames"]:
            b = as_box(f.get("box"))
            if b is not None:
                out.append((float(f["t"]), b, float(f.get("speaking", 0.0) or 0.0)))
        return out
    scores = track.get("speaking") or []
    for k, row in enumerate(track.get("boxes") or []):
        if len(row) >= 5:
            p = float(scores[k]) if k < len(scores) and scores[k] is not None else 0.0
            out.append((float(row[0]), as_box(list(row[1:5])), p))
    return out


def _box_at(frames: list[tuple[float, Box, float]], t: float, lo: float, hi: float) -> Box | None:
    near = nearest(frames, t, key=lambda f: f[0])
    if near is not None and abs(near[0] - t) <= TIME_TOL:
        return near[1]
    inside = [f for f in frames if lo <= f[0] <= hi]
    near = nearest(inside, t, key=lambda f: f[0])
    return near[1] if near is not None else None


def predicted_faces(out: dict[str, Any], ref_lines: list[dict[str, Any]]) -> list[Box | None]:
    """The predicted speaking face (box at the reference time) per reference line, None = off."""
    tracks = {str(tr.get("id", k)): _track_frames(tr)
              for k, tr in enumerate(out.get("tracks") or [])}
    threshold = float(out.get("threshold", 0.5))
    preds: list[Box | None] = []
    for ln in ref_lines:
        s, e = float(ln["start"]), float(ln["end"])
        t = float(ln.get("t", (s + e) / 2))
        plines = out.get("lines")
        if plines is not None:
            best = max(plines, key=lambda p: overlap(s, e, float(p.get("start", -1)),
                                                     float(p.get("end", -1))), default=None)
            if best is None or overlap(s, e, float(best.get("start", -1)),
                                       float(best.get("end", -1))) <= 0:
                preds.append(None)
                continue
            if best.get("offscreen"):
                preds.append(None)
            elif as_box(best.get("box")) is not None:
                preds.append(as_box(best.get("box")))
            elif best.get("track") is not None and str(best["track"]) in tracks:
                preds.append(_box_at(tracks[str(best["track"])], t, s, e))
            else:
                preds.append(None)
            continue
        choice, best_p = None, -1.0
        for frames in tracks.values():
            inside = [f for f in frames if s <= f[0] <= e]
            if not inside:
                continue
            # visible span (+ one sampling step of slack) must cover MIN_COVER of the line
            if inside[-1][0] - inside[0][0] + 0.1 < MIN_COVER * (e - s):
                continue
            p = sum(f[2] for f in inside) / len(inside)
            if p > best_p:
                choice, best_p = frames, p
        preds.append(_box_at(choice, t, s, e) if choice is not None and best_p >= threshold
                     else None)
    return preds


def speaker_lines(item: Item, out: dict[str, Any], row: Any) -> list[tuple[str, float, float]]:
    ref_lines = item.refs.get("lines") or []
    if not ref_lines:
        return []
    preds = predicted_faces(out, ref_lines)
    correct = on_n = on_ok = 0
    tp_off = n_pred_off = n_ref_off = 0
    for ln, pb in zip(ref_lines, preds, strict=True):
        rb = as_box(ln.get("box"))
        if rb is None:
            n_ref_off += 1
            ok = pb is None
            tp_off += ok
        else:
            on_n += 1
            ok = same_face(pb, rb)
            on_ok += ok
        n_pred_off += pb is None
        correct += ok
    rows = ratio_row("line_acc", correct, len(ref_lines))
    rows += ratio_row("onscreen_acc", on_ok, on_n)
    rows += f_row("offscreen_f1", tp_off, n_pred_off, n_ref_off)
    return rows


def average_precision(scores: list[float], labels: list[bool]) -> float | None:
    """Non-interpolated AP (sklearn ``average_precision_score`` semantics)."""
    pos = sum(labels)
    if not pos or pos == len(labels):
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp, ap = 0, 0.0
    for rank, i in enumerate(order, 1):
        if labels[i]:
            tp += 1
            ap += tp / rank
    return ap / pos


def ava_frame_ap(item: Item, out: dict[str, Any], row: Any) -> list[tuple[str, float, float]]:
    """AVA-style AP over reference face boxes: each reference face at time t gets the speaking
    probability of the predicted track box that is the same face at t (0 when none). Clip-level
    AP weighted by the number of reference faces (an approximation to AVA's pooled frame mAP)."""
    faces = item.refs.get("faces") or []
    if not faces:
        return []
    tracks = [_track_frames(tr) for tr in out.get("tracks") or []]
    scores, labels = [], []
    for f in faces:
        t, rb = float(f["t"]), as_box(f.get("box"))
        best = 0.0
        for frames in tracks:
            near = nearest(frames, t, key=lambda x: x[0])
            if near is not None and abs(near[0] - t) <= 0.1 and same_face(near[1], rb, 0.5):
                best = max(best, near[2])
        scores.append(best)
        labels.append(bool(f.get("speaking")))
    ap = average_precision(scores, labels)
    return [("ava_ap", ap, float(len(faces)))] if ap is not None else []


SPEC = Spec(
    id="B1",
    title="Active speaker detection and face tracking",
    judges={"speaker_lines@1": speaker_lines, "ava_ap@1": ava_frame_ap, "speed@1": speed,
            "cost@1": cost},
    primary={"*": "line_acc"},
    higher_is_better={"line_acc": True, "onscreen_acc": True, "offscreen_f1": True,
                      "ava_ap": True, "rtfx": True, "cost_usd": False},
    threshold={"line_acc": 0.02, "ava_ap": 0.01},
    secondary=["onscreen_acc", "offscreen_f1", "ava_ap", "rtfx", "cost_usd"],
    packs=["ava_activespeaker", "unitalk", "b_local"],
    io="""item.inputs: {video, lines?: [{start, end, text?}]} (the dub lines to attribute; workers
may ignore them and emit tracks only); item.meta: {duration_s, fps}.
item.refs: {lines: [{start, end, t, box: [x1,y1,x2,y2] normalised | null (off-screen)}],
faces?: [{t, box, speaking: bool}] (AVA-style per-frame face labels)}.
payload: {tracks: [{id, boxes: [[t, x1, y1, x2, y2], ...] normalised, speaking: [p 0..1, ...]}],
lines?: [{start, end, box? | track? | offscreen?: true, score?}], threshold?: 0.5, fps,
duration_s}""",
)
