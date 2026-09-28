"""F3 — 2D animation mouth retiming (research watch item; the honest default is to decline).

Nothing in 2026 retimes drawn mouths in finished animation; every candidate is a method, a cue
exporter or a negative control (see bench/candidates/F3.yaml). The pack (``scene_dub``) pairs
SPY×FAMILY picture with the official dub audio.

Metric: **flap F1** — frame-level (25 fps) agreement between the mouth-open track the delivered
*picture* shows (``payload.mouth_open``) and when the dub is speaking (``refs.dub_speech`` from
the pack, else an energy VAD of the dub audio). Declining keeps the original flaps, whose track
is the original line timing (``inputs.src_speech``), so the baseline measures today's mismatch.
A cue exporter (Rhubarb) leaves the picture as is; its cue track is scored separately as
``cue_f1`` because a viseme track only helps a rigged character, never finished footage.
Picture-editing negative controls (F1 models run on anime) carry no measurable track: they get
``changed_frac`` and the sync panel, and the user's audit decides.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..judges import Row, Spec, cost, output_file, speed
from ..packs import Item
from .f1 import F_DIRECTIONS, SYNC_PRIMARY, abs_metric, judge_size, read_gray, sync_panel

FPS = 25.0
# Rhubarb shapes with the mouth visibly open (A and X are closed; B is nearly closed).
OPEN_SHAPES = {"C", "D", "E", "F", "G", "H", "open"}  # + f3_decline's open/closed cues


def intervals_to_frames(intervals: list, n: int, fps: float = FPS) -> np.ndarray:
    mask = np.zeros(n, bool)
    for iv in intervals or []:
        s, e = (iv["start"], iv["end"]) if isinstance(iv, dict) else (iv[0], iv[1])
        a, b = max(0, round(float(s) * fps)), min(n, round(float(e) * fps))
        if b > a:
            mask[a:b] = True
    return mask


def energy_vad(path: str | Path, fps: float = FPS) -> list[list[float]]:
    """Speech-ish frames of a (dub) mix by RMS energy. Crude on music-heavy mixes; the pack's
    ``refs.dub_speech`` (line timings) takes precedence whenever it exists."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = audio.mean(axis=1)
    hop = int(sr / fps)
    n = len(x) // hop
    if n == 0:
        return []
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = max(float(np.percentile(db, 10)) + 15.0, float(db.max()) - 35.0)
    voiced = db > thr
    out, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start / fps, i / fps])
            start = None
    if start is not None:
        out.append([start / fps, n / fps])
    return out


def frame_f1(pred: np.ndarray, ref: np.ndarray) -> float:
    tp = float(np.sum(pred & ref))
    fp, fn = float(np.sum(pred & ~ref)), float(np.sum(~pred & ref))
    if tp + fp + fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn)


def dub_speech(item: Item) -> list | None:
    if item.refs.get("dub_speech"):
        return item.refs["dub_speech"]
    audio = item.inputs.get("audio")
    return energy_vad(audio) if audio and Path(audio).exists() else None


def _n_frames(item: Item, out: dict[str, Any]) -> int:
    dur = out.get("duration_s") or item.meta.get("duration_s") or 0
    return round(float(dur) * FPS)


def flap_alignment(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    n = _n_frames(item, out)
    speech = dub_speech(item) if n else None
    if not n or speech is None:
        return []
    ref = intervals_to_frames(speech, n)
    rows: list[Row] = []
    if out.get("mouth_open") is not None:
        rows.append(("flap_f1", frame_f1(intervals_to_frames(out["mouth_open"], n), ref),
                     float(n)))
    if out.get("cues"):
        opened = [c for c in out["cues"] if str(c.get("shape") or c.get("value")) in OPEN_SHAPES]
        rows.append(("cue_f1", frame_f1(intervals_to_frames(opened, n), ref), float(n)))
    return rows


def picture_change(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    """Fraction of frames whose luma differs visibly from the source (0 = picture untouched)."""
    src, video = item.inputs.get("video"), output_file(out, "video")
    if not src or not video:
        return []
    w, h = judge_size(src, 144)
    a, b = read_gray(src, width=w, height=h), read_gray(video, width=w, height=h)
    m = min(len(a), len(b))
    if m == 0:
        return []
    diff = np.abs(a[:m] - b[:m]).reshape(m, -1).mean(axis=1)
    return [("changed_frac", float(np.mean(diff > 2.0)), float(m))]


SPEC = Spec(
    id="F3",
    title="2D animation mouth retiming",
    judges={"flap@1": flap_alignment, "change@1": picture_change, "speed@1": speed,
            "cost@1": cost},
    primary={"*": "flap_f1"},
    higher_is_better={**F_DIRECTIONS, "flap_f1": True, "cue_f1": True, "changed_frac": False},
    threshold={"flap_f1": 0.02},
    secondary=["cue_f1", "changed_frac", SYNC_PRIMARY, "sync_p0_synchformer", "rtfx",
               "cost_usd"],
    model_judges=sync_panel,
    derived={SYNC_PRIMARY: abs_metric("sync_offset_ms_synchformer")},
    packs=["scene_dub"],
    io="""item.inputs: {video: 2D animation clip (original picture), audio: official dub audio,
src_speech?: [[start, end]] original line timings}; item.refs: {dub_speech?: [[start, end]]}.
payload: {files: {video, audio, cues?: json}, mouth_open: [[start, end]] | null (mouth-open
track of the delivered picture; null when unknown), cues?: [{start, end, shape}]}""",
)
