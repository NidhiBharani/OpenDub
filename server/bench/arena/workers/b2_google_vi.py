"""B2 / B3 worker: Google Cloud Video Intelligence API (``google-cloud-videointelligence``).

- ``feature: SHOT_CHANGE_DETECTION`` (B2) → shot ranges → hard cuts between consecutive shots.
- ``feature: TEXT_DETECTION`` (B3) → tracked text with per-frame rotated boxes (normalised); the
  box nearest each annotated time is reported. Still images are wrapped into a 1 s video first.

Auth: service account / ADC (``GOOGLE_APPLICATION_CREDENTIALS``); the API has no API-key path in
its docs. Video is sent inline (``input_content``). Pricing (cloud.google.com/video-intelligence/
pricing, checked 2026-09-28): shot change $0.05/min, text detection $0.15/min, billed per started
minute (the first 1,000 free minutes a month are ignored here). **Deprecated 2026-09-14, shuts
down 2027-09-14** (docs.cloud.google.com/video-intelligence/docs).

params: feature, price_per_min (0.05 | 0.15), timeout_s (900), language_hints (B3; from lang).
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

from _sdk import serve
from b_common import probe


def load(params: dict, lang: str) -> dict:
    from google.cloud import videointelligence

    return {"vi": videointelligence, "client": videointelligence.VideoIntelligenceServiceClient(),
            "params": params, "lang": lang}


def _secs(td) -> float:
    return td.total_seconds() if hasattr(td, "total_seconds") else float(td.seconds) + \
        float(getattr(td, "nanos", 0)) / 1e9


def _annotate(state: dict, video: str, feature: str):
    vi, p = state["vi"], state["params"]
    request = {"features": [getattr(vi.Feature, feature)],
               "input_content": Path(video).read_bytes()}
    if feature == "TEXT_DETECTION" and state["lang"]:
        request["video_context"] = {"text_detection_config": {
            "language_hints": p.get("language_hints") or [state["lang"]]}}
    op = state["client"].annotate_video(request=request)
    return op.result(timeout=int(p.get("timeout_s", 900))).annotation_results[0]


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    feature = p.get("feature", "SHOT_CHANGE_DETECTION")
    inputs = item["inputs"]
    if inputs.get("image"):
        video = str(out.parent / f"{out.name}.still.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", inputs["image"], "-t",
                        "1", "-r", "25", "-pix_fmt", "yuv420p", "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2", video], check=True)
        times = [0.5]
    else:
        video = inputs["video"]
        times = inputs.get("times")
    info = probe(video)
    res = _annotate(state, video, feature)
    cost = math.ceil(max(info["duration_s"], 1e-3) / 60.0) * float(p.get("price_per_min", 0.05))
    if feature == "SHOT_CHANGE_DETECTION":
        shots = sorted((_secs(s.start_time_offset), _secs(s.end_time_offset))
                       for s in res.shot_annotations)
        return {"shots": [{"start": a, "end": b} for a, b in shots],
                "transitions": [{"start": a, "end": a, "type": "cut"} for a, _b in shots[1:]],
                "fps": info["fps"], "duration_s": info["duration_s"], "_cost_usd": cost}
    texts = []
    for ann in res.text_annotations:
        for seg in ann.segments:
            frames = [(_secs(f.time_offset), [(v.x, v.y) for v in f.rotated_bounding_box.vertices])
                      for f in seg.frames]
            if not frames:
                continue
            wanted = times if inputs.get("image") is None and times else [None]
            for t in wanted:
                if t is not None and not (_secs(seg.segment.start_time_offset) - 0.5 <= t
                                          <= _secs(seg.segment.end_time_offset) + 0.5):
                    continue
                _ft, verts = min(frames, key=lambda f: abs(f[0] - (t if t is not None else 0.5)))
                xs, ys = [v[0] for v in verts], [v[1] for v in verts]
                texts.append({"text": ann.text, "score": float(seg.confidence),
                              "t": None if inputs.get("image") else t,
                              "box": [min(xs), min(ys), max(xs), max(ys)]})
    return {"texts": texts, "_cost_usd": cost}


def describe(state: dict) -> dict:
    return {"feature": state["params"].get("feature"), "api": "videointelligence v1"}


if __name__ == "__main__":
    serve(load, run, describe)
