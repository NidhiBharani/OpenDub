"""B2 worker: PySceneDetect detectors (BSD-3; CPU, OpenCV backend).

params:
  detector: content | adaptive | threshold | hash | histogram
  kwargs: detector keyword arguments (e.g. {threshold: 27.0} for content,
          {threshold: 12, fade_bias: 0.0} for threshold)
  min_scene_len: frames (default 15, PySceneDetect's default)

Scene list → transitions: a cut at every scene start after the first. The ``threshold`` detector
reports the boundary in the middle of a fade (its fade in/out span is not exposed), so it is scored
as a cut there.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve

DETECTORS = {"content": "ContentDetector", "adaptive": "AdaptiveDetector",
             "threshold": "ThresholdDetector", "hash": "HashDetector",
             "histogram": "HistogramDetector"}


def load(params: dict, lang: str) -> dict:
    import scenedetect

    cls = getattr(scenedetect, DETECTORS[params.get("detector", "content")])
    return {"sd": scenedetect, "cls": cls, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    sd, p = state["sd"], state["params"]
    video = sd.open_video(item["inputs"]["video"])
    kwargs = dict(p.get("kwargs") or {})
    kwargs.setdefault("min_scene_len", int(p.get("min_scene_len", 15)))
    manager = sd.SceneManager()
    manager.add_detector(state["cls"](**kwargs))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    fps = float(video.frame_rate)
    shots = [{"start": s.get_seconds(), "end": e.get_seconds()} for s, e in scenes]
    cuts = [s["start"] for s in shots[1:]]
    return {"shots": shots,
            "transitions": [{"start": t, "end": t, "type": "cut"} for t in cuts],
            "fps": fps, "duration_s": float(video.duration.get_seconds())}


def describe(state: dict) -> dict:
    return {"scenedetect": state["sd"].__version__,
            "detector": state["params"].get("detector", "content")}


if __name__ == "__main__":
    serve(load, run, describe)
