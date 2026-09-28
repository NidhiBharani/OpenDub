"""B2 worker: TransNetV2 (PyTorch), optionally merged with a PySceneDetect threshold/fade channel
(the compute-tiers Spark pick: "TransNetV2 + PySceneDetect threshold/fade detectors as a second
channel").

Backend: the ``transnetv2-pytorch`` PyPI package (1.0.5, MIT, a fork of soCzech/TransNetV2 whose
wheel bundles converted PyTorch weights). Its per-frame probabilities are thresholded here the
same way as upstream ``predictions_to_scenes`` (a transition is a run of frames above the
threshold), so a run longer than a couple of frames is reported as a gradual transition.
Provenance of the converted weights is weaker than upstream (the fork's README is a template):
``weights`` may point at a ``transnetv2-pytorch-weights.pth`` converted locally with the
upstream ``inference-pytorch/convert_weights.py`` instead.

params: threshold (0.5), weights (optional path), second_channel: null | threshold (PySceneDetect
ThresholdDetector fades/black frames; merged, deduplicated within ``merge_tol_s``),
threshold_kwargs ({threshold: 12}), merge_tol_s (0.5).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from b_common import probe


def load(params: dict, lang: str) -> dict:
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="cuda" if torch.cuda.is_available() else "cpu")
    if params.get("weights"):
        state = torch.load(str(Path(params["weights"]).expanduser()), map_location="cpu")
        target = getattr(model, "model", model)
        target.load_state_dict(state)
    return {"model": model, "params": params}


def _runs(probs, threshold: float) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, p in enumerate(probs):
        if p > threshold and start is None:
            start = i
        elif p <= threshold and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(probs) - 1))
    return runs


def _single_frame_probs(model, video: str):
    """Per-frame transition probabilities from whichever API the installed package exposes."""
    import numpy as np

    res = model.predict_video(video)
    if isinstance(res, tuple):  # upstream-style (frames, single_frame_pred, all_frame_pred)
        single = res[1] if len(res) >= 3 else res[0]
    elif isinstance(res, dict):
        single = res.get("single_frame_predictions", res.get("predictions"))
    else:
        single = res
    arr = np.asarray(single.detach().cpu() if hasattr(single, "detach") else single, dtype=float)
    return arr.reshape(-1)


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    video = item["inputs"]["video"]
    info = probe(video)
    fps = info["fps"]
    probs = _single_frame_probs(state["model"], video)
    transitions = []
    for a, b in _runs(probs, float(p.get("threshold", 0.5))):
        # a one/two-frame run is a hard cut: the new shot starts at the frame after the peak
        if b - a <= 1:
            t = (b + 1) / fps
            transitions.append({"start": t, "end": t, "type": "cut",
                                "score": float(probs[a:b + 1].max())})
        else:
            transitions.append({"start": a / fps, "end": (b + 1) / fps, "type": "gradual",
                                "score": float(probs[a:b + 1].max())})
    if p.get("second_channel") == "threshold":
        import scenedetect

        kw = dict(p.get("threshold_kwargs") or {"threshold": 12})
        scenes = scenedetect.detect(video, scenedetect.ThresholdDetector(**kw))
        tol = float(p.get("merge_tol_s", 0.5))
        for s, _e in scenes[1:]:
            t = s.get_seconds()
            if all(abs(t - (tr["start"] + tr["end"]) / 2) > tol for tr in transitions):
                transitions.append({"start": t, "end": t, "type": "fade", "score": None,
                                    "channel": "scenedetect-threshold"})
        transitions.sort(key=lambda tr: tr["start"])
    return {"transitions": transitions, "fps": fps, "duration_s": info["duration_s"]}


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"transnetv2_pytorch": version("transnetv2-pytorch"),
            "weights": state["params"].get("weights") or "bundled",
            "second_channel": state["params"].get("second_channel")}


if __name__ == "__main__":
    serve(load, run, describe)
