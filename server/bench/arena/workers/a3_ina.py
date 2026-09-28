"""A3 worker: inaSpeechSegmenter (INA, MIT) ``Segmenter(vad_engine='smn', detect_gender=False)``
→ ``[(label, start, stop)]`` with labels speech / music / noise / noEnergy (PyPI 0.8.0 README,
checked 2026-09-28). It labels singing as music — the known weakness this pack measures.
TensorFlow + onnxruntime-gpu stack: x86_64 only.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    from inaSpeechSegmenter import Segmenter

    return {"seg": Segmenter(vad_engine=params.get("vad_engine", "smn"), detect_gender=False),
            "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    segs = []
    for label, start, stop in state["seg"](item["inputs"]["audio"]):
        if label in ("speech", "male", "female", "music"):
            segs.append({"start": float(start), "end": float(stop),
                         "label": "music" if label == "music" else "speech", "score": 1.0})
    return {"segments": segs}


if __name__ == "__main__":
    serve(load, run)
