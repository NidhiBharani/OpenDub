"""A6/A7 worker: pyannote.audio pipelines and embeddings.

params: model (pyannote/speaker-diarization-3.1 | pyannote/speaker-diarization-community-1 |
pyannote/wespeaker-voxceleb-resnet34-LM for A7), task (diarize | embed), exclusive
(community-1 only: the no-overlap view), num_speakers (optional oracle count; default none —
the arena measures estimated counts). Gated HF repos: set HF_TOKEN after accepting terms.

pyannote.audio 3.x (3.4.0): ``Pipeline.from_pretrained(id, use_auth_token=…)``, output
``Annotation.itertracks(yield_label=True)``. 4.x (4.0.7, needs torchcodec + ffmpeg):
``Pipeline.from_pretrained(id, token=…)``, output ``.speaker_diarization`` iterated as
``(turn, speaker)``. Model cards checked 2026-09-28. A7 embeddings via
``Inference(Model.from_pretrained(id), window="whole")``.
"""
from __future__ import annotations

import os
from pathlib import Path

from _sdk import serve


def _token_kw(fn) -> dict:
    import inspect

    tok = os.environ.get("HF_TOKEN")
    if not tok:
        return {}
    return {"token": tok} if "token" in inspect.signature(fn).parameters else \
        {"use_auth_token": tok}


def load(params: dict, lang: str):
    import torch

    task = params.get("task", "diarize")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if task == "embed":
        from pyannote.audio import Inference, Model

        model = Model.from_pretrained(params["model"], **_token_kw(Model.from_pretrained))
        inf = Inference(model, window="whole")
        inf.to(dev)
        return {"task": task, "inf": inf, "params": params}
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained(params["model"], **_token_kw(Pipeline.from_pretrained))
    pipe.to(dev)
    return {"task": task, "pipe": pipe, "params": params}


def embed(state: dict, path: str):
    import numpy as np

    return np.asarray(state["inf"](path), dtype=np.float32).reshape(-1)


def run(state: dict, item: dict, out: Path) -> dict:
    if state["task"] == "embed":
        from a7_common import score_item

        return score_item(lambda p: embed(state, p), item, out, state["params"])
    p = state["params"]
    kw = {}
    if p.get("num_speakers"):
        kw["num_speakers"] = int(p["num_speakers"])
    res = state["pipe"](item["inputs"]["audio"], **kw)
    if hasattr(res, "speaker_diarization"):  # pyannote.audio 4
        ann = res.exclusive_speaker_diarization if p.get("exclusive") else res.speaker_diarization
        turns = [{"start": t.start, "end": t.end, "speaker": str(s)} for t, s in ann]
    else:
        turns = [{"start": seg.start, "end": seg.end, "speaker": str(spk)}
                 for seg, _, spk in res.itertracks(yield_label=True)]
    return {"turns": turns}


def describe(state: dict) -> dict:
    import pyannote.audio

    return {"model": state["params"].get("model"), "pyannote_audio": pyannote.audio.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
