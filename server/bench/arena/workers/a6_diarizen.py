"""A6 worker: DiariZen (BUT Speech@FIT; code MIT, weights CC BY-NC 4.0 — eval only).
``from diarizen.pipelines.inference import DiariZenPipeline``;
``DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")(path)`` → pyannote
Annotation (github.com/BUTSpeechFIT/DiariZen README, 2026-09-28). Installed from git with its
bundled pyannote-audio fork. params: model.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    from diarizen.pipelines.inference import DiariZenPipeline

    return {"pipe": DiariZenPipeline.from_pretrained(
        params.get("model", "BUT-FIT/diarizen-wavlm-large-s80-md-v2")), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    ann = state["pipe"](item["inputs"]["audio"])
    return {"turns": [{"start": s.start, "end": s.end, "speaker": str(k)}
                      for s, _, k in ann.itertracks(yield_label=True)]}


if __name__ == "__main__":
    serve(load, run)
