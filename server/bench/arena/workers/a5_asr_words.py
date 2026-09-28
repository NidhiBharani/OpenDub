"""A5 worker: ASR word timestamps projected onto the given text (what OpenDub does today: A4
faster-whisper word timings, no separate aligner). The transcript's words are aligned to the
reference text by characters (``a5_common.project``); text the ASR missed is interpolated.
params: model, compute_type, beam_size (faster-whisper). Server env.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import CUDA12_LIBS, preload_pip_cuda_libs, serve
from a5_common import project, tokenize


def load(params: dict, lang: str):
    preload_pip_cuda_libs(*CUDA12_LIBS)
    from faster_whisper import WhisperModel

    model = WhisperModel(params.get("model", "large-v3-turbo"), device="cuda",
                         compute_type=params.get("compute_type", "float16"))
    return {"model": model, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    segs, _ = state["model"].transcribe(
        item["inputs"]["audio"], language=state["lang"] or None, word_timestamps=True,
        beam_size=int(state["params"].get("beam_size", 5)),
        initial_prompt=item["inputs"]["text"] if state["params"].get("prompt_text") else None)
    units = [{"start": w.start, "end": w.end, "word": w.word} for s in segs for w in (s.words
                                                                                        or [])]
    return {"words": project(tokenize(item["inputs"]["text"], state["lang"]), units),
            "asr_words": units}


if __name__ == "__main__":
    serve(load, run)
