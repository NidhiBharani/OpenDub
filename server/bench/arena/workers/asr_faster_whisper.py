"""A4 worker: faster-whisper (CTranslate2) transcription with word timestamps.

params: model (name or local dir), compute_type, beam_size, device, vad_filter, word_timestamps.
The item's language is forced from the pack: the arena measures transcription, not language ID.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import CUDA12_LIBS, preload_pip_cuda_libs, serve


def load(params: dict, lang: str):
    preload_pip_cuda_libs(*CUDA12_LIBS)
    from faster_whisper import WhisperModel

    model = WhisperModel(params.get("model", "large-v3-turbo"),
                         device=params.get("device", "cuda"),
                         compute_type=params.get("compute_type", "float16"))
    return {"model": model, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    segments, info = state["model"].transcribe(
        item["inputs"]["audio"], language=state["lang"] or None,
        beam_size=int(p.get("beam_size", 5)),
        word_timestamps=bool(p.get("word_timestamps", True)),
        vad_filter=bool(p.get("vad_filter", False)),
        condition_on_previous_text=bool(p.get("condition_on_previous_text", True)))
    segs = []
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                     "words": [{"start": w.start, "end": w.end, "word": w.word}
                               for w in (s.words or [])]})
    return {"text": " ".join(s["text"] for s in segs).strip(), "segments": segs,
            "language": info.language, "duration_s": info.duration}


def describe(state: dict) -> dict:
    import ctranslate2
    import faster_whisper

    return {"model": state["params"].get("model"), "faster_whisper": faster_whisper.__version__,
            "ctranslate2": ctranslate2.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
