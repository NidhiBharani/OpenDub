"""A4 worker: WhisperX (faster-whisper batched transcription + wav2vec2 forced alignment).

params: model (large-v3 …), compute_type, batch_size, align_model (override the per-language
default), vad (whisperx default). Language forced at load and for alignment.
Checked 2026-09-28 (whisperx 3.8.6, alignment.py defaults: ja
jonatasgrosman/wav2vec2-large-xlsr-53-japanese, hi theainerd/Wav2Vec2-large-xlsr-hindi).
x86_64 only on GPU: CTranslate2's aarch64 PyPI wheels are CPU-only.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import CUDA12_LIBS, preload_pip_cuda_libs, serve


def load(params: dict, lang: str):
    preload_pip_cuda_libs(*CUDA12_LIBS)
    import whisperx

    model = whisperx.load_model(params.get("model", "large-v3"), "cuda",
                                compute_type=params.get("compute_type", "float16"),
                                language=lang or None)
    align, meta = whisperx.load_align_model(language_code=lang, device="cuda",
                                            model_name=params.get("align_model"))
    return {"wx": whisperx, "model": model, "align": align, "meta": meta, "params": params,
            "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    wx = state["wx"]
    audio = wx.load_audio(item["inputs"]["audio"])
    res = state["model"].transcribe(audio, batch_size=int(state["params"].get("batch_size", 16)),
                                    language=state["lang"] or None)
    res = wx.align(res["segments"], state["align"], state["meta"], audio, "cuda",
                   return_char_alignments=False)
    segs = []
    for s in res["segments"]:
        words = [{"start": w["start"], "end": w["end"], "word": w["word"]}
                 for w in s.get("words", []) if "start" in w and "end" in w]
        segs.append({"start": s["start"], "end": s["end"], "text": s["text"].strip(),
                     "words": words})
    return {"text": " ".join(s["text"] for s in segs).strip(), "segments": segs,
            "language": state["lang"]}


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"model": state["params"].get("model"), "whisperx": version("whisperx")}


if __name__ == "__main__":
    serve(load, run, describe)
