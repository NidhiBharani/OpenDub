"""A5 worker: CTC forced alignment with a wav2vec2/MMS acoustic model.

params: backend = ctc_forced_aligner (MahmoudAshraf97 package; default checkpoint
MahmoudAshraf/mms-300m-1130-forced-aligner is CC BY-NC 4.0, romanises via uroman) |
whisperx (``whisperx.load_align_model(language_code)`` + ``whisperx.align`` with one segment
covering the clip: jonatasgrosman/wav2vec2-large-xlsr-53-japanese for ja,
theainerd/Wav2Vec2-large-xlsr-hindi for hi, torchaudio WAV2VEC2_ASR_BASE_960H for en);
model (override), batch_size. ``torchaudio.functional.forced_align`` itself stays in torchaudio
2.11, but the MMS_FA bundle is deprecated — the ctc-forced-aligner path avoids it.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a5_common import project, tokenize

ISO3 = {"en": "eng", "hi": "hin", "ja": "jpn", "zh": "cmn", "ko": "kor", "de": "deu",
        "fr": "fra", "es": "spa", "ta": "tam", "te": "tel", "bn": "ben"}


def load(params: dict, lang: str):
    import torch

    backend = params.get("backend", "ctc_forced_aligner")
    st = {"params": params, "lang": lang, "backend": backend}
    if backend == "whisperx":
        import whisperx

        st["wx"] = whisperx
        st["model"], st["meta"] = whisperx.load_align_model(
            language_code=lang, device="cuda", model_name=params.get("model"))
    else:
        from ctc_forced_aligner import load_alignment_model

        kw = {"dtype": torch.float16}
        if params.get("model"):
            kw["model_path"] = params["model"]
        st["model"], st["tok"] = load_alignment_model("cuda", **kw)
    return st


def run(state: dict, item: dict, out: Path) -> dict:
    text, lang, audio = item["inputs"]["text"], state["lang"], item["inputs"]["audio"]
    toks = tokenize(text, lang)
    if state["backend"] == "whisperx":
        wx = state["wx"]
        wav = wx.load_audio(audio)
        seg = [{"start": 0.0, "end": len(wav) / 16000, "text": text}]
        res = wx.align(seg, state["model"], state["meta"], wav, "cuda",
                       return_char_alignments=lang in ("ja", "zh"))
        units = []
        for s in res["segments"]:
            src = s.get("chars") if lang in ("ja", "zh") and s.get("chars") else s.get("words")
            for u in src or []:
                if "start" in u and "end" in u:
                    units.append({"start": u["start"], "end": u["end"],
                                  "word": u.get("word", u.get("char", ""))})
        return {"words": project(toks, units)}
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        load_audio,
        postprocess_results,
        preprocess_text,
    )

    m = state["model"]
    wav = load_audio(audio, m.dtype, m.device)
    em, stride = generate_emissions(m, wav, batch_size=int(state["params"].get("batch_size", 8)))
    tokens_starred, text_starred = preprocess_text(" ".join(toks), romanize=True,
                                                   language=ISO3.get(lang, lang))
    segments, scores, blank = get_alignments(em, tokens_starred, state["tok"])
    spans = get_spans(tokens_starred, segments, blank)
    res = postprocess_results(text_starred, spans, stride, scores)
    units = [{"start": r["start"], "end": r["end"], "word": r["text"]} for r in res]
    return {"words": project(toks, units)}


if __name__ == "__main__":
    serve(load, run)
