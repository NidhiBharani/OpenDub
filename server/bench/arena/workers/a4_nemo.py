"""A4 worker: NVIDIA NeMo ASR (Parakeet TDT / TDT-CTC, Canary, Nemotron ASR, Canary-Qwen SALM).

params: model (HF id), family (asr | canary | salm), timestamps (bool, default true),
max_new_tokens (salm). Language is forced for Canary (source_lang = target_lang = lang); the
monolingual/EU Parakeet models have no language switch, so the worker refuses languages the
candidate does not declare (``params.languages``) rather than letting them guess.

Verified 2026-09-28 (model cards): ``ASRModel.from_pretrained(id).transcribe([wav],
timestamps=True)`` → ``out[0].text`` and ``out[0].timestamp['word']`` (dicts with start/end in
seconds and ``word``); Canary takes ``source_lang``/``target_lang``; Canary-Qwen is
``nemo.collections.speechlm2.models.SALM`` with ``generate(prompts=…)`` and has no timestamps.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve


def load(params: dict, lang: str):
    langs = params.get("languages")
    if langs and lang and lang not in langs:
        raise ValueError(f"{params.get('model')} does not declare language {lang!r}")
    family = params.get("family", "asr")
    if family == "salm":
        from nemo.collections.speechlm2.models import SALM

        model = SALM.from_pretrained(params["model"])
    else:
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.from_pretrained(params["model"])
    model = model.cuda().eval() if params.get("device", "cuda") == "cuda" else model.eval()
    return {"model": model, "params": params, "lang": lang, "family": family}


def _mono16k(path: str, tmp: Path) -> str:
    """NeMo expects 16 kHz mono files; convert anything else next to the output."""
    import soundfile as sf

    info = sf.info(path)
    if info.samplerate == 16000 and info.channels == 1:
        return path
    import librosa

    y, _ = librosa.load(path, sr=16000, mono=True)
    sf.write(str(tmp), y, 16000)
    return str(tmp)


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    wav = _mono16k(item["inputs"]["audio"], out.with_suffix(".16k.wav"))
    if state["family"] == "salm":
        m = state["model"]
        prompt = [[{"role": "user", "content": f"Transcribe the following: {m.audio_locator_tag}",
                    "audio": [wav]}]]
        ids = m.generate(prompts=prompt, max_new_tokens=int(p.get("max_new_tokens", 256)))
        text = m.tokenizer.ids_to_text(ids[0].cpu())
        return {"text": text.strip(), "segments": [], "language": lang}
    kwargs = {"timestamps": bool(p.get("timestamps", True))}
    if state["family"] == "canary":
        if not lang:
            raise Unsupported("canary needs a language")
        kwargs.update(source_lang=lang, target_lang=lang)
    hyp = state["model"].transcribe([wav], **kwargs)
    hyp = hyp[0] if isinstance(hyp, list) else hyp
    if isinstance(hyp, tuple):  # older RNNT API: (best, all)
        hyp = hyp[0][0] if isinstance(hyp[0], list) else hyp[0]
    text = getattr(hyp, "text", str(hyp)).strip()
    words = []
    ts = getattr(hyp, "timestamp", None) or {}
    for w in ts.get("word", []) if isinstance(ts, dict) else []:
        words.append({"start": float(w["start"]), "end": float(w["end"]),
                      "word": w.get("word") or w.get("segment", "")})
    segs = [{"start": words[0]["start"], "end": words[-1]["end"], "text": text,
             "words": words}] if words else []
    return {"text": text, "segments": segs, "language": lang}


def describe(state: dict) -> dict:
    import nemo

    return {"model": state["params"].get("model"), "nemo": nemo.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
