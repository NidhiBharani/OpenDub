"""A4 worker: Hugging Face transformers ASR models, one ``family`` switch per model API.

params: model (HF id), revision, family, dtype (bfloat16|float16), max_new_tokens, prompt
(vibevoice context), word_timestamps (whisper / granite-plus), chunk_s (whisper long-form).

Families (APIs checked against the model cards 2026-09-28; transformers 5.x):

- ``whisper``: ``pipeline("automatic-speech-recognition")`` with ``return_timestamps="word"`` and
  ``generate_kwargs={"language", "task": "transcribe"}`` (openai/whisper-*, Hindi fine-tunes,
  kotoba-whisper, distil-whisper).
- ``cohere``: ``CohereAsrForConditionalGeneration`` + processor(…, language=lang); no timestamps.
- ``granite``: ``AutoModelForSpeechSeq2Seq`` with the ``<|audio|>`` chat prompt; the ``-plus``
  models emit ``[T:N]`` end-time tags (centiseconds mod 1000) when ``word_timestamps``.
- ``voxtral``: ``VoxtralForConditionalGeneration`` + ``apply_transcription_request(language=…)``.
- ``vibevoice``: ``VibeVoiceAsrForConditionalGeneration`` (microsoft/VibeVoice-ASR-HF), decoded
  with ``return_format="parsed"`` into speaker segments → also returns ``turns`` (A6). The model
  has no language switch; ``prompt`` carries the language as context ("Language: Hindi").
- ``mms``: ``Wav2Vec2ForCTC`` with per-language adapters (facebook/mms-1b-all; ISO 639-3).
"""
from __future__ import annotations

import re
from pathlib import Path

from _sdk import serve

LANG_NAMES = {"en": "english", "hi": "hindi", "ja": "japanese", "zh": "chinese",
              "ko": "korean", "de": "german", "fr": "french", "es": "spanish", "it": "italian",
              "pt": "portuguese", "ru": "russian", "nl": "dutch", "ar": "arabic",
              "ta": "tamil", "te": "telugu", "bn": "bengali"}
ISO3 = {"en": "eng", "hi": "hin", "ja": "jpn", "zh": "cmn", "ko": "kor", "de": "deu",
        "fr": "fra", "es": "spa", "ta": "tam", "te": "tel", "bn": "ben"}


def _audio16k(path: str):
    import librosa

    y, _ = librosa.load(path, sr=16000, mono=True)
    return y


def load(params: dict, lang: str):
    import torch
    import transformers as tf

    fam = params.get("family", "whisper")
    mid, rev = params["model"], params.get("revision")
    dtype = getattr(torch, params.get("dtype", "bfloat16"))
    st = {"params": params, "lang": lang, "family": fam}
    common = {"revision": rev}
    if fam == "whisper":
        st["pipe"] = tf.pipeline("automatic-speech-recognition", model=mid, dtype=dtype,
                                 device="cuda:0", **common)
    elif fam == "cohere":
        st["proc"] = tf.AutoProcessor.from_pretrained(mid, **common)
        st["model"] = tf.CohereAsrForConditionalGeneration.from_pretrained(
            mid, dtype=dtype, device_map="cuda", **common)
    elif fam == "granite":
        st["proc"] = tf.AutoProcessor.from_pretrained(mid, **common)
        st["model"] = tf.AutoModelForSpeechSeq2Seq.from_pretrained(
            mid, dtype=dtype, device_map="cuda", **common)
    elif fam == "voxtral":
        st["proc"] = tf.AutoProcessor.from_pretrained(mid, **common)
        st["model"] = tf.VoxtralForConditionalGeneration.from_pretrained(
            mid, dtype=dtype, device_map="cuda", **common)
    elif fam == "vibevoice":
        st["proc"] = tf.AutoProcessor.from_pretrained(mid, **common)
        st["model"] = tf.VibeVoiceAsrForConditionalGeneration.from_pretrained(
            mid, dtype=dtype, device_map="auto", **common)
    elif fam == "mms":
        st["proc"] = tf.AutoProcessor.from_pretrained(mid, target_lang=ISO3[lang], **common)
        model = tf.Wav2Vec2ForCTC.from_pretrained(mid, target_lang=ISO3[lang],
                                                  ignore_mismatched_sizes=True, **common)
        st["model"] = model.to("cuda").eval()
    else:
        raise ValueError(f"unknown family {fam!r}")
    return st


def _new_tokens(ids, inputs):
    n = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    return ids[:, n:] if n and ids.shape[1] > n else ids


def _granite_words(text: str) -> tuple[str, list[dict]]:
    """``hello [T:45] world [T:82]`` → words with end times; start = previous end."""
    words, prev_end, offset, last_raw = [], 0.0, 0.0, -1
    for m in re.finditer(r"(.*?)\[T:(\d+)\]", text):
        raw = int(m.group(2))
        if raw < last_raw:  # counter wraps every 10 s
            offset += 10.0
        last_raw = raw
        end = offset + raw / 100.0
        w = m.group(1).strip()
        if w:
            words.append({"start": prev_end, "end": end, "word": w})
        prev_end = end
    return " ".join(w["word"] for w in words), words


def run(state: dict, item: dict, out: Path) -> dict:
    import torch

    p, lang, fam = state["params"], state["lang"], state["family"]
    path = item["inputs"]["audio"]
    max_new = int(p.get("max_new_tokens", 448))
    if fam == "whisper":
        audio = {"raw": _audio16k(path), "sampling_rate": 16000}
        ts = "word" if p.get("word_timestamps", True) else False
        gen = {"task": "transcribe"}
        if lang:
            gen["language"] = LANG_NAMES.get(lang, lang)
        res = state["pipe"](audio, return_timestamps=ts, generate_kwargs=gen,
                            chunk_length_s=float(p.get("chunk_s", 30)))
        words = [{"start": c["timestamp"][0], "end": c["timestamp"][1], "word": c["text"]}
                 for c in res.get("chunks", []) if c["timestamp"][0] is not None]
        for w in words:
            if w["end"] is None:
                w["end"] = w["start"]
        text = res["text"].strip()
        segs = [{"start": words[0]["start"], "end": words[-1]["end"], "text": text,
                 "words": words}] if words else []
        return {"text": text, "segments": segs, "language": lang}

    proc, model = state["proc"], state["model"]
    with torch.inference_mode():
        if fam == "cohere":
            inputs = proc(_audio16k(path), sampling_rate=16000, return_tensors="pt",
                          language=lang).to(model.device, model.dtype)
            ids = model.generate(**inputs, max_new_tokens=max_new)
            text = proc.batch_decode(_new_tokens(ids, inputs), skip_special_tokens=True)[0]
            return {"text": text.strip(), "segments": [], "language": lang}
        if fam == "granite":
            instr = ("Timestamps: Transcribe the speech. After each word, add a timestamp tag "
                     "showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
                     if p.get("word_timestamps") else "Transcribe the speech into written text.")
            chat = [{"role": "user", "content": f"<|audio|> {instr}"}]
            prompt = proc.tokenizer.apply_chat_template(chat, tokenize=False,
                                                        add_generation_prompt=True)
            inputs = proc(prompt, _audio16k(path), return_tensors="pt").to(model.device)
            ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            text = proc.tokenizer.batch_decode(_new_tokens(ids, inputs),
                                               skip_special_tokens=True)[0].strip()
            if p.get("word_timestamps"):
                text, words = _granite_words(text)
                segs = [{"start": words[0]["start"], "end": words[-1]["end"], "text": text,
                         "words": words}] if words else []
                return {"text": text, "segments": segs, "language": lang}
            return {"text": text, "segments": [], "language": lang}
        if fam == "voxtral":
            inputs = proc.apply_transcription_request(language=lang, audio=path,
                                                      model_id=p["model"])
            inputs = inputs.to(model.device, dtype=model.dtype)
            ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            text = proc.batch_decode(_new_tokens(ids, inputs), skip_special_tokens=True)[0]
            return {"text": text.strip(), "segments": [], "language": lang}
        if fam == "vibevoice":
            prompt = p.get("prompt") or (f"Language: {LANG_NAMES.get(lang, lang).title()}"
                                         if lang else None)
            inputs = proc.apply_transcription_request(audio=path, prompt=prompt)
            inputs = inputs.to(model.device, model.dtype)
            gen = {"max_new_tokens": int(p.get("max_new_tokens", 8192))}
            if p.get("acoustic_tokenizer_chunk_size"):
                gen["acoustic_tokenizer_chunk_size"] = int(p["acoustic_tokenizer_chunk_size"])
            ids = model.generate(**inputs, **gen)
            parsed = proc.decode(_new_tokens(ids, inputs), return_format="parsed")[0]
            segs, turns = [], []
            for s in parsed or []:
                start, end = float(s.get("Start", 0)), float(s.get("End", 0))
                txt = str(s.get("Content", "")).strip()
                segs.append({"start": start, "end": end, "text": txt, "words": []})
                if s.get("Speaker") is not None:
                    turns.append({"start": start, "end": end, "speaker": str(s["Speaker"])})
            return {"text": " ".join(s["text"] for s in segs).strip(), "segments": segs,
                    "turns": turns, "language": lang}
        if fam == "mms":
            inputs = proc(_audio16k(path), sampling_rate=16000, return_tensors="pt").to("cuda")
            logits = model(**inputs).logits
            text = proc.decode(torch.argmax(logits, dim=-1)[0])
            return {"text": text.strip(), "segments": [], "language": lang}
    raise ValueError(fam)


def describe(state: dict) -> dict:
    import transformers

    p = state["params"]
    return {"model": p.get("model"), "revision": p.get("revision"), "family": p.get("family"),
            "transformers": transformers.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
