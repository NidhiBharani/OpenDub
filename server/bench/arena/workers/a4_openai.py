"""A4/A6 worker: OpenAI audio transcriptions (``POST https://api.openai.com/v1/audio/
transcriptions``, multipart, ``Authorization: Bearer``; 25 MB file limit).

params: model (gpt-transcribe | gpt-4o-transcribe | gpt-4o-mini-transcribe | whisper-1 |
gpt-4o-transcribe-diarize), usd_per_min. Language forced (``language``; gpt-transcribe takes a
``languages[]`` hint list). Only whisper-1 returns word timestamps (``verbose_json`` +
``timestamp_granularities[]=word``); the diarize model returns ``diarized_json`` segments
``{speaker, start, end, text}`` → ``turns`` (needs ``chunking_strategy=auto``).
List prices per audio minute, developers.openai.com/api/docs/pricing read 2026-09-28:
gpt-transcribe 0.0045, gpt-4o-transcribe 0.006, gpt-4o-mini-transcribe 0.003, whisper-1 0.006,
gpt-4o-transcribe-diarize 0.006.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, post_form

URL = "https://api.openai.com/v1/audio/transcriptions"
PRICE = {"gpt-transcribe": 0.0045, "gpt-4o-transcribe": 0.006, "gpt-4o-mini-transcribe": 0.003,
         "whisper-1": 0.006, "gpt-4o-transcribe-diarize": 0.006}


def load(params: dict, lang: str):
    return {"key": api_key("OPENAI_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    model = p.get("model", "gpt-4o-transcribe")
    audio = item["inputs"]["audio"]
    fields: dict = {"model": model}
    if model == "gpt-transcribe":
        fields["languages[]"] = [lang] if lang else None
    else:
        fields["language"] = lang or None
    if model == "whisper-1":
        fields.update({"response_format": "verbose_json", "timestamp_granularities[]": "word"})
    elif model.endswith("diarize"):
        fields.update({"response_format": "diarized_json", "chunking_strategy": "auto"})
    else:
        fields["response_format"] = "json"
    res = post_form(URL, fields, {"file": audio}, {"Authorization": f"Bearer {state['key']}"})
    text = (res.get("text") or "").strip()
    payload: dict = {"text": text, "segments": [], "language": lang,
                     "_cost_usd": audio_seconds(audio) / 60
                     * float(p.get("usd_per_min", PRICE.get(model, 0.006)))}
    if res.get("words"):
        words = [{"start": w["start"], "end": w["end"], "word": w["word"]} for w in res["words"]]
        payload["segments"] = [{"start": words[0]["start"], "end": words[-1]["end"],
                                "text": text, "words": words}]
    if model.endswith("diarize"):
        segs = res.get("segments") or []
        payload["segments"] = [{"start": s["start"], "end": s["end"], "text": s.get("text", ""),
                                "words": []} for s in segs]
        payload["turns"] = [{"start": s["start"], "end": s["end"], "speaker": str(s["speaker"])}
                            for s in segs if s.get("speaker") is not None]
        if not text:
            payload["text"] = " ".join(s.get("text", "") for s in segs).strip()
    return payload


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model"), "api": URL}


if __name__ == "__main__":
    serve(load, run, describe)
