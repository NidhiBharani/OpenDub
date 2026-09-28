"""A4/A6 worker: Gemini 3.5 Transcribe (Google DeepMind, launched 2026-08-26).

``POST https://generativelanguage.googleapis.com/v1beta/interactions`` (header
``x-goog-api-key``) with ``{"model": "gemini-3.5-transcribe", "input": [{"type": "audio",
"uri": <Files API uri>, "mime_type": …}], "generation_config": {"transcription_config":
{"language_codes": ["hi-IN"], "mode": {"type": "verbatim", "timestamp_granularities":
["word"]}}}}``. Response: ``output_text``; words in ``steps[].content[].annotations[]`` with
``type == "word_info"`` and ``{text, speaker, start_offset, end_offset}``. Per
ai.google.dev/gemini-api/docs/transcribe, read 2026-09-28 (offsets may be "1.23s" strings).
Audio is uploaded with the Files API resumable protocol first.

params: model, diarize, usd_per_min (default 0.005 = ~$0.003/min audio input + ~$0.002/min
text output, ai.google.dev/gemini-api/docs/pricing read 2026-09-28).
"""
from __future__ import annotations

import json
import mimetypes
import urllib.request
from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, request, words_to_turns

BASE = "https://generativelanguage.googleapis.com"
LOCALES = {"en": "en-US", "hi": "hi-IN", "ja": "ja-JP", "zh": "cmn-Hans-CN", "ko": "ko-KR",
           "de": "de-DE", "fr": "fr-FR", "es": "es-ES", "pt": "pt-BR", "ta": "ta-IN",
           "te": "te-IN", "bn": "bn-IN"}


def upload(key: str, path: str) -> tuple[str, str]:
    """Files API resumable upload → (file uri, mime type)."""
    data = Path(path).read_bytes()
    mime = mimetypes.guess_type(path)[0] or "audio/wav"
    start = urllib.request.Request(
        f"{BASE}/upload/v1beta/files", method="POST",
        data=json.dumps({"file": {"display_name": Path(path).name}}).encode(),
        headers={"x-goog-api-key": key, "X-Goog-Upload-Protocol": "resumable",
                 "X-Goog-Upload-Command": "start",
                 "X-Goog-Upload-Header-Content-Length": str(len(data)),
                 "X-Goog-Upload-Header-Content-Type": mime,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(start, timeout=120) as resp:
        url = resp.headers["X-Goog-Upload-URL"]
    res = request("POST", url, data=data, headers={
        "Content-Length": str(len(data)), "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"})
    return res["file"]["uri"], mime


def _secs(v) -> float:
    if isinstance(v, str):
        return float(v.rstrip("s"))
    if isinstance(v, dict):  # protobuf Duration
        return float(v.get("seconds", 0)) + float(v.get("nanos", 0)) / 1e9
    return float(v)


def load(params: dict, lang: str):
    return {"key": api_key("GEMINI_API_KEY", "GOOGLE_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    audio = item["inputs"]["audio"]
    uri, mime = upload(state["key"], audio)
    cfg: dict = {"language_codes": [LOCALES.get(lang, lang)],
                 "mode": {"type": "verbatim", "timestamp_granularities": ["word"]}}
    if p.get("diarize"):
        cfg["diarization"] = True
    body = {"model": p.get("model", "gemini-3.5-transcribe"),
            "input": [{"type": "audio", "uri": uri, "mime_type": mime}],
            "generation_config": {"transcription_config": cfg}}
    res = request("POST", f"{BASE}/v1beta/interactions", json_body=body,
                  headers={"x-goog-api-key": state["key"]})
    words = []
    for step in res.get("steps") or []:
        for c in step.get("content") or []:
            for a in c.get("annotations") or []:
                if a.get("type") == "word_info":
                    words.append({"start": _secs(a["start_offset"]),
                                  "end": _secs(a["end_offset"]), "word": a.get("text", ""),
                                  "speaker": a.get("speaker")})
    text = (res.get("output_text") or " ".join(w["word"] for w in words)).strip()
    payload = {"text": text, "language": lang,
               "segments": [{"start": words[0]["start"], "end": words[-1]["end"], "text": text,
                             "words": [{k: w[k] for k in ("start", "end", "word")}
                                       for w in words]}] if words else [],
               "_cost_usd": audio_seconds(audio) / 60 * float(p.get("usd_per_min", 0.005))}
    if p.get("diarize"):
        payload["turns"] = words_to_turns(words)
    return payload


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model", "gemini-3.5-transcribe")}


if __name__ == "__main__":
    serve(load, run, describe)
