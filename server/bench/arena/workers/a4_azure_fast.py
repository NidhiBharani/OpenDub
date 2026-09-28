"""A4/A6 worker: Azure AI Speech fast transcription (incl. Microsoft MAI-Transcribe-2).

``POST https://{region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe
?api-version=2025-10-15`` (header ``Ocp-Apim-Subscription-Key``), multipart ``audio`` +
``definition`` JSON ``{"locales": ["hi-IN"], "model": "mai-transcribe-2"?, "diarization":
{"enabled": true, "maxSpeakers": N}?}``. Words at ``phrases[].words[{text, offsetMilliseconds,
durationMilliseconds}]``; ``phrases[].speaker`` when diarizing. learn.microsoft.com fast-
transcription page read 2026-09-28. ``endpoint`` param overrides the host for Foundry
resources. Price params: fast transcription ~$0.36/h (unverified), MAI-Transcribe-2 $0.10/h
launch price until end of 2026 (microsoft.ai, 2026-09-03).
"""
from __future__ import annotations

import os
from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, post_form

LOCALES = {"en": "en-US", "hi": "hi-IN", "ja": "ja-JP", "zh": "zh-CN", "ko": "ko-KR",
           "de": "de-DE", "fr": "fr-FR", "es": "es-ES", "pt": "pt-BR", "ta": "ta-IN",
           "te": "te-IN", "bn": "bn-IN"}


def load(params: dict, lang: str):
    key = api_key("AZURE_SPEECH_KEY")
    host = params.get("endpoint") or \
        f"https://{os.environ.get('AZURE_SPEECH_REGION', 'eastus')}.api.cognitive.microsoft.com"
    return {"key": key, "host": host.rstrip("/"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    audio = item["inputs"]["audio"]
    definition: dict = {"locales": [LOCALES.get(lang, lang)]}
    if p.get("model"):
        definition["model"] = p["model"]
    if p.get("diarize"):
        definition["diarization"] = {"enabled": True, "maxSpeakers": int(p.get("max_speakers",
                                                                              8))}
    url = (f"{state['host']}/speechtotext/transcriptions:transcribe"
           f"?api-version={p.get('api_version', '2025-10-15')}")
    res = post_form(url, {"definition": definition}, {"audio": audio},
                    {"Ocp-Apim-Subscription-Key": state["key"]})
    words, turns = [], []
    for ph in res.get("phrases") or []:
        for w in ph.get("words") or []:
            s = w["offsetMilliseconds"] / 1000
            words.append({"start": s, "end": s + w["durationMilliseconds"] / 1000,
                          "word": w["text"]})
        if ph.get("speaker") is not None:
            s = ph["offsetMilliseconds"] / 1000
            turns.append({"start": s, "end": s + ph["durationMilliseconds"] / 1000,
                          "speaker": str(ph["speaker"])})
    text = " ".join(c.get("text", "") for c in res.get("combinedPhrases") or []).strip()
    payload = {"text": text, "language": lang,
               "segments": [{"start": words[0]["start"], "end": words[-1]["end"], "text": text,
                             "words": words}] if words else [],
               "_cost_usd": audio_seconds(audio) / 3600 * float(p.get("usd_per_hour", 0.36))}
    if p.get("diarize"):
        payload["turns"] = turns
    return payload


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model") or "azure-fast-default"}


if __name__ == "__main__":
    serve(load, run, describe)
