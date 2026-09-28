"""A4 worker: Sarvam AI speech-to-text (Indian languages only).

``POST https://api.sarvam.ai/speech-to-text`` (header ``api-subscription-key``; multipart
``file``, ``model`` = saaras:v3, ``language_code`` = hi-IN, ``mode`` = transcribe). REST accepts
clips under 30 s (longer ones → Unsupported; the batch API is not wired). Timestamps are
phrase-level only. docs.sarvam.ai read 2026-09-28; ₹30/h ≈ $0.34/h at ₹88/$ (param).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, api_key, serve
from a4_http import audio_seconds, post_form

URL = "https://api.sarvam.ai/speech-to-text"
LOCALES = {"hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
           "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN", "pa": "pa-IN", "od": "od-IN",
           "en": "en-IN"}


def load(params: dict, lang: str):
    return {"key": api_key("SARVAM_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang, audio = state["params"], state["lang"], item["inputs"]["audio"]
    if lang not in LOCALES:
        raise Unsupported(f"Sarvam does not cover {lang!r}")
    secs = audio_seconds(audio)
    if secs >= 30:
        raise Unsupported("REST endpoint takes clips < 30 s")
    res = post_form(URL, {"model": p.get("model", "saaras:v3"), "language_code": LOCALES[lang],
                          "mode": p.get("mode", "transcribe")},
                    {"file": audio}, {"api-subscription-key": state["key"]})
    return {"text": (res.get("transcript") or "").strip(), "segments": [], "language": lang,
            "_cost_usd": secs / 3600 * float(p.get("usd_per_hour", 0.34))}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model", "saaras:v3")}


if __name__ == "__main__":
    serve(load, run, describe)
