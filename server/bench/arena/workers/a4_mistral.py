"""A4 worker: Mistral Voxtral transcription API.

``POST https://api.mistral.ai/v1/audio/transcriptions`` (``Authorization: Bearer``; multipart
``file``, ``model`` = voxtral-mini-2602, ``language``). docs.mistral.ai offline transcription page
(read 2026-09-28) notes ``timestamp_granularities`` cannot be combined with ``language``; the
arena forces the language, so this worker returns text only. $0.003/min batch (Feb 2026).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, post_form

URL = "https://api.mistral.ai/v1/audio/transcriptions"


def load(params: dict, lang: str):
    return {"key": api_key("MISTRAL_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, audio = state["params"], item["inputs"]["audio"]
    res = post_form(URL, {"model": p.get("model", "voxtral-mini-2602"),
                          "language": state["lang"] or None},
                    {"file": audio}, {"Authorization": f"Bearer {state['key']}"})
    return {"text": (res.get("text") or "").strip(), "segments": [], "language": state["lang"],
            "_cost_usd": audio_seconds(audio) / 60 * float(p.get("usd_per_min", 0.003))}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model", "voxtral-mini-2602")}


if __name__ == "__main__":
    serve(load, run, describe)
