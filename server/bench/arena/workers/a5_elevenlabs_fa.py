"""A5 worker: ElevenLabs Forced Alignment API. ``POST https://api.elevenlabs.io/v1/
forced-alignment`` (header ``xi-api-key``; multipart ``file`` + ``text``) → ``{characters:
[{text, start, end}], words: [{text, start, end, loss}], loss}`` in seconds; 29 languages incl.
en/hi/ja (docs read 2026-09-28). Billed "at the Speech-to-Text rate": param usd_per_hour
(default 0.22, the Scribe v2 list price read the same day).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, post_form
from a5_common import UNSPACED, project, tokenize

URL = "https://api.elevenlabs.io/v1/forced-alignment"


def load(params: dict, lang: str):
    return {"key": api_key("ELEVENLABS_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    audio, text, lang = item["inputs"]["audio"], item["inputs"]["text"], state["lang"]
    res = post_form(URL, {"text": text}, {"file": audio}, {"xi-api-key": state["key"]})
    src = res.get("characters") if lang in UNSPACED else res.get("words")
    units = [{"start": u["start"], "end": u["end"], "word": u["text"]} for u in src or []]
    return {"words": project(tokenize(text, lang), units), "loss": res.get("loss"),
            "_cost_usd": audio_seconds(audio) / 3600 * float(state["params"].get(
                "usd_per_hour", 0.22))}


if __name__ == "__main__":
    serve(load, run)
