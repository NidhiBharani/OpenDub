"""A4/A6 worker: Deepgram pre-recorded ``POST https://api.deepgram.com/v1/listen`` (header
``Authorization: Token <key>``, raw audio body) with ``model`` (nova-3), ``language`` (forced),
``smart_format``, ``diarize``. Words at ``results.channels[0].alternatives[0].words[{word,
start, end, confidence, punctuated_word, speaker?}]`` in seconds.
Pricing (deepgram.com/pricing read 2026-09-28): Nova-3 pre-recorded $0.0077/min pay-as-you-go
(third-party sources quote $0.0043–0.0052), diarization add-on ~$0.002/min — params.
"""
from __future__ import annotations

import mimetypes
import urllib.parse
from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, request, words_to_turns


def load(params: dict, lang: str):
    return {"key": api_key("DEEPGRAM_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    audio = item["inputs"]["audio"]
    q = {"model": p.get("model", "nova-3"), "language": lang, "smart_format": "true",
         "punctuate": "true"}
    if p.get("diarize"):
        q["diarize"] = "true"
    url = "https://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(q)
    res = request("POST", url, data=Path(audio).read_bytes(), headers={
        "Authorization": f"Token {state['key']}",
        "Content-Type": mimetypes.guess_type(audio)[0] or "audio/wav"})
    alt = res["results"]["channels"][0]["alternatives"][0]
    words = [{"start": w["start"], "end": w["end"],
              "word": w.get("punctuated_word") or w["word"], "speaker": w.get("speaker")}
             for w in alt.get("words") or []]
    text = (alt.get("transcript") or "").strip()
    rate = float(p.get("usd_per_min", 0.0077)) + (float(p.get("diarize_usd_per_min", 0.002))
                                                  if p.get("diarize") else 0.0)
    payload = {"text": text, "language": lang,
               "segments": [{"start": words[0]["start"], "end": words[-1]["end"], "text": text,
                             "words": [{k: w[k] for k in ("start", "end", "word")}
                                       for w in words]}] if words else [],
               "_cost_usd": audio_seconds(audio) / 60 * rate}
    if p.get("diarize"):
        payload["turns"] = words_to_turns(words)
    return payload


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model", "nova-3")}


if __name__ == "__main__":
    serve(load, run, describe)
