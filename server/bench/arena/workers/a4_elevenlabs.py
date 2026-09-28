"""A4/A6/A3/A8 worker: ElevenLabs Speech-to-Text (Scribe).

``POST https://api.elevenlabs.io/v1/speech-to-text`` (multipart; header ``xi-api-key``) with
``model_id`` (scribe_v2 | scribe_v1), ``language_code`` (forced from the pack),
``timestamps_granularity=word``, ``diarize``, ``tag_audio_events``. Response ``{language_code,
text, words: [{text, start, end, type: word|spacing|audio_event, speaker_id, logprob}]}``
(seconds). Checked 2026-09-28 against elevenlabs.io/docs/api-reference/speech-to-text/convert.

params: model_id, diarize (→ ``turns`` for A6), tag_audio_events, task (asr | events: A3/A8
payload of ``segments`` = speech spans from words + audio-event spans), usd_per_hour.
Pricing: Scribe v2 list price $0.22 / audio hour on elevenlabs.io/pricing/api (read 2026-09-28;
lower than earlier Scribe prices, so it is a param — correct it if the page changes).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, post_form, words_to_turns

URL = "https://api.elevenlabs.io/v1/speech-to-text"


def load(params: dict, lang: str):
    return {"key": api_key("ELEVENLABS_API_KEY"), "params": params, "lang": lang}


def event_label(text: str) -> str:
    t = text.strip("()[] ").lower()
    if "sing" in t or "song" in t:
        return "singing"
    if "music" in t:
        return "music"
    if "laugh" in t:
        return "laughter"
    if "cry" in t or "sob" in t:
        return "crying"
    if "sigh" in t:
        return "sigh"
    if "scream" in t:
        return "scream"
    return t or "event"


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    audio = item["inputs"]["audio"]
    task = p.get("task", "asr")
    fields = {"model_id": p.get("model_id", "scribe_v2"), "language_code": state["lang"] or None,
              "timestamps_granularity": "word", "diarize": bool(p.get("diarize", False)),
              "tag_audio_events": bool(p.get("tag_audio_events", task == "events"))}
    res = post_form(URL, fields, {"file": audio}, {"xi-api-key": state["key"]})
    words = res.get("words") or []
    cost = audio_seconds(audio) / 3600 * float(p.get("usd_per_hour", 0.22))
    spoken = [{"start": w["start"], "end": w["end"], "word": w["text"],
               "speaker": w.get("speaker_id")} for w in words
              if w.get("type", "word") == "word" and w.get("start") is not None]
    payload = {"text": (res.get("text") or "").strip(),
               "segments": [{"start": spoken[0]["start"], "end": spoken[-1]["end"],
                             "text": (res.get("text") or "").strip(),
                             "words": [{k: w[k] for k in ("start", "end", "word")}
                                       for w in spoken]}] if spoken else [],
               "language": state["lang"], "detected_language": res.get("language_code"),
               "_cost_usd": cost}
    if p.get("diarize"):
        payload["turns"] = words_to_turns(spoken)
    if task == "events":
        segs = []
        for w in words:
            if w.get("start") is None:
                continue
            if w.get("type") == "audio_event":
                segs.append({"start": w["start"], "end": w["end"],
                             "label": event_label(w.get("text", "")), "score": 1.0})
            elif w.get("type", "word") == "word":
                if segs and segs[-1]["label"] == "speech" and w["start"] - segs[-1]["end"] < 0.5:
                    segs[-1]["end"] = w["end"]
                else:
                    segs.append({"start": w["start"], "end": w["end"], "label": "speech",
                                 "score": 1.0})
        payload["segments"] = segs
    return payload


def describe(state: dict) -> dict:
    return {"model_id": state["params"].get("model_id", "scribe_v2"), "api": URL}


if __name__ == "__main__":
    serve(load, run, describe)
