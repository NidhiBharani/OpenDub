"""Shared prompts and JSON parsing for audio-LLM workers (not a worker): Gemini, OpenAI audio,
Qwen3-Omni (local / DashScope), Audio Flamingo. One ``task`` per job:

- ``delivery`` (A8): emotion from the pack's classes + arousal/valence 0–1 + nonverbal events +
  a one-line delivery note;
- ``regions`` (A3): speech / music / singing segments with times;
- ``asr`` (A4): verbatim transcript in the forced language.

The model's labels are constrained in the prompt; the judges map synonyms and score anything
else as wrong. Specialist labels are never put into the prompt as hints (G4 rule).
"""
from __future__ import annotations

import json
import re

LANG_NAMES = {"en": "English", "hi": "Hindi", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
              "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
              "ta": "Tamil", "te": "Telugu", "bn": "Bengali"}
DEFAULT_EMOTIONS = ["neutral", "happy", "angry", "sad", "fear", "disgust", "surprise"]
NONVERBAL = ["laughter", "crying", "sigh", "scream", "breath", "cough", "gasp", "sniff",
             "groan", "yawn"]


def prompt(task: str, lang: str, meta: dict | None = None) -> str:
    meta = meta or {}
    lname = LANG_NAMES.get(lang, lang)
    if task == "asr":
        return (f"Transcribe this {lname} audio verbatim in {lname}, in its native script. "
                "Output only the transcript, no commentary, no timestamps.")
    if task == "regions":
        return ("List every region of this audio that contains speech, music, or singing. "
                "Regions may overlap (e.g. speech over music). Answer with JSON only: "
                '{"segments": [{"start": <seconds>, "end": <seconds>, "label": '
                '"speech"|"music"|"singing"}]}.')
    classes = meta.get("classes") or DEFAULT_EMOTIONS
    return (f"You are a dubbing director listening to one line of {lname} dialogue. "
            f"Classify the speaker's emotion as exactly one of: {', '.join(classes)}. "
            "Rate arousal and valence from 0 (low/negative) to 1 (high/positive). List any "
            f"nonverbal vocalisations ({', '.join(NONVERBAL)}) with start/end seconds. Write a "
            "one-sentence delivery note for a voice actor. Answer with JSON only: "
            '{"emotion": "...", "arousal": 0.0, "valence": 0.0, "events": [{"start": 0.0, '
            '"end": 0.0, "label": "..."}], "note": "..."}')


def parse_json(text: str) -> dict:
    """First JSON object in a model reply (tolerates code fences and prose)."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start = t.find("{")
    while start >= 0:
        depth = 0
        for k in range(start, len(t)):
            depth += {"{": 1, "}": -1}.get(t[k], 0)
            if depth == 0:
                try:
                    return json.loads(t[start:k + 1])
                except json.JSONDecodeError:
                    break
        start = t.find("{", start + 1)
    return {}


def to_payload(task: str, reply: str, duration_s: float | None = None) -> dict:
    if task == "asr":
        return {"text": reply.strip(), "segments": []}
    d = parse_json(reply)
    if task == "regions":
        segs = [{"start": float(s["start"]), "end": float(s["end"]),
                 "label": str(s.get("label", "")).lower(), "score": 1.0}
                for s in d.get("segments") or [] if "start" in s and "end" in s]
        return {"segments": segs, "raw": reply}
    segs = []
    if d.get("emotion"):
        segs.append({"start": 0.0, "end": float(duration_s or 0.0),
                     "label": str(d["emotion"]).lower(), "score": 1.0})
    for e in d.get("events") or []:
        if "start" in e and "end" in e:
            segs.append({"start": float(e["start"]), "end": float(e["end"]),
                         "label": str(e.get("label", "")).lower(), "score": 1.0})
    dims = {k: float(d[k]) for k in ("arousal", "valence", "dominance")
            if isinstance(d.get(k), (int, float))}
    return {"segments": segs, "emotion": d.get("emotion"), "dims": dims,
            "note": d.get("note", ""), "raw": reply}


def duration(path: str) -> float:
    from a4_http import audio_seconds

    return audio_seconds(path)
