"""D2/D5 worker: Hume Octave TTS (acting instructions via ``description``).

POST https://api.hume.ai/v0/tts/file, header ``X-Hume-Api-Key``, body {utterances: [{text,
description?, voice?: {name|id, provider: HUME_AI|CUSTOM_VOICE}}], format: {type: wav},
version: "2"} → wav bytes (Hume TTS reference as known 2026; NOT re-verified).
Voice cloning needs a voice created from audio in the Hume platform (pass its id in
``voices``); without a voice, Octave designs one from ``description`` (D5).

params: voices ({lang: {female, male}} voice names/ids), provider (HUME_AI), description
("{emotion}" / "{gender}" filled from the item), version ("2"), usd_per_million_chars (estimate).
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import api_key, serve


def load(params: dict, lang: str):
    return {"key": api_key("HUME_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    utt: dict = {"text": text}
    if p.get("description"):
        utt["description"] = p["description"].format(emotion=dv.emotion_of(item) or "natural",
                                                     gender=dv.gender_of(item) or "female")
    if p.get("voices"):
        utt["voice"] = {"name": dv.pick_voice(p["voices"], state["lang"], item),
                        "provider": p.get("provider", "HUME_AI")}
    raw, _ = dv.post_json("https://api.hume.ai/v0/tts/file",
                          {"utterances": [utt], "format": {"type": "wav"},
                           "version": str(p.get("version", "2"))},
                          headers={"X-Hume-Api-Key": state["key"]}, timeout=300)
    wav = dv.transcode_to_wav(raw, dv.wav_path(out), ".wav")
    usd = float(p.get("usd_per_million_chars", 150.0))
    return {**dv.audio_payload(wav), "_cost_usd": dv.char_cost(text, usd)}


if __name__ == "__main__":
    serve(load, run)
