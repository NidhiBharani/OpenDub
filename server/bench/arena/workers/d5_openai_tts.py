"""D2/D5 worker: OpenAI TTS (gpt-4o-mini-tts with ``instructions``; tts-1-hd). Stock voices only.

POST https://api.openai.com/v1/audio/speech {model, input, voice, instructions?,
response_format: wav} → wav bytes (24 kHz). The model reads the language from the text; there is
no language parameter, so the declared list (multilingual, Whisper languages) is taken on trust.

params: model, voices ({lang: {female, male}} or one name: alloy, ash, ballad, coral, echo, fable,
nova, onyx, sage, shimmer, verse …), instructions ("{emotion}" filled from the item; D2),
usd_per_million_chars (15 for tts-1-hd / estimate ~12 for gpt-4o-mini-tts audio tokens;
list prices as known 2026, not re-verified).
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import api_key, serve


def load(params: dict, lang: str):
    return {"key": api_key("OPENAI_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    model = p.get("model", "gpt-4o-mini-tts")
    body = {"model": model, "input": text, "response_format": "wav",
            "voice": dv.pick_voice(p.get("voices", {"*": {"female": "coral", "male": "ash"}}),
                                   state["lang"], item)}
    if p.get("instructions") and model.startswith("gpt-"):
        body["instructions"] = p["instructions"].format(emotion=dv.emotion_of(item) or "natural",
                                                        language=dv.LANG_NAMES[state["lang"]])
    raw, _ = dv.post_json("https://api.openai.com/v1/audio/speech", body,
                          headers={"Authorization": f"Bearer {state['key']}"}, timeout=300)
    wav = dv.transcode_to_wav(raw, dv.wav_path(out), ".wav")
    usd = float(p.get("usd_per_million_chars", 30.0 if model == "tts-1-hd" else 12.0))
    return {**dv.audio_payload(wav, voice=body["voice"]), "_cost_usd": dv.char_cost(text, usd)}


if __name__ == "__main__":
    serve(load, run)
