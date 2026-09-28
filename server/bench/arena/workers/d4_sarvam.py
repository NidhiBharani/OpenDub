"""D4/D5 worker: Sarvam AI Bulbul (Indian languages, stock speakers; no cloning).

POST https://api.sarvam.ai/text-to-speech, header ``api-subscription-key``, body
{text, target_language_code: "hi-IN", speaker, model: "bulbul:v3", speech_sample_rate: 24000}
→ {audios: [base64 wav]} (Sarvam API reference as known 2026; NOT re-verified — Bulbul v4 was
announced 2026-07-30, set ``model`` when its id is public).

params: model, speakers ({lang: {female, male}}; e.g. anushka / abhilash for bulbul:v2),
pace (1.0), usd_per_10k_chars (estimate, set from the Sarvam price list).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import d_voice as dv
from _sdk import api_key, serve

LANGS = {"hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
         "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN", "pa": "pa-IN", "or": "od-IN",
         "en": "en-IN"}


def load(params: dict, lang: str):
    dv.require_lang(lang, LANGS, "Sarvam Bulbul")
    return {"key": api_key("SARVAM_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    speaker = dv.pick_voice(p.get("speakers", {"*": {"female": "anushka", "male": "abhilash"}}),
                            state["lang"], item)
    body = {"text": text, "target_language_code": LANGS[state["lang"]], "speaker": speaker,
            "model": p.get("model", "bulbul:v3"), "pace": float(p.get("pace", 1.0)),
            "speech_sample_rate": int(p.get("sample_rate", 24000))}
    raw, _ = dv.post_json("https://api.sarvam.ai/text-to-speech", body,
                          headers={"api-subscription-key": state["key"]}, timeout=300)
    audio = base64.b64decode(json.loads(raw)["audios"][0])
    wav = dv.transcode_to_wav(audio, dv.wav_path(out), ".wav")
    cost = round(len(text) / 10000 * float(p.get("usd_per_10k_chars", 0.18)), 6)
    return {**dv.audio_payload(wav, speaker=speaker), "_cost_usd": cost}


if __name__ == "__main__":
    serve(load, run)
