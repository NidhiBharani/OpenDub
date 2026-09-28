"""D3/D5 worker: Google Cloud Text-to-Speech, Chirp 3: HD voices (no cloning — the cleanest
non-cloning guarantee). POST https://texttospeech.googleapis.com/v1/text:synthesize?key=<key>
{input: {text}, voice: {languageCode, name}, audioConfig: {audioEncoding: LINEAR16,
sampleRateHertz: 24000, speakingRate?}} → {audioContent: base64 wav}. Voice names follow
``<locale>-Chirp3-HD-<Name>`` (e.g. en-US-Chirp3-HD-Charon); en-US, hi-IN and ja-JP are among
the 57 GA locales per the compute-tiers artifact (not re-verified).

params: voices {lang: {female, male}}, rate_to_target (D3: speakingRate from a first pass),
usd_per_million_chars (30 — Chirp 3 HD list price, not re-verified).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import d_voice as dv
from _sdk import api_key, serve

DEFAULT = {"en": {"female": "en-US-Chirp3-HD-Aoede", "male": "en-US-Chirp3-HD-Charon"},
           "hi": {"female": "hi-IN-Chirp3-HD-Aoede", "male": "hi-IN-Chirp3-HD-Charon"},
           "ja": {"female": "ja-JP-Chirp3-HD-Aoede", "male": "ja-JP-Chirp3-HD-Charon"}}


def load(params: dict, lang: str):
    return {"key": api_key("GOOGLE_API_KEY", "GEMINI_API_KEY"), "params": params, "lang": lang}


def _synth(state: dict, voice: str, text: str, wav: Path, rate: float | None = None) -> None:
    cfg = {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}
    if rate:
        cfg["speakingRate"] = rate
    body = {"input": {"text": text},
            "voice": {"languageCode": "-".join(voice.split("-")[:2]), "name": voice},
            "audioConfig": cfg}
    raw, _ = dv.post_json("https://texttospeech.googleapis.com/v1/text:synthesize?key="
                          + state["key"], body, timeout=300)
    wav.write_bytes(base64.b64decode(json.loads(raw)["audioContent"]))


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    voice = dv.pick_voice(p.get("voices", DEFAULT), state["lang"], item)
    wav = dv.wav_path(out)
    _synth(state, voice, text, wav)
    calls, rate = 1, None
    tgt = dv.target_s(item)
    if tgt and p.get("rate_to_target"):
        rate = max(0.25, min(2.0, dv.duration_s(wav) / tgt))
        if abs(rate - 1) > 0.03:
            _synth(state, voice, text, wav, rate)
            calls = 2
    usd = float(p.get("usd_per_million_chars", 30.0))
    return {**dv.audio_payload(wav, voice=voice, rate=rate),
            "_cost_usd": dv.char_cost(text, usd) * calls}


if __name__ == "__main__":
    serve(load, run)
