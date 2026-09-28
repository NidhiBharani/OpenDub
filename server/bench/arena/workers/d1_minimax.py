"""D1/D2/D4/D5 worker: MiniMax Speech (speech-2.8-hd / speech-2.8-turbo) over the international API.

T2A (verified against platform.minimax.io/docs/api-reference/speech-t2a-http, 2026-09-28):
POST https://api.minimax.io/v1/t2a_v2, Bearer key, body {model, text, stream: false,
language_boost, voice_setting: {voice_id, speed, vol, pitch, emotion?}, audio_setting:
{format: wav, sample_rate: 32000, channel: 1}, output_format: hex} → data.audio (hex).
Voice clone (not re-verified): POST /v1/files/upload (multipart purpose=voice_clone, file) →
file.file_id; POST /v1/voice_clone {file_id, voice_id}. Cloned voices expire unless used; the
clone fee is charged on first synthesis.

params: model, mode (clone | preset), voices (preset {lang: {female, male}}), emotion (use
inputs.emotion when it is one of happy/sad/angry/fearful/disgusted/surprised/calm/whisper),
sample_rate (32000), usd_per_million_chars (default 100 hd / 60 turbo — list prices of the
speech-02 generation, NOT re-verified for 2.8), base_url.
"""
from __future__ import annotations

import binascii
import hashlib
import json
from pathlib import Path

import d_voice as dv
from _sdk import api_key, serve

BOOST = {"en": "English", "hi": "Hindi", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
         "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "it": "Italian",
         "ru": "Russian", "ar": "Arabic"}
EMOTIONS = {"happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm", "fluent",
            "whisper"}


def load(params: dict, lang: str):
    dv.require_lang(lang, BOOST, "MiniMax language_boost")
    return {"key": api_key("MINIMAX_API_KEY"), "params": params, "lang": lang, "voices": {},
            "base": params.get("base_url", "https://api.minimax.io").rstrip("/")}


def _clone(state: dict, ref: str) -> str:
    data = Path(ref).read_bytes()
    digest = hashlib.sha1(data).hexdigest()
    if digest in state["voices"]:
        return state["voices"][digest]
    h = {"Authorization": f"Bearer {state['key']}"}
    body, ctype = dv.multipart({"purpose": "voice_clone"},
                               {"file": (Path(ref).name, data, "audio/wav")})
    raw, _ = dv.http("POST", f"{state['base']}/v1/files/upload", body=body,
                     headers={**h, "Content-Type": ctype})
    file_id = json.loads(raw)["file"]["file_id"]
    voice_id = f"OpenDubArena{digest[:12]}"
    dv.post_json(f"{state['base']}/v1/voice_clone", {"file_id": file_id, "voice_id": voice_id},
                 headers=h)
    state["voices"][digest] = voice_id
    return voice_id


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    model = p.get("model", "speech-2.8-hd")
    if p.get("mode", "clone") == "clone":
        ref, _ = dv.ref_of(item)
        voice = _clone(state, ref)
    else:
        voice = dv.pick_voice(p.get("voices"), state["lang"], item)
    vs = {"voice_id": voice, "speed": 1.0, "vol": 1.0, "pitch": 0}
    emo = dv.emotion_of(item).lower()
    if p.get("emotion") and emo in EMOTIONS:
        vs["emotion"] = emo
    sr = int(p.get("sample_rate", 32000))
    body = {"model": model, "text": text, "stream": False, "language_boost": BOOST[state["lang"]],
            "voice_setting": vs, "audio_setting": {"format": "wav", "sample_rate": sr,
                                                   "channel": 1},
            "output_format": "hex"}
    raw, _ = dv.post_json(f"{state['base']}/v1/t2a_v2", body,
                          headers={"Authorization": f"Bearer {state['key']}"}, timeout=300)
    resp = json.loads(raw)
    if (resp.get("base_resp") or {}).get("status_code", 0) != 0:
        raise RuntimeError(f"MiniMax error: {resp.get('base_resp')}")
    wav = dv.transcode_to_wav(binascii.unhexlify(resp["data"]["audio"]), dv.wav_path(out), ".wav")
    rate = float(p.get("usd_per_million_chars", 60.0 if "turbo" in model else 100.0))
    return {**dv.audio_payload(wav, model=model), "_cost_usd": dv.char_cost(text, rate)}


if __name__ == "__main__":
    serve(load, run)
