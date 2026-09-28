"""D1/D2/D4/D5/D6/D7 worker: ElevenLabs API (stdlib HTTP), mirroring app/providers/tts/elevenlabs.py.

modes:
  clone   Instant Voice Clone from inputs.ref_audio (POST /v1/voices/add, one per distinct
          reference, deleted when the job ends) then POST /v1/text-to-speech/{voice_id}
  library a stock/Voice-Library voice per language: params.voices {lang: {female, male}} (D5)
  sts     Voice Changer: POST /v1/speech-to-speech/{voice_id} with inputs.audio, the voice
          cloned from inputs.ref_audio (D6)

params: model (eleven_v3 | eleven_multilingual_v2 | eleven_flash_v2_5 |
        eleven_multilingual_sts_v2 for sts), stability (0.4), similarity (0.8),
        output_format (pcm_24000 → wav at 24 kHz), emotion_tags (v3: prefix "[<emotion>]"),
        usd_per_1k_chars / usd_per_minute (list prices used for _cost_usd).

Pricing (elevenlabs.io/pricing/api, read 2026-09-28, NOT re-verified per plan tier): about
$0.10 per 1k characters for multilingual v2 / v3 and $0.05 for Flash on the API usage tiers;
speech-to-speech is billed like TTS minutes. Treat _cost_usd as an estimate.
``language_code`` (ISO 639-1) is sent to force the language (v2.5+/v3 honour it). ``seed`` is
sent for repeatability.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import sys
from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, api_key, serve

API = "https://api.elevenlabs.io"


def load(params: dict, lang: str):
    state = {"key": api_key("ELEVENLABS_API_KEY"), "params": params, "lang": lang,
             "voices": {}, "created": []}

    def cleanup() -> None:
        for vid in state["created"]:
            try:
                dv.http("DELETE", f"{API}/v1/voices/{vid}", headers={"xi-api-key": state["key"]},
                        retries=1)
            except Exception as exc:  # noqa: BLE001 - best effort; the voice slot is reusable anyway
                print(f'cleanup failed: {exc!r}', file=sys.stderr)

    atexit.register(cleanup)
    return state


def _clone(state: dict, ref: str) -> str:
    data = Path(ref).read_bytes()
    digest = hashlib.sha1(data).hexdigest()
    if digest not in state["voices"]:
        body, ctype = dv.multipart({"name": f"opendub-arena-{digest[:8]}"},
                                   {"files": (Path(ref).name, data, "audio/wav")})
        raw, _ = dv.http("POST", f"{API}/v1/voices/add", body=body,
                         headers={"xi-api-key": state["key"], "Content-Type": ctype})
        vid = json.loads(raw)["voice_id"]
        state["voices"][digest] = vid
        state["created"].append(vid)
    return state["voices"][digest]


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    mode = p.get("mode", "clone")
    model = p.get("model", "eleven_multilingual_v2")
    fmt = p.get("output_format", "pcm_24000")
    wav = dv.wav_path(out)
    headers = {"xi-api-key": state["key"]}
    if mode == "sts":
        src = (item.get("inputs") or {}).get("audio")
        if not src:
            raise Unsupported("voice changer item has no inputs.audio")
        ref, _ = dv.ref_of(item)
        vid = _clone(state, ref)
        body, ctype = dv.multipart({"model_id": model or "eleven_multilingual_sts_v2",
                                    "seed": dv.item_seed(item, p) % 2**31},
                                   {"audio": (Path(src).name, Path(src).read_bytes(),
                                              "audio/wav")})
        raw, _ = dv.http("POST", f"{API}/v1/speech-to-speech/{vid}?output_format={fmt}",
                         body=body, headers={**headers, "Content-Type": ctype}, timeout=300)
        minutes = dv.duration_s(src) / 60
        cost = round(minutes * float(p.get("usd_per_minute", 0.10)), 6)
    else:
        text = dv.text_of(item)
        if model == "eleven_v3" and p.get("emotion_tags") and dv.emotion_of(item):
            text = f"[{dv.emotion_of(item).lower()}] {text}"
        if mode == "library":
            vid = dv.pick_voice(p.get("voices"), lang, item)
        else:
            ref, _ = dv.ref_of(item)
            vid = _clone(state, ref)
        settings = {"stability": float(p.get("stability", 0.4))}
        if model != "eleven_v3":
            settings["similarity_boost"] = float(p.get("similarity", 0.8))
        body = {"text": text, "model_id": model, "voice_settings": settings,
                "language_code": lang, "seed": dv.item_seed(item, p) % 2**31}
        raw, _ = dv.post_json(f"{API}/v1/text-to-speech/{vid}?output_format={fmt}", body,
                              headers=headers, timeout=300)
        rate = 0.05 if "flash" in model or "turbo" in model else 0.10
        cost = dv.char_cost(text, 1000 * float(p.get("usd_per_1k_chars", rate)))
    if fmt.startswith("pcm_"):
        dv.pcm16_to_wav(raw, int(fmt.split("_")[1]), wav)
    else:
        dv.transcode_to_wav(raw, wav)
    return {**dv.audio_payload(wav, mode=mode, model=model), "_cost_usd": cost}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model"), "mode": state["params"].get("mode", "clone")}


if __name__ == "__main__":
    serve(load, run, describe)
