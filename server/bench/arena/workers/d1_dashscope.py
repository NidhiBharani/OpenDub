"""D1/D5/D7 worker: Alibaba Qwen TTS APIs.

backend openrouter (verified on openrouter.ai, 2026-09-28): POST
  https://openrouter.ai/api/v1/audio/speech {model: "qwen/qwen-audio-3.0-tts-plus", input,
  voice (longanlingxin | longanlufeng), response_format: pcm} → raw audio bytes. Preset voices
  only (no cloning) → a D5 candidate.
backend dashscope (NOT re-verified): POST
  https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
  {model: qwen3-tts-flash, input: {text, voice, language_type}} → output.audio.url (wav).
  Cloning via voice enrollment (``/api/v1/services/audio/tts/customization``, model
  qwen-voice-enrollment, action create, audio data URL) → a voice id for the VC TTS model.

params: backend, model, voice / voices ({lang: {female, male}}), mode (preset | clone),
enroll_target_model (DashScope VC model id), sample_rate (24000 for pcm),
usd_per_million_chars (estimate; set from the console price).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import d_voice as dv
from _sdk import api_key, serve

DASH = "https://dashscope-intl.aliyuncs.com/api/v1"


def load(params: dict, lang: str):
    backend = params.get("backend", "openrouter")
    key = api_key("OPENROUTER_API_KEY") if backend == "openrouter" else api_key("DASHSCOPE_API_KEY")
    return {"key": key, "backend": backend, "params": params, "lang": lang, "voices": {}}


def _enroll(state: dict, ref: str) -> str:
    data = Path(ref).read_bytes()
    digest = hashlib.sha1(data).hexdigest()
    if digest not in state["voices"]:
        body = {"model": "qwen-voice-enrollment",
                "input": {"action": "create",
                          "target_model": state["params"]["enroll_target_model"],
                          "preferred_name": f"odarena{digest[:8]}",
                          "audio": {"data": f"data:audio/wav;base64,{dv.b64file(ref)}"}}}
        raw, _ = dv.post_json(f"{DASH}/services/audio/tts/customization", body,
                              headers={"Authorization": f"Bearer {state['key']}"})
        state["voices"][digest] = json.loads(raw)["output"]["voice"]
    return state["voices"][digest]


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    wav = dv.wav_path(out)
    if state["backend"] == "openrouter":
        voice = dv.pick_voice(p.get("voices", p.get("voice", "longanlingxin")), state["lang"], item)
        raw, _ = dv.post_json("https://openrouter.ai/api/v1/audio/speech",
                              {"model": p.get("model", "qwen/qwen-audio-3.0-tts-plus"),
                               "input": text, "voice": voice, "response_format": "pcm"},
                              headers={"Authorization": f"Bearer {state['key']}"}, timeout=300)
        if raw[:4] == b"RIFF":
            wav.write_bytes(raw)
        else:
            dv.pcm16_to_wav(raw, int(p.get("sample_rate", 24000)), wav)
    else:
        if p.get("mode", "preset") == "clone":
            ref, _ = dv.ref_of(item)
            voice = _enroll(state, ref)
        else:
            voice = dv.pick_voice(p.get("voices", p.get("voice", "Cherry")), state["lang"], item)
        body = {"model": p.get("model", "qwen3-tts-flash"),
                "input": {"text": text, "voice": voice,
                          "language_type": dv.LANG_NAMES.get(state["lang"], "Auto")}}
        raw, _ = dv.post_json(f"{DASH}/services/aigc/multimodal-generation/generation", body,
                              headers={"Authorization": f"Bearer {state['key']}"}, timeout=300)
        url = json.loads(raw)["output"]["audio"]["url"]
        audio, _ = dv.http("GET", url)
        dv.transcode_to_wav(audio, wav, ".wav")
    rate = float(p.get("usd_per_million_chars", 0.0))
    return {**dv.audio_payload(wav), "_cost_usd": dv.char_cost(text, rate)}


if __name__ == "__main__":
    serve(load, run)
