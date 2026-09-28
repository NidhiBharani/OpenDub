"""A3/A8 worker: OpenAI audio-input chat models (gpt-audio-1.5, gpt-audio, gpt-audio-mini) via
``POST https://api.openai.com/v1/chat/completions`` with ``modalities: ["text"]`` and a user
content part ``{"type": "input_audio", "input_audio": {"data": <base64>, "format": "wav"}}``.
Prices (developers.openai.com/api/docs/models/gpt-audio-1.5, read 2026-09-28): audio input $32
/ 1M tokens, text in $2.50, text out $10; cost from ``usage.prompt_tokens_details.audio_tokens``.
params: model, task, usd_per_m_audio_in, usd_per_m_text_in, usd_per_m_text_out.
"""
from __future__ import annotations

import base64
from pathlib import Path

from _sdk import api_key, serve
from a4_http import request
from a8_llm import duration, prompt, to_payload

URL = "https://api.openai.com/v1/chat/completions"


def load(params: dict, lang: str):
    return {"key": api_key("OPENAI_API_KEY"), "params": params, "lang": lang}


def _wav16k(path: str, tmp: Path) -> str:
    if path.lower().endswith(".wav"):
        return path
    import librosa
    import soundfile as sf

    y, _ = librosa.load(path, sr=16000, mono=True)
    sf.write(str(tmp), y, 16000)
    return str(tmp)


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    task = p.get("task", "delivery")
    wav = _wav16k(item["inputs"]["audio"], out.with_suffix(".in.wav"))
    body = {"model": p.get("model", "gpt-audio-1.5"), "modalities": ["text"], "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt(task, lang, item.get("meta"))},
                {"type": "input_audio", "input_audio": {
                    "data": base64.b64encode(Path(wav).read_bytes()).decode(),
                    "format": "wav"}}]}]}
    res = request("POST", URL, json_body=body,
                  headers={"Authorization": f"Bearer {state['key']}"})
    reply = res["choices"][0]["message"].get("content") or ""
    u = res.get("usage", {})
    audio_in = (u.get("prompt_tokens_details") or {}).get("audio_tokens", 0)
    cost = (audio_in * float(p.get("usd_per_m_audio_in", 32.0))
            + (u.get("prompt_tokens", 0) - audio_in) * float(p.get("usd_per_m_text_in", 2.5))
            + u.get("completion_tokens", 0) * float(p.get("usd_per_m_text_out", 10.0))) / 1e6
    return {**to_payload(task, reply, duration(wav)), "language": lang, "_cost_usd": cost}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model"), "task": state["params"].get("task")}


if __name__ == "__main__":
    serve(load, run, describe)
