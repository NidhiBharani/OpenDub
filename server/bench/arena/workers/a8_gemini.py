"""A3/A4/A8 worker: Gemini ``generateContent`` with inline audio (≤ 20 MB) and a task prompt
(``a8_llm``). ``POST https://generativelanguage.googleapis.com/v1beta/models/{model}:
generateContent`` (header ``x-goog-api-key``), parts ``[{text}, {inline_data: {mime_type,
data: base64}}]``; reply in ``candidates[0].content.parts[].text``. Audio = 32 tokens/s
(ai.google.dev/gemini-api/docs/audio, read 2026-09-28).

params: model (gemini-3.8-flash, gemini-3.1-pro …), task, temperature, usd_per_m_audio_tokens
(audio input price per 1M tokens; the page lists 2.5 Flash $1.00, Flash-Lite $0.30 — set it for
the pinned model), usd_per_m_out_tokens. Cost = audio tokens + output tokens from usageMetadata.
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from _sdk import api_key, serve
from a4_http import request
from a8_llm import duration, prompt, to_payload

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def load(params: dict, lang: str):
    return {"key": api_key("GEMINI_API_KEY", "GOOGLE_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    task = p.get("task", "delivery")
    audio = item["inputs"]["audio"]
    body = {"contents": [{"parts": [
        {"text": prompt(task, lang, item.get("meta"))},
        {"inline_data": {"mime_type": mimetypes.guess_type(audio)[0] or "audio/wav",
                         "data": base64.b64encode(Path(audio).read_bytes()).decode()}}]}],
        "generationConfig": {"temperature": float(p.get("temperature", 0.0))}}
    if task != "asr":
        body["generationConfig"]["responseMimeType"] = "application/json"
    model = p.get("model", "gemini-3.8-flash")
    res = request("POST", f"{BASE}/{model}:generateContent", json_body=body,
                  headers={"x-goog-api-key": state["key"]})
    parts = res["candidates"][0]["content"].get("parts", [])
    reply = "".join(pt.get("text", "") for pt in parts)
    dur = duration(audio)
    usage = res.get("usageMetadata", {})
    cost = (dur * 32 / 1e6 * float(p.get("usd_per_m_audio_tokens", 1.0))
            + usage.get("candidatesTokenCount", 0) / 1e6 * float(p.get("usd_per_m_out_tokens",
                                                                       2.5)))
    return {**to_payload(task, reply, dur), "language": lang, "_cost_usd": cost}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model"), "task": state["params"].get("task")}


if __name__ == "__main__":
    serve(load, run, describe)
