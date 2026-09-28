"""A3/A8 worker: Qwen3.5-Omni-Plus (Alibaba Cloud Model Studio, API-only) through the
OpenAI-compatible endpoint: ``POST {base}/chat/completions`` with ``model: qwen3.5-omni-plus``,
``modalities: ["text"]``, ``stream: true`` (required for omni models) and a content part
``{"type": "input_audio", "input_audio": {"data": "data:;base64,…", "format": "wav"}}``
(base64 < 10 MB). alibabacloud.com/help/en/model-studio/qwen-omni read 2026-09-28.
params: model, task, base_url (default the international compatible-mode host), usd_per_call
(price not verified; recorded per call).
"""
from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

from _sdk import api_key, serve
from a8_llm import duration, prompt, to_payload

BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def load(params: dict, lang: str):
    return {"key": api_key("DASHSCOPE_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    task = p.get("task", "delivery")
    audio = item["inputs"]["audio"]
    b64 = base64.b64encode(Path(audio).read_bytes()).decode()
    body = {"model": p.get("model", "qwen3.5-omni-plus"), "modalities": ["text"], "stream": True,
            "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": f"data:;base64,{b64}",
                                                        "format": "wav"}},
                {"type": "text", "text": prompt(task, lang, item.get("meta"))}]}]}
    req = urllib.request.Request(f"{p.get('base_url', BASE)}/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {state['key']}",
                                          "Content-Type": "application/json"})
    reply = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            for ch in chunk.get("choices") or []:
                reply.append((ch.get("delta") or {}).get("content") or "")
    text = "".join(reply)
    return {**to_payload(task, text, duration(audio)), "language": lang,
            "_cost_usd": float(p.get("usd_per_call", 0.0))}


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model"), "task": state["params"].get("task")}


if __name__ == "__main__":
    serve(load, run, describe)
