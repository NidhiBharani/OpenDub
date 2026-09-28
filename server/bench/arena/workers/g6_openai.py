"""G6 worker (also G3/G4 LLM-judge candidates): OpenAI-compatible audio chat models.

Covers OpenAI GPT audio (``gpt-audio-1.5``: audio in, text out; no video, so the reviewer
cannot see lip sync and off_sync counts as a miss) and Alibaba Qwen3.5-Omni through DashScope's
OpenAI-compatible endpoint (``base_url``
https://dashscope-intl.aliyuncs.com/compatible-mode/v1, ``api_key_env: DASHSCOPE_API_KEY``,
``video: true``, ``data_url: true``, ``stream: true``).

Protocol and prompts: ``g6_rubric``. params: model, base_url, api_key_env, task, protocol,
video, data_url (send media as ``data:`` URLs, DashScope style), stream, prices (USD per 1M
tokens: ``text_in``, ``audio_in``, ``out``).

Pricing checked 2026-09-28 (developers.openai.com model page): gpt-audio-1.5 audio input $32 /
1M, text input $2.50 / 1M, text output $10 / 1M. Qwen3.5-Omni prices are not pinned here
(verify on Alibaba Model Studio before a paid run).

Env: g6_api (openai SDK).
"""
from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import g6_rubric as R
from _sdk import api_key, serve


def load(params: dict, lang: str):
    from openai import OpenAI

    key = api_key(params.get("api_key_env", "OPENAI_API_KEY"))
    kwargs = {"api_key": key}
    if params.get("base_url"):
        kwargs["base_url"] = params["base_url"]
    return {"client": OpenAI(**kwargs), "params": params, "lang": lang,
            "tmp": Path(tempfile.mkdtemp(prefix="opendub-oai-")),
            "model": params.get("model", "gpt-audio-1.5")}


def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _content(state: dict, parts: list, prompt: str) -> list:
    data_url = bool(state["params"].get("data_url"))
    content = []
    for kind, val in parts:
        if kind == "text":
            content.append({"type": "text", "text": val})
        elif kind == "audio":
            wav = R.compact_audio(val, state["tmp"])
            data = f"data:;base64,{_b64(wav)}" if data_url else _b64(wav)
            content.append({"type": "input_audio", "input_audio": {"data": data,
                                                                   "format": "wav"}})
        elif kind == "video":
            content.append({"type": "video_url",
                            "video_url": {"url": f"data:{R.mime_of(val)};base64,{_b64(val)}"}})
    content.append({"type": "text", "text": prompt})
    return content


def _retryable(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)
    return code in (408, 409, 429, 500, 502, 503, 504) or "Timeout" in type(exc).__name__ \
        or "Connection" in type(exc).__name__


def ask(state: dict, parts: list, prompt: str) -> tuple[str, dict]:
    p = state["params"]
    messages = [{"role": "system", "content": R.SYSTEM},
                {"role": "user", "content": _content(state, parts, prompt)}]
    kwargs = {"model": state["model"], "messages": messages, "modalities": ["text"]}
    if "temperature" in p:
        kwargs["temperature"] = float(p["temperature"])

    def call():
        if not p.get("stream"):
            return state["client"].chat.completions.create(**kwargs), None
        chunks, usage = [], None
        for ev in state["client"].chat.completions.create(
                stream=True, stream_options={"include_usage": True}, **kwargs):
            if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                chunks.append(ev.choices[0].delta.content)
            if getattr(ev, "usage", None):
                usage = ev.usage
        return "".join(chunks), usage

    resp, usage = R.with_retries(call, retryable=_retryable)
    if usage is None:
        text, usage = resp.choices[0].message.content or "", resp.usage
    else:
        text = resp
    details = getattr(usage, "prompt_tokens_details", None)
    audio_in = getattr(details, "audio_tokens", 0) or 0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    return text, {"audio_in": audio_in, "text_in": max(0, prompt_tokens - audio_in),
                  "out": getattr(usage, "completion_tokens", 0) or 0}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    return R.run_item(ask, state, item, state["lang"], task=p.get("task", "review"),
                      protocol=p.get("protocol", "holistic"),
                      caps={"video": bool(p.get("video"))}, prices=p.get("prices"))


def describe(state: dict) -> dict:
    import openai

    p = state["params"]
    return {"model": state["model"], "base_url": p.get("base_url") or "api.openai.com",
            "task": p.get("task", "review"), "protocol": p.get("protocol", "holistic"),
            "openai": openai.__version__, "pid": os.getpid()}


if __name__ == "__main__":
    serve(load, run, describe)
