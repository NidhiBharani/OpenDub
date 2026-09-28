"""C2 worker (also C1/C3/C4/C5/C6 via ``params.task``): Gemini through the Gemini API
(google-genai SDK).

params: model (API model id, pinned in the candidate), thinking_level (optional, e.g. low|high),
max_tokens, task, prompt (see c_common), price ([USD per 1M input, per 1M output]; the candidate
records the list price and date; long-context tiers are not modelled — dub lines are short).

Thinking tokens bill as output: ``usage.output_tokens`` = candidates + thoughts tokens. Needs
GEMINI_API_KEY (or GOOGLE_API_KEY); runs in the ``c2_api`` env (SDK imported in ``load``).
"""
from __future__ import annotations

from pathlib import Path

import c_common
from _sdk import api_key, serve


def load(params: dict, lang: str):
    from google import genai

    c_common.check_task_lang(params, lang)
    client = genai.Client(api_key=api_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    return {"client": client, "params": params, "lang": lang}


def _complete(state: dict):
    from google.genai import types

    p = state["params"]

    def complete(messages, *, temperature=None, json_mode=False, max_tokens=2048):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [types.Content(role="user" if m["role"] == "user" else "model",
                                  parts=[types.Part(text=m["content"])])
                    for m in messages if m["role"] != "system"]
        cfg: dict = {"max_output_tokens": max(int(max_tokens), 16000)}
        if system:
            cfg["system_instruction"] = system
        if temperature is not None:
            cfg["temperature"] = temperature
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if p.get("thinking_level"):
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=p["thinking_level"])
        resp = state["client"].models.generate_content(
            model=p["model"], contents=contents, config=types.GenerateContentConfig(**cfg))
        um = resp.usage_metadata
        out_tokens = (getattr(um, "candidates_token_count", 0) or 0) + \
                     (getattr(um, "thoughts_token_count", 0) or 0)
        return resp.text or "", {"input_tokens": getattr(um, "prompt_token_count", 0) or 0,
                                 "output_tokens": out_tokens}

    return complete


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    return c_common.finish(c_common.run_task(_complete(state), item, state["lang"], p),
                           p.get("price"))


def describe(state: dict) -> dict:
    from importlib.metadata import version

    p = state["params"]
    return {"model": p["model"], "google_genai": version("google-genai"),
            "task": p.get("task", "translate"), "prompt": p.get("prompt", "dub")}


if __name__ == "__main__":
    serve(load, run, describe)
