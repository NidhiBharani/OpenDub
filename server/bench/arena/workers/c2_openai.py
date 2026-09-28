"""C2 worker (also C1/C3/C4/C5/C6 via ``params.task``): OpenAI models through the Responses API.

params: model (API model id, pinned in the candidate), reasoning_effort (none|low|medium|high;
optional), max_tokens, task, prompt (see c_common), price ([USD per 1M input, per 1M output];
overrides the table below).

List prices: see :data:`PRICES` (USD per 1M tokens input / output, from
https://platform.openai.com/docs/pricing as recorded in the candidate's ``verified`` notes).
Reasoning tokens bill as output and are included in ``usage.output_tokens``. Needs
OPENAI_API_KEY; runs in the ``c2_api`` env (openai SDK imported in ``load``).
"""
from __future__ import annotations

from pathlib import Path

import c_common
from _sdk import api_key, serve

# USD per 1M tokens (input, output), developers.openai.com/api/docs/pricing, 2026-09-28
PRICES: dict[str, tuple[float, float]] = {
    "gpt-6-astra": (10.0, 50.0), "gpt-6-sol": (2.0, 10.0), "gpt-6-luna": (0.10, 0.50),
    "gpt-5.6-sol": (4.0, 20.0), "gpt-5.6-terra": (2.0, 12.0), "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5": (5.0, 30.0), "gpt-5-mini": (0.25, 2.0), "gpt-5-nano": (0.05, 0.40)}


def load(params: dict, lang: str):
    import openai

    c_common.check_task_lang(params, lang)
    client = openai.OpenAI(api_key=api_key("OPENAI_API_KEY"), max_retries=6)
    return {"client": client, "params": params, "lang": lang}


def _complete(state: dict):
    p = state["params"]

    def complete(messages, *, temperature=None, json_mode=False, max_tokens=2048):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [{"role": m["role"], "content": m["content"]} for m in messages
                 if m["role"] != "system"]
        kwargs: dict = {"model": p["model"], "input": convo,
                        "max_output_tokens": max(int(max_tokens), 16000)}
        if system:
            kwargs["instructions"] = system
        if p.get("reasoning_effort"):
            kwargs["reasoning"] = {"effort": p["reasoning_effort"]}
        elif temperature is not None and p.get("sampling", False):
            kwargs["temperature"] = temperature   # reasoning models reject temperature
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}
        resp = state["client"].responses.create(**kwargs)
        usage = resp.usage
        return resp.output_text or "", {"input_tokens": getattr(usage, "input_tokens", 0),
                                        "output_tokens": getattr(usage, "output_tokens", 0)}

    return complete


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    price = p.get("price") or PRICES.get(p["model"])
    return c_common.finish(c_common.run_task(_complete(state), item, state["lang"], p), price)


def describe(state: dict) -> dict:
    import openai

    p = state["params"]
    return {"model": p["model"], "reasoning_effort": p.get("reasoning_effort"),
            "openai": openai.__version__, "task": p.get("task", "translate"),
            "prompt": p.get("prompt", "dub")}


if __name__ == "__main__":
    serve(load, run, describe)
