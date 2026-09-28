"""C2 worker (also C1/C3/C4/C5/C6 via ``params.task``): Claude through the Anthropic Messages API.

params: model (claude-opus-5 | claude-fable-5-1 | claude-sonnet-5 | …), effort
(low|medium|high|xhigh|max; default high), max_tokens, task, prompt (see c_common), price
([USD per 1M input, per 1M output]; overrides the table below).

List prices (Anthropic API, USD per 1M tokens input / output; skill cache 2026-06-24, checked
2026-09-28): claude-fable-5-1 10/50 · claude-fable-5 10/50 · claude-opus-5 5/25 ·
claude-sonnet-5 2/10 · claude-haiku-4-5 1/5. Thinking tokens bill as output and are included in
``usage.output_tokens``, so ``_cost_usd`` covers them.

API notes for the 5-series: adaptive thinking is on by default (depth via ``effort``); sampling
parameters (temperature) are rejected, so the task temperatures are not sent; assistant prefill
is rejected. A ``refusal`` stop reason is recorded as an error for the item. Server-side
refusal fallbacks are deliberately NOT enabled: a fallback would silently answer with another
model and the arena would credit it to this candidate. Needs ANTHROPIC_API_KEY; runs in the
``c2_api`` env (anthropic SDK imported in ``load``).
"""
from __future__ import annotations

from pathlib import Path

import c_common
from _sdk import api_key, serve

PRICES = {"claude-fable-5-1": (10.0, 50.0), "claude-fable-5": (10.0, 50.0),
          "claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-haiku-4-5": (1.0, 5.0)}
NO_SAMPLING = ("claude-fable-5", "claude-opus-5", "claude-sonnet-5", "claude-opus-4-7",
               "claude-opus-4-8")


def load(params: dict, lang: str):
    import anthropic

    c_common.check_task_lang(params, lang)
    client = anthropic.Anthropic(api_key=api_key("ANTHROPIC_API_KEY"), max_retries=6)
    return {"client": client, "params": params, "lang": lang}


def _complete(state: dict):
    p = state["params"]
    model = p["model"]

    def complete(messages, *, temperature=None, json_mode=False, max_tokens=2048):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        kwargs: dict = {"model": model, "max_tokens": max(int(max_tokens), 16000),
                        "messages": convo}
        if system:
            kwargs["system"] = system
        if p.get("effort"):
            kwargs["output_config"] = {"effort": p["effort"]}
        if temperature is not None and not model.startswith(NO_SAMPLING):
            kwargs["temperature"] = temperature
        resp = state["client"].messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise RuntimeError(f"refusal ({getattr(details, 'category', None)})")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, {"input_tokens": resp.usage.input_tokens,
                      "output_tokens": resp.usage.output_tokens}

    return complete


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    price = p.get("price") or PRICES.get(p["model"])
    return c_common.finish(c_common.run_task(_complete(state), item, state["lang"], p), price)


def describe(state: dict) -> dict:
    import anthropic

    p = state["params"]
    return {"model": p["model"], "effort": p.get("effort"), "anthropic": anthropic.__version__,
            "task": p.get("task", "translate"), "prompt": p.get("prompt", "dub")}


if __name__ == "__main__":
    serve(load, run, describe)
