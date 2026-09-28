"""G6 worker (also G2/G3/G4/G5 LLM-judge candidates): Google Gemini as an audio/video reviewer.

Protocol and prompts: ``g6_rubric`` (``params.task``: review | speaker | mos | emotion | sync;
``params.protocol``: holistic | decomposed). Gemini takes audio and video natively, so the
off-sync check sees the picture.

params: model (pinned id, e.g. gemini-3.7-flash), task, protocol, temperature (omit to keep the
model default; Gemini 3 recommends the default), prices (USD per 1M tokens: ``in``, ``out``;
output includes thinking tokens), inline_max_mb (default 18: larger media go through the Files
API).

Pricing checked 2026-09-28 at https://ai.google.dev/gemini-api/docs/pricing (paid tier, not
split by modality): gemini-3.7-flash / gemini-3.8-flash $0.75 in / $3.75 out per 1M tokens
through 2026-12-31 (then $1.50 / $7.50); gemini-3.1-pro-preview $2.00 / $12.00 (≤200k prompt);
gemini-3.5-flash $1.50 / $9.00. The candidate YAML carries the numbers used for ``_cost_usd``.

Key: GEMINI_API_KEY (or GOOGLE_API_KEY). Env: g6_api (google-genai SDK).
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import g6_rubric as R
from _sdk import api_key, serve


def load(params: dict, lang: str):
    from google import genai

    client = genai.Client(api_key=api_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    return {"client": client, "params": params, "lang": lang,
            "tmp": Path(tempfile.mkdtemp(prefix="opendub-gemini-")),
            "model": params.get("model", "gemini-3.7-flash")}


def _media_part(state: dict, path: str):
    from google.genai import types

    if R.mime_of(path).startswith("audio/"):
        path = R.compact_audio(path, state["tmp"])
    data = Path(path).read_bytes()
    if len(data) <= float(state["params"].get("inline_max_mb", 18)) * 2**20:
        return types.Part.from_bytes(data=data, mime_type=R.mime_of(path))
    client = state["client"]
    f = client.files.upload(file=path)
    while getattr(f.state, "name", str(f.state)) == "PROCESSING":
        time.sleep(2)
        f = client.files.get(name=f.name)
    return f


def _retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code in (408, 429, 500, 502, 503, 504) or "timeout" in type(exc).__name__.lower()


def ask(state: dict, parts: list, prompt: str) -> tuple[str, dict]:
    from google.genai import types

    contents = [p[1] if p[0] == "text" else _media_part(state, p[1]) for p in parts] + [prompt]
    cfg: dict = {"system_instruction": R.SYSTEM, "response_mime_type": "application/json"}
    if "temperature" in state["params"]:
        cfg["temperature"] = float(state["params"]["temperature"])
    resp = R.with_retries(lambda: state["client"].models.generate_content(
        model=state["model"], contents=contents, config=types.GenerateContentConfig(**cfg)),
        retryable=_retryable)
    um = resp.usage_metadata
    usage = {"in": getattr(um, "prompt_token_count", 0) or 0,
             "out": (getattr(um, "candidates_token_count", 0) or 0)
             + (getattr(um, "thoughts_token_count", 0) or 0)}
    return resp.text or "", usage


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    return R.run_item(ask, state, item, state["lang"], task=p.get("task", "review"),
                      protocol=p.get("protocol", "holistic"), caps={"video": True},
                      prices=p.get("prices"))


def describe(state: dict) -> dict:
    import google.genai

    p = state["params"]
    return {"model": state["model"], "task": p.get("task", "review"),
            "protocol": p.get("protocol", "holistic"), "google_genai": google.genai.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
