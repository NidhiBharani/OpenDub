"""C2 worker: DeepL API (dedicated MT; no duration budget, no emotion or speaker awareness).

params: model_type (quality_optimized | latency_optimized | prefer_quality_optimized; optional),
formality (optional), context (bool, default true: send context_before/after as DeepL
``context``, which is not billed), price_per_mchar (USD per 1M characters; default 25.0, the
DeepL API Pro usage price as listed on deepl.com/pro-api, checked 2026-09-28 — verify).

Billing is per source character, so ``_cost_usd`` = len(text) × price / 1e6. Free-tier keys
(ending ``:fx``) go to api-free.deepl.com automatically. Needs DEEPL_API_KEY; stdlib only, so it
runs in the ``server`` env.
"""
from __future__ import annotations

from pathlib import Path

import c_common
from _sdk import Unsupported, api_key, serve

TARGETS = {"en": "EN-US", "ja": "JA", "hi": "HI", "de": "DE", "fr": "FR", "es": "ES",
           "ko": "KO", "zh": "ZH-HANS", "pt": "PT-BR", "it": "IT"}


def load(params: dict, lang: str):
    key = api_key("DEEPL_API_KEY")
    base = "https://api-free.deepl.com" if key.endswith(":fx") else "https://api.deepl.com"
    return {"key": key, "base": base, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    src, tgt = c_common.direction(state["lang"], item)
    if tgt not in TARGETS:
        raise Unsupported(f"DeepL target {tgt!r} not configured")
    inp = item["inputs"]
    body: dict = {"text": [inp["text"]], "target_lang": TARGETS[tgt],
                  "source_lang": src.upper()}
    if p.get("model_type"):
        body["model_type"] = p["model_type"]
    if p.get("formality"):
        body["formality"] = p["formality"]
    ctx = [*(inp.get("context_before") or []), *(inp.get("context_after") or [])]
    if ctx and p.get("context", True):
        body["context"] = "\n".join(ctx)
    data = c_common.http_json(f"{state['base']}/v2/translate", body,
                              headers={"Authorization": f"DeepL-Auth-Key {state['key']}"})
    text = data["translations"][0]["text"]
    price = float(p.get("price_per_mchar", 25.0))
    return {"text": text, "src_lang": src, "tgt_lang": tgt,
            "model_type_used": data["translations"][0].get("model_type_used"),
            "_cost_usd": round(len(inp["text"]) * price / 1e6, 6)}


def describe(state: dict) -> dict:
    return {"api": state["base"], "model_type": state["params"].get("model_type")}


if __name__ == "__main__":
    serve(load, run, describe)
