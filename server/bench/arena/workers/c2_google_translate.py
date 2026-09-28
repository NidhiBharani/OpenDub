"""C2 worker: Google Cloud Translation (Basic, v2 REST with an API key) — the "ONLINE-G" style
commercial MT reference. No duration budget, no context.

params: model (nmt; optional), price_per_mchar (USD per 1M characters; default 20.0, the
Cloud Translation Basic list price — verify on cloud.google.com/translate/pricing).

Needs GOOGLE_API_KEY with the Cloud Translation API enabled; stdlib only (``server`` env).
"""
from __future__ import annotations

import html
import urllib.parse
from pathlib import Path

import c_common
from _sdk import api_key, serve


def load(params: dict, lang: str):
    return {"key": api_key("GOOGLE_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    src, tgt = c_common.direction(state["lang"], item)
    text = item["inputs"]["text"]
    body = {"q": [text], "source": src, "target": tgt, "format": "text"}
    if p.get("model"):
        body["model"] = p["model"]
    url = ("https://translation.googleapis.com/language/translate/v2?key="
           + urllib.parse.quote(state["key"]))
    data = c_common.http_json(url, body)
    out_text = html.unescape(data["data"]["translations"][0]["translatedText"])
    price = float(p.get("price_per_mchar", 20.0))
    return {"text": out_text, "src_lang": src, "tgt_lang": tgt,
            "_cost_usd": round(len(text) * price / 1e6, 6)}


if __name__ == "__main__":
    serve(load, run)
