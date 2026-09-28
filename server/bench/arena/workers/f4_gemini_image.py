"""F4 worker: Google Gemini image editing ("Nano Banana Pro" = gemini-3-pro-image, "Nano Banana 2"
= gemini-3.1-flash-image) on the mid-span keyframe.

API (ai.google.dev, checked 2026-09-28): ``POST https://generativelanguage.googleapis.com/
v1beta/models/{model}:generateContent`` with header ``x-goog-api-key``; body parts = the
instruction text + ``inline_data`` (base64 PNG); ``generationConfig.responseModalities =
["TEXT", "IMAGE"]``, ``imageConfig.imageSize``; the edited image comes back as
``candidates[0].content.parts[].inlineData.data``.
List price per output image (2026-09-28): 3 Pro $0.134 (1K/2K), $0.24 (4K), +$0.0011 per input
image; 3.1 Flash $0.045 (1K), $0.067 (2K), $0.101 (4K).
params: model (pinned id), image_size ("2K"), price_per_image (override). Key: GEMINI_API_KEY
(or GOOGLE_API_KEY). One API call per text instance.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from _sdk import api_key, serve
from f_common import http, video_payload

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PRICE = {("gemini-3-pro-image", "1K"): 0.1351, ("gemini-3-pro-image", "2K"): 0.1351,
         ("gemini-3-pro-image", "4K"): 0.2411, ("gemini-3.1-flash-image", "1K"): 0.045,
         ("gemini-3.1-flash-image", "2K"): 0.067, ("gemini-3.1-flash-image", "4K"): 0.101}


def load(params: dict, lang: str):
    return {"key": api_key("GEMINI_API_KEY", "GOOGLE_API_KEY"), "params": params, "calls": 0}


def _edit(state: dict):
    from PIL import Image

    p = state["params"]
    model, size = p["model"], p.get("image_size", "2K")

    def fn(image, prompt: str, text: dict):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        body = {"contents": [{"parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png",
                                     "data": base64.b64encode(buf.getvalue()).decode()}}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                                     "imageConfig": {"imageSize": size}}}
        resp = http("POST", URL.format(model=model), headers={"x-goog-api-key": state["key"]},
                    body=body)
        state["calls"] += 1
        for part in resp["candidates"][0]["content"]["parts"]:
            data = (part.get("inlineData") or part.get("inline_data") or {}).get("data")
            if data:
                return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        raise RuntimeError(f"no image in the response: {str(resp)[:300]}")
    return fn


def run(state: dict, item: dict, out: Path) -> dict:
    from f4_common import keyframe_edit

    p = state["params"]
    dst = keyframe_edit(item, out, _edit(state), p)
    price = float(p.get("price_per_image", PRICE.get((p["model"], p.get("image_size", "2K")),
                                                     0.134)))
    n = len(item["inputs"]["texts"])
    return video_payload(out, dst, None, remux=False, _cost_usd=round(n * price, 4))


def describe(state: dict) -> dict:
    return {"model": state["params"]["model"], "image_size": state["params"].get("image_size")}


if __name__ == "__main__":
    serve(load, run, describe)
