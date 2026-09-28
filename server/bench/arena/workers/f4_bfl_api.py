"""F4 worker: Black Forest Labs API image editing (FLUX.1 Kontext [pro], FLUX.2 [pro]/[flex]).

API (docs.bfl.ai, checked 2026-09-28): ``POST https://api.bfl.ai/v1/<endpoint>`` with header
``x-key``; body ``{prompt, input_image: <base64>, output_format: "png", seed,
safety_tolerance}``; poll the returned ``polling_url`` until ``status == "Ready"``; download
``result.sample`` (a signed URL valid for 10 minutes).
List price (2026-09-28): flux-kontext-pro $0.04/image, flux-kontext-max $0.08; FLUX.2 pro
editing from $0.045, flex from $0.05 (per megapixel; the minimum is used).
params: endpoint ("flux-kontext-pro" | "flux-2-pro" | "flux-2-flex" …), seed (0),
price_per_image. Key: BFL_API_KEY. The open FLUX.1 Kontext [dev] weights are non-commercial.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from _sdk import api_key, serve
from f_common import http, poll, video_payload

PRICE = {"flux-kontext-pro": 0.04, "flux-kontext-max": 0.08, "flux-2-pro": 0.045,
         "flux-2-flex": 0.05, "flux-2-max": 0.07}


def load(params: dict, lang: str):
    return {"key": api_key("BFL_API_KEY"), "params": params}


def _edit(state: dict):
    import urllib.request

    from PIL import Image

    p = state["params"]
    hdr = {"x-key": state["key"], "accept": "application/json"}

    def fn(image, prompt: str, text: dict):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        job = http("POST", f"https://api.bfl.ai/v1/{p['endpoint']}", headers=hdr,
                   body={"prompt": prompt, "input_image": base64.b64encode(buf.getvalue()).decode(),
                         "output_format": "png", "seed": int(p.get("seed", 0)),
                         "safety_tolerance": int(p.get("safety_tolerance", 2))})
        done = poll(lambda: http("GET", job["polling_url"], headers=hdr),
                    done=lambda j: j.get("status") == "Ready",
                    failed=lambda j: j.get("status") in ("Error", "Failed",
                                                          "Request Moderated",
                                                          "Content Moderated"), interval=2)
        with urllib.request.urlopen(done["result"]["sample"], timeout=120) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return fn


def run(state: dict, item: dict, out: Path) -> dict:
    from f4_common import keyframe_edit

    p = state["params"]
    dst = keyframe_edit(item, out, _edit(state), p)
    price = float(p.get("price_per_image", PRICE.get(p["endpoint"], 0.05)))
    return video_payload(out, dst, None, remux=False,
                         _cost_usd=round(len(item["inputs"]["texts"]) * price, 4))


def describe(state: dict) -> dict:
    return {"model": state["params"]["endpoint"], "vendor": "bfl.ai"}


if __name__ == "__main__":
    serve(load, run, describe)
