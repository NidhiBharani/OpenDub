"""F4 worker: Alibaba Model Studio (DashScope) Qwen image editing — qwen-image-edit-plus,
qwen-image-2.0 (API-only; no open weights for 2.0) — on the mid-span keyframe.

API (alibabacloud.com Model Studio docs, checked 2026-09-28): synchronous ``POST
{base_url}/api/v1/services/aigc/multimodal-generation/generation`` with ``Authorization: Bearer
$DASHSCOPE_API_KEY``; body ``{model, input: {messages: [{role: user, content: [{image:
<data URI>}, {text: …}]}]}, parameters: {n: 1, watermark: false}}``; the edited image URL is
``output.choices[0].message.content[0].image`` (expires after 24 h).
List price per image, Singapore (2026-09-28): qwen-image-edit-plus $0.03, qwen-image-2.0 $0.035,
qwen-image-2.0-pro $0.075 (2 requests/minute).
params: model (pinned id), base_url (default https://dashscope-intl.aliyuncs.com — the newer
workspace host ``https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com`` also works),
price_per_image. Key: DASHSCOPE_API_KEY.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from _sdk import api_key, serve
from f_common import http, video_payload

PRICE = {"qwen-image-edit-plus": 0.03, "qwen-image-2.0": 0.035, "qwen-image-2.0-pro": 0.075}


def load(params: dict, lang: str):
    return {"key": api_key("DASHSCOPE_API_KEY"), "params": params}


def _edit(state: dict):
    import time
    import urllib.request

    from PIL import Image

    p = state["params"]
    base = p.get("base_url", "https://dashscope-intl.aliyuncs.com").rstrip("/")

    def fn(image, prompt: str, text: dict):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        resp = http("POST", f"{base}/api/v1/services/aigc/multimodal-generation/generation",
                    headers={"Authorization": f"Bearer {state['key']}"},
                    body={"model": p["model"], "input": {"messages": [{"role": "user", "content": [
                        {"image": uri}, {"text": prompt}]}]},
                          "parameters": {"n": 1, "watermark": False}})
        url = resp["output"]["choices"][0]["message"]["content"][0]["image"]
        if p.get("min_interval_s"):
            time.sleep(float(p["min_interval_s"]))
        with urllib.request.urlopen(url, timeout=120) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return fn


def run(state: dict, item: dict, out: Path) -> dict:
    from f4_common import keyframe_edit

    p = state["params"]
    dst = keyframe_edit(item, out, _edit(state), p)
    price = float(p.get("price_per_image", PRICE.get(p["model"], 0.05)))
    return video_payload(out, dst, None, remux=False,
                         _cost_usd=round(len(item["inputs"]["texts"]) * price, 4))


def describe(state: dict) -> dict:
    return {"model": state["params"]["model"], "vendor": "dashscope"}


if __name__ == "__main__":
    serve(load, run, describe)
