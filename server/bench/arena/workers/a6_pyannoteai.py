"""A6 worker: pyannoteAI hosted diarization (Precision-2 / Precision-3 / community-1).

docs.pyannote.ai (checked 2026-09-28), header ``Authorization: Bearer``: ``POST /v1/media/input``
``{url: "media://<key>"}`` → presigned ``url``; ``PUT`` the file there; ``POST /v1/diarize``
``{url: "media://<key>", model, exclusive?}`` → ``{jobId}``; poll ``GET /v1/jobs/{jobId}`` until
``status == succeeded`` → ``output.diarization[{speaker, start, end}]`` (and
``exclusiveDiarization``). Prices (pyannote.ai/pricing, 2026-09-28): Precision-3 €0.112/h,
community-1 €0.035/h; Precision-2 not listed separately. params: model, exclusive,
usd_per_hour.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, poll, request

BASE = "https://api.pyannote.ai/v1"
PRICE = {"precision-3": 0.13, "precision-2": 0.13, "community-1": 0.04}  # € → $ ≈ 1.17


def load(params: dict, lang: str):
    return {"key": api_key("PYANNOTEAI_API_KEY"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p, hdr = state["params"], {"Authorization": f"Bearer {state['key']}"}
    audio = item["inputs"]["audio"]
    key = f"media://opendub-{uuid.uuid4().hex}"
    up = request("POST", f"{BASE}/media/input", json_body={"url": key}, headers=hdr)
    request("PUT", up["url"], data=Path(audio).read_bytes(),
            headers={"Content-Type": "application/octet-stream"}, raw=True)
    model = p.get("model", "precision-3")
    body = {"url": key, "model": model}
    if p.get("exclusive"):
        body["exclusive"] = True
    job = request("POST", f"{BASE}/diarize", json_body=body, headers=hdr)

    def check():
        r = request("GET", f"{BASE}/jobs/{job['jobId']}", headers=hdr)
        if r.get("status") in ("failed", "canceled"):
            raise RuntimeError(f"pyannoteAI job {r.get('status')}: {r}")
        return r if r.get("status") == "succeeded" else None

    res = poll(check)
    rows = res["output"].get("exclusiveDiarization" if p.get("exclusive") else "diarization", [])
    return {"turns": [{"start": r["start"], "end": r["end"], "speaker": str(r["speaker"])}
                      for r in rows],
            "_cost_usd": audio_seconds(audio) / 3600 * float(p.get("usd_per_hour",
                                                                   PRICE.get(model, 0.13)))}


if __name__ == "__main__":
    serve(load, run)
