"""A1/A2 worker: AudioShake API (``https://api.audioshake.ai``, header ``x-api-key``).

Flow (developer.audioshake.ai, checked 2026-09-28 for /tasks and the model list; the asset
upload route is from the same docs but not exercised): upload the file as an asset, ``POST
/tasks`` ``{"assetId", "targets": [{"model": "dialogue", "formats": ["wav"], "residual":
true}, …]}``, poll ``GET /tasks/{id}`` until every target is completed, download
``targets[].output[].link`` (presigned, 1 h).

params: targets (A1 default: dialogue with residual = bed, plus music_fx; A2: speech_denoise /
speech_dereverb), role (dialogue | audio), usd_per_credit (not public — dashboard only; default
0 so spend is recorded as 0 until set), credits_per_min (dialogue 1.5, speech_denoise 1.5,
speech_dereverb 2.0, music_fx 1.5 per the model page).
"""
from __future__ import annotations

import math
from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, download, poll, post_form, request

BASE = "https://api.audioshake.ai"
CREDITS = {"dialogue": 1.5, "effects": 1.5, "music_fx": 1.5, "speech_denoise": 1.5,
           "speech_dereverb": 2.0, "multi_voice": 10.0, "vocals": 1.0, "instrumental": 1.0}


def load(params: dict, lang: str):
    return {"key": api_key("AUDIOSHAKE_API_KEY"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p, hdr = state["params"], {"x-api-key": state["key"]}
    audio = item["inputs"]["audio"]
    role = p.get("role", "dialogue")
    models = p.get("targets") or (["dialogue"] if role == "dialogue" else ["speech_denoise"])
    asset = post_form(f"{BASE}/assets", {}, {"file": audio}, hdr)
    targets = [{"model": m, "formats": ["wav"], "residual": m == "dialogue"} for m in models]
    task = request("POST", f"{BASE}/tasks", json_body={"assetId": asset["id"],
                                                        "targets": targets}, headers=hdr)

    def check():
        t = request("GET", f"{BASE}/tasks/{task['id']}", headers=hdr)
        sts = [tg.get("status") for tg in t.get("targets", [])]
        if "error" in sts or "failed" in sts:
            raise RuntimeError(f"AudioShake task failed: {t}")
        return t if sts and all(s == "completed" for s in sts) else None

    t = poll(check, interval=5.0)
    files: dict[str, str] = {}
    for tg in t["targets"]:
        for o in tg.get("output") or []:
            name = o.get("name") or o.get("type") or tg["model"]
            dst = out.with_name(f"{out.name}.{tg['model']}.{name}.wav".replace("/", "_"))
            download(o["link"], dst)
            residual = "residual" in str(name).lower() or "background" in str(name).lower()
            if tg["model"] == "dialogue":
                files["background" if residual else "dialogue"] = str(dst)
            elif tg["model"] == "music_fx" and not residual:
                files["music_effects"] = str(dst)
            elif role == "audio" and not residual:
                files["audio"] = str(dst)
    minutes = math.ceil(audio_seconds(audio) / 60)
    credits = sum(CREDITS.get(m, 1.5) for m in models) * minutes
    return {"files": files, "_cost_usd": credits * float(p.get("usd_per_credit", 0.0)),
            "credits": credits}


if __name__ == "__main__":
    serve(load, run)
