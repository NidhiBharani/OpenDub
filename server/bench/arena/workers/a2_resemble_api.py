"""A2 worker: Resemble AI hosted audio enhancement. ``POST https://app.resemble.ai/api/v2/
audio_enhancements`` (``Authorization: Bearer``; multipart ``audio_file``, ``remove_noise``,
``normalize``, ``studio_sound``) → 202 ``{uuid, status: pending}``; poll ``GET …/{uuid}`` until
``status == completed`` and download ``enhanced_audio_url``. $0.045 per started minute
(docs.resemble.ai/audio-tools/audio-enhancement, read 2026-09-28; the GET path is inferred).
"""
from __future__ import annotations

import math
from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, download, poll, post_form, request

BASE = "https://app.resemble.ai/api/v2/audio_enhancements"


def load(params: dict, lang: str):
    return {"key": api_key("RESEMBLE_API_KEY"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p, hdr = state["params"], {"Authorization": f"Bearer {state['key']}"}
    audio = item["inputs"]["audio"]
    job = post_form(BASE, {"remove_noise": bool(p.get("remove_noise", True)),
                           "normalize": bool(p.get("normalize", False)),
                           "studio_sound": bool(p.get("studio_sound", True))},
                    {"audio_file": audio}, hdr)

    def check():
        r = request("GET", f"{BASE}/{job['uuid']}", headers=hdr)
        r = r.get("item", r)
        if r.get("status") in ("failed", "error"):
            raise RuntimeError(f"enhancement failed: {r}")
        return r if r.get("status") == "completed" else None

    res = poll(check, interval=5.0)
    dst = out.with_suffix(".wav")
    download(res["enhanced_audio_url"], dst)
    return {"files": {"audio": str(dst)},
            "_cost_usd": math.ceil(audio_seconds(audio) / 60) * float(p.get("usd_per_min",
                                                                           0.045))}


if __name__ == "__main__":
    serve(load, run)
