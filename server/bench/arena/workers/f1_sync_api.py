"""F1 worker: sync. labs API (sync.so) — sync-3, lipsync-2-pro, lipsync-2, lipsync-1.9.0-beta.

API (docs.sync.so, checked 2026-09-28): ``POST https://api.sync.so/v2/generate`` with header
``x-api-key``; multipart form (``video``, ``audio`` files < 20 MB, ``model``, ``options`` as a
JSON string) or JSON with URLs. Poll ``GET /v2/generate/{id}`` until ``status`` is COMPLETED
(FAILED / REJECTED = error); the result is ``outputUrl``.

List price per second of output at 25 fps (sync.so/docs/models/lipsync, 2026-09-28; the range is
plan tier, the higher pay-as-you-go end is used): sync-3 $0.107–0.133, lipsync-2-pro
$0.067–0.083, lipsync-2 $0.04–0.05, lipsync-1.9.0-beta $0.02–0.025.

params: model (pinned id), sync_mode ("cut_off"), model_mode (sync-3: lips|face|head),
temperature, active_speaker_detection, occlusion_detection_enabled, price_per_s (override).
Key: SYNC_API_KEY.
"""
from __future__ import annotations

import json
from pathlib import Path

from _sdk import api_key, serve
from f_common import (
    cleanup,
    download,
    driving_audio,
    http,
    item_io,
    keep_audio_copy,
    multipart,
    poll,
    probe,
    to_cfr,
    video_payload,
    workdir,
)

BASE = "https://api.sync.so/v2"
PRICE_PER_S = {"sync-3": 0.133, "lipsync-2-pro": 0.083, "lipsync-2": 0.05,
               "lipsync-1.9.0-beta": 0.025}
OPTION_KEYS = ("sync_mode", "model_mode", "temperature", "active_speaker_detection",
               "occlusion_detection_enabled")


def load(params: dict, lang: str):
    return {"key": api_key("SYNC_API_KEY"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    model = p["model"]
    video, _ = item_io(item)
    work = workdir(out)
    clip = to_cfr(video, work / "clip.mp4", fps=25)
    audio = driving_audio(item, work, sr=44100)
    options = {k: p[k] for k in OPTION_KEYS if k in p}
    options.setdefault("sync_mode", "cut_off")
    body, ctype = multipart({"model": model, "options": json.dumps(options)},
                            {"video": clip, "audio": audio})
    hdr = {"x-api-key": state["key"]}
    job = http("POST", f"{BASE}/generate", headers={**hdr, "Content-Type": ctype}, data=body)
    done = poll(lambda: http("GET", f"{BASE}/generate/{job['id']}", headers=hdr),
                done=lambda j: j.get("status") == "COMPLETED",
                failed=lambda j: j.get("status") in ("FAILED", "REJECTED"), interval=10)
    produced = download(done["outputUrl"], work / "sync.mp4")
    seconds = float(done.get("outputDuration") or probe(produced)["duration"])
    price = float(p.get("price_per_s", PRICE_PER_S.get(model, 0.133)))
    payload = video_payload(out, produced, keep_audio_copy(audio, out), job_id=job["id"],
                            _cost_usd=round(seconds * price, 4))
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": state["params"]["model"], "vendor": "sync.so"}


if __name__ == "__main__":
    serve(load, run, describe)
