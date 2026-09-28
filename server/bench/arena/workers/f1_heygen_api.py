"""F1 worker: HeyGen lipsync API (v3) with our own driving audio.

API (developers.heygen.com, checked 2026-09-28): upload each file with
``POST https://api.heygen.com/v3/assets`` (multipart, ≤ 32 MB; the form field name is not
documented — ``file`` is assumed) → ``asset_id``; ``POST /v3/lipsyncs`` with
``{"video": {"type": "asset_id", …}, "audio": {…}, "mode": "speed"|"precision"}`` →
``data.lipsync_id``; poll ``GET /v3/lipsyncs/{id}`` until ``status`` is completed; download the
presigned ``video_url``. Header ``x-api-key``.

Price (2026-09-28; HeyGen's docs show no table, the rates match its Replicate listings):
speed $0.0333/s, precision $0.0667/s of output. Video Translate (their end-to-end product) is
not used: it replaces translation and TTS too, which F1 must not do.
params: mode (speed | precision), price_per_s (override). Key: HEYGEN_API_KEY.
"""
from __future__ import annotations

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

BASE = "https://api.heygen.com/v3"
PRICE_PER_S = {"speed": 0.0333, "precision": 0.0667}


def _data(resp: dict) -> dict:
    return resp.get("data") or resp


def load(params: dict, lang: str):
    return {"key": api_key("HEYGEN_API_KEY"), "params": params}


def _upload(key: str, path: Path) -> str:
    body, ctype = multipart({}, {"file": path})
    resp = _data(http("POST", f"{BASE}/assets", headers={"x-api-key": key, "Content-Type": ctype},
                      data=body))
    return resp.get("asset_id") or resp["id"]


def run(state: dict, item: dict, out: Path) -> dict:
    p, key = state["params"], state["key"]
    mode = p.get("mode", "precision")
    video, _ = item_io(item)
    work = workdir(out)
    clip = to_cfr(video, work / "clip.mp4", fps=25)
    audio = driving_audio(item, work, sr=44100)
    body = {"video": {"type": "asset_id", "asset_id": _upload(key, clip)},
            "audio": {"type": "asset_id", "asset_id": _upload(key, audio)},
            "mode": mode, "enable_dynamic_duration": False, "enable_watermark": False}
    job = _data(http("POST", f"{BASE}/lipsyncs", headers={"x-api-key": key}, body=body))
    jid = job.get("lipsync_id") or job["id"]
    done = poll(lambda: _data(http("GET", f"{BASE}/lipsyncs/{jid}", headers={"x-api-key": key})),
                done=lambda j: j.get("status") == "completed",
                failed=lambda j: j.get("status") == "failed", interval=10)
    produced = download(done["video_url"], work / "heygen.mp4")
    seconds = probe(produced)["duration"]
    price = float(p.get("price_per_s", PRICE_PER_S[mode]))
    payload = video_payload(out, produced, keep_audio_copy(audio, out), job_id=jid,
                            _cost_usd=round(seconds * price, 4))
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": f"heygen-lipsync-{state['params'].get('mode', 'precision')}"}


if __name__ == "__main__":
    serve(load, run, describe)
