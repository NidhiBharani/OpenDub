"""F1 worker: Replicate-hosted lip-sync models (the app's ``lipsync.replicate`` provider).

API (replicate.com/docs, checked 2026-09-28): upload with ``POST /v1/files`` (multipart field
``content``) → ``urls.get``; ``POST /v1/models/{owner}/{name}/predictions`` with
``{"input": {…}}`` (official models, latest version) or ``POST /v1/predictions`` with
``version`` (community models); poll ``GET /v1/predictions/{id}`` until ``succeeded``.
Header ``Authorization: Bearer $REPLICATE_API_TOKEN``.

params: model ("owner/name"), version (optional pin), video_field / audio_field (input names
differ per model: ``video``/``audio`` for bytedance/latentsync, sync/*, heygen/*, pixverse;
``face``/``input_audio`` for chenxwh/video-retalking; ``video_input``/``audio_input`` for
tmappdev/lipsync), extra_input (dict), price_per_run or price_per_s (list price; Replicate bills
per GPU-second for community models, so this is an estimate — see the candidate notes).
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

BASE = "https://api.replicate.com/v1"


def load(params: dict, lang: str):
    return {"key": api_key("REPLICATE_API_TOKEN"), "params": params}


def _upload(hdr: dict, path: Path) -> str:
    body, ctype = multipart({}, {"content": path})
    resp = http("POST", f"{BASE}/files", headers={**hdr, "Content-Type": ctype}, data=body)
    return (resp.get("urls") or {}).get("get") or resp["url"]


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    hdr = {"Authorization": f"Bearer {state['key']}"}
    video, _ = item_io(item)
    work = workdir(out)
    clip = to_cfr(video, work / "clip.mp4", fps=25)
    audio = driving_audio(item, work, sr=44100)
    inputs = {p.get("video_field", "video"): _upload(hdr, clip),
              p.get("audio_field", "audio"): _upload(hdr, audio), **p.get("extra_input", {})}
    if p.get("version"):
        pred = http("POST", f"{BASE}/predictions", headers=hdr,
                    body={"version": p["version"], "input": inputs})
    else:
        pred = http("POST", f"{BASE}/models/{p['model']}/predictions", headers=hdr,
                    body={"input": inputs})
    done = poll(lambda: http("GET", f"{BASE}/predictions/{pred['id']}", headers=hdr),
                done=lambda j: j.get("status") == "succeeded",
                failed=lambda j: j.get("status") in ("failed", "canceled"), interval=5)
    output = done.get("output")
    url = output[-1] if isinstance(output, list) else output
    if isinstance(url, dict):
        url = url.get("video") or next(iter(url.values()))
    produced = download(str(url), work / "replicate.mp4")
    if "price_per_s" in p:
        cost = probe(produced)["duration"] * float(p["price_per_s"])
    else:
        cost = float(p.get("price_per_run", 0.0))
    payload = video_payload(out, produced, keep_audio_copy(audio, out), prediction=pred["id"],
                            _cost_usd=round(cost, 4),
                            predict_time_s=(done.get("metrics") or {}).get("predict_time"))
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": state["params"]["model"], "version": state["params"].get("version")}


if __name__ == "__main__":
    serve(load, run, describe)
