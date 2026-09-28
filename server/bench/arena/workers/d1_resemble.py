"""D1/D5 worker: Resemble AI synthesis API.

POST https://f.cluster.resemble.ai/synthesize, ``Authorization: Bearer``, body {voice_uuid, data:
text, sample_rate: 24000, output_format: wav} → {audio_content: base64, success} (Resemble docs
as known 2026; NOT re-verified). Voices (stock or rapid clones made in the Resemble app) are
configured per language in ``voices``; per-item instant cloning over the API is not wired, so
the D1 cloning candidate is registered disabled.

params: voices ({lang: {female, male}} voice uuids), usd_per_million_chars (estimate).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import d_voice as dv
from _sdk import api_key, serve


def load(params: dict, lang: str):
    return {"key": api_key("RESEMBLE_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    body = {"voice_uuid": dv.pick_voice(p.get("voices"), state["lang"], item), "data": text,
            "sample_rate": 24000, "output_format": "wav"}
    raw, _ = dv.post_json("https://f.cluster.resemble.ai/synthesize", body,
                          headers={"Authorization": f"Bearer {state['key']}"}, timeout=300)
    resp = json.loads(raw)
    if not resp.get("success", True):
        raise RuntimeError(f"Resemble error: {resp.get('issues') or resp}")
    wav = dv.transcode_to_wav(base64.b64decode(resp["audio_content"]), dv.wav_path(out), ".wav")
    usd = float(p.get("usd_per_million_chars", 100.0))
    return {**dv.audio_payload(wav), "_cost_usd": dv.char_cost(text, usd)}


if __name__ == "__main__":
    serve(load, run)
