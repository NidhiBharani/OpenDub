"""D1/D2/D4/D5/D7 worker: Cartesia Sonic (sonic-3.6, snapshot sonic-3.6-2026-08-27).

POST https://api.cartesia.ai/tts/bytes, headers ``X-API-Key``, ``Cartesia-Version: 2026-03-01``
(current per Cartesia docs, 2026-09-28), body {model_id, transcript, voice: {mode: "id", id},
language, output_format: {container: "wav", encoding: "pcm_s16le", sample_rate: 44100}}.
Clone (mode clone, not re-verified): POST /voices/clone multipart {clip, name, language} → {id};
voices are deleted when the job ends (DELETE /voices/{id}).

params: model (sonic-3.6-2026-08-27), mode (clone | preset), voices ({lang: {female, male}} ids
for preset), usd_per_million_chars (49, Sonic 3.6 list price read 2026-09-28).
Sonic infers emotion and non-verbals from the transcript (no tags needed; D2/D7).
"""
from __future__ import annotations

import atexit
import hashlib
import json
import sys
from pathlib import Path

import d_voice as dv
from _sdk import api_key, serve

API = "https://api.cartesia.ai"


def load(params: dict, lang: str):
    state = {"key": api_key("CARTESIA_API_KEY"), "params": params, "lang": lang, "voices": {},
             "version": params.get("api_version", "2026-03-01")}

    def cleanup() -> None:
        for vid in state["voices"].values():
            try:
                dv.http("DELETE", f"{API}/voices/{vid}", headers=_h(state), retries=1)
            except Exception as exc:  # noqa: BLE001 - best effort
                print(f'cleanup failed: {exc!r}', file=sys.stderr)

    atexit.register(cleanup)
    return state


def _h(state: dict) -> dict:
    return {"X-API-Key": state["key"], "Cartesia-Version": state["version"]}


def _clone(state: dict, ref: str) -> str:
    data = Path(ref).read_bytes()
    digest = hashlib.sha1(data).hexdigest()
    if digest not in state["voices"]:
        body, ctype = dv.multipart({"name": f"opendub-arena-{digest[:8]}",
                                    "language": state["lang"]},
                                   {"clip": (Path(ref).name, data, "audio/wav")})
        raw, _ = dv.http("POST", f"{API}/voices/clone", body=body,
                         headers={**_h(state), "Content-Type": ctype})
        state["voices"][digest] = json.loads(raw)["id"]
    return state["voices"][digest]


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    if p.get("mode", "clone") == "clone":
        ref, _ = dv.ref_of(item)
        vid = _clone(state, ref)
    else:
        vid = dv.pick_voice(p.get("voices"), state["lang"], item)
    body = {"model_id": p.get("model", "sonic-3.6-2026-08-27"), "transcript": text,
            "voice": {"mode": "id", "id": vid}, "language": state["lang"],
            "output_format": {"container": "wav", "encoding": "pcm_s16le",
                              "sample_rate": int(p.get("sample_rate", 44100))}}
    raw, _ = dv.post_json(f"{API}/tts/bytes", body, headers=_h(state), timeout=300)
    wav = dv.transcode_to_wav(raw, dv.wav_path(out), ".wav")
    rate = float(p.get("usd_per_million_chars", 49.0))
    return {**dv.audio_payload(wav), "_cost_usd": dv.char_cost(text, rate)}


if __name__ == "__main__":
    serve(load, run)
