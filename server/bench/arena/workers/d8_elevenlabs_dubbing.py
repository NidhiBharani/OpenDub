"""D3 control / D8 comparator: ElevenLabs Dubbing API (productised cascade: ASR → MT → cloned TTS
with its own isochrony). POST /v1/dubbing (multipart: file, source_lang, target_lang,
num_speakers, watermark=false, drop_background_audio), poll GET /v1/dubbing/{id} until
status == "dubbed", download GET /v1/dubbing/{id}/audio/{target_lang}, then fetch the target
transcript (GET /v1/dubbing/{id}/transcript/{lang}?format_type=srt) for ``text``.
The dub is deleted afterwards (DELETE /v1/dubbing/{id}).

Pricing (read 2026-09-28, estimate): dubbing bills per source minute; default
usd_per_minute 0.50 (override in params). Output: the dubbed audio as wav at its native rate.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, api_key, serve

API = "https://api.elevenlabs.io"


def load(params: dict, lang: str):
    return {"key": api_key("ELEVENLABS_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, key = state["params"], state["key"]
    src = (item.get("inputs") or {}).get("audio")
    if not src:
        raise Unsupported("dubbing item has no inputs.audio")
    h = {"xi-api-key": key}
    fields = {"target_lang": state["lang"], "num_speakers": p.get("num_speakers", 1),
              "watermark": "false",
              "drop_background_audio": str(bool(p.get("drop_background_audio", True))).lower()}
    if dv.src_lang(item):
        fields["source_lang"] = dv.src_lang(item)
    body, ctype = dv.multipart(fields, {"file": (Path(src).name, Path(src).read_bytes(),
                                                 "audio/wav")})
    raw, _ = dv.http("POST", f"{API}/v1/dubbing", body=body,
                     headers={**h, "Content-Type": ctype}, timeout=300)
    dub = json.loads(raw)["dubbing_id"]
    try:
        deadline = time.time() + float(p.get("timeout_s", 1800))
        while True:
            st = dv.get_json(f"{API}/v1/dubbing/{dub}", headers=h)
            if st.get("status") == "dubbed":
                break
            if st.get("status") in ("failed", "error") or time.time() > deadline:
                raise RuntimeError(f"dubbing {dub} status {st.get('status')}: {st.get('error')}")
            time.sleep(10)
        audio, _ = dv.http("GET", f"{API}/v1/dubbing/{dub}/audio/{state['lang']}", headers=h,
                           timeout=300)
        wav = dv.transcode_to_wav(audio, dv.wav_path(out), ".mp4")
        text = ""
        try:
            srt, _ = dv.http("GET", f"{API}/v1/dubbing/{dub}/transcript/{state['lang']}"
                             "?format_type=srt", headers=h)
            lines = [ln for ln in srt.decode("utf-8", "replace").splitlines()
                     if ln.strip() and not ln.strip().isdigit() and "-->" not in ln]
            text = " ".join(re.sub(r"<[^>]+>", "", ln) for ln in lines)
        except dv.HttpError:
            pass
    finally:
        try:
            dv.http("DELETE", f"{API}/v1/dubbing/{dub}", headers=h, retries=1)
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            print(f'cleanup failed: {exc!r}', file=sys.stderr)
    cost = round(dv.duration_s(src) / 60 * float(p.get("usd_per_minute", 0.50)), 6)
    return {**dv.audio_payload(wav), "text": text, "_cost_usd": cost}


if __name__ == "__main__":
    serve(load, run)
