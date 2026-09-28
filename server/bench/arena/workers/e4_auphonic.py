"""E4 API worker: Auphonic (Adaptive Leveler + loudness normalisation) via the Simple API.

``POST https://auphonic.com/api/simple/productions.json`` (Bearer API key) with the premix,
``leveler``, ``normloudness``, ``loudnesstarget`` and ``maxpeak`` from the item's delivery preset
(filtering/denoise off), then poll ``/api/production/{uuid}.json`` until Done and download the
first output file. Output format follows ``params.preset`` (create one with 48 kHz WAV output);
without a preset Auphonic's default output format applies (unverified: may be lossy).
Auphonic's own gating is programme loudness (a dialogue-gated preset is sent the same target).

Pricing: credits per hour of input audio; list price varies by plan (≈ $1–2.5/h; S plan 9 h/month).
``usd_per_hour`` (default 1.5, unverified) × duration is reported as the item cost.
params: preset?, usd_per_hour (1.5), poll_s (5), timeout_s (1800).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

from _sdk import api_key, serve

API = "https://auphonic.com/api"


def load(params: dict, lang: str):
    return {"key": api_key("AUPHONIC_API_KEY"), "p": params}


def _req(state: dict, url: str, data: bytes | None = None, ctype: str | None = None):
    headers = {"Authorization": f"bearer {state['key']}"}
    if ctype:
        headers["Content-Type"] = ctype
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers),
                                        timeout=600) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"HTTP {exc.code} for {url.split('?')[0]}") from None
    raise RuntimeError("unreachable")


def run(state: dict, item: dict, out: Path) -> dict:
    p, inp = state["p"], item["inputs"]
    d = inp["delivery"]
    src = Path(inp["audio"])
    with wave.open(str(src)) as w:
        hours = w.getnframes() / w.getframerate() / 3600
    fields = {"title": item["id"], "action": "start", "leveler": "true", "normloudness": "true",
              "loudnesstarget": str(round(float(d["target_lufs"]))),
              "maxpeak": str(round(float(d["true_peak_dbtp"]))), "filtering": "false",
              "denoise": "false"}
    if p.get("preset"):
        fields["preset"] = p["preset"]
    b = uuid.uuid4().hex
    body = b"".join(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
                    .encode() for k, v in fields.items())
    body += (f"--{b}\r\nContent-Disposition: form-data; name=\"input_file\"; filename=\""
             f"{src.name}\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + src.read_bytes() + \
        f"\r\n--{b}--\r\n".encode()
    prod = json.loads(_req(state, f"{API}/simple/productions.json", body,
                           f"multipart/form-data; boundary={b}"))["data"]
    uid = prod["uuid"]
    deadline = time.time() + float(p.get("timeout_s", 1800))
    while True:
        info = json.loads(_req(state, f"{API}/production/{uid}.json"))["data"]
        status = info.get("status_string", "")
        if status == "Done":
            break
        if status == "Error" or time.time() > deadline:
            raise RuntimeError(f"Auphonic production {uid}: {status or 'timeout'} "
                               f"{info.get('error_message', '')}")
        time.sleep(float(p.get("poll_s", 5)))
    files = info.get("output_files") or []
    if not files:
        raise RuntimeError(f"Auphonic production {uid} has no output files")
    ext = files[0].get("ending") or files[0].get("format") or "wav"
    raw = out.with_suffix(f".{ext}")
    raw.write_bytes(_req(state, files[0]["download_url"]))
    audio = raw
    if raw.suffix != ".wav":
        import subprocess

        audio = out.with_suffix(".wav")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                        "-ar", "48000", "-c:a", "pcm_s24le", str(audio)], check=True)
    return {"files": {"audio": str(audio)}, "params": {"production": uid},
            "_cost_usd": round(hours * float(p.get("usd_per_hour", 1.5)), 6)}


if __name__ == "__main__":
    serve(load, run)
