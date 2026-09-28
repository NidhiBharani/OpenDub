"""E2 API worker: ElevenLabs Voice Isolator (``POST /v1/audio-isolation``), a hosted denoiser —
there is no hosted bandwidth-extension API; this is the research's API alternate for E2.

Pricing: $0.12 per minute of audio (elevenlabs.io/pricing/api, checked 2026-09-28); the item cost
is ``duration_min × usd_per_min``. Output (the response body, audio bytes) is decoded to wav with
ffmpeg at whatever rate the API returns. Retries 429/5xx with backoff. Stdlib HTTP only.

params: usd_per_min (0.12), timeout_s (300).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

from _sdk import api_key, serve

URL = "https://api.elevenlabs.io/v1/audio-isolation"


def load(params: dict, lang: str):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not on PATH")
    return {"key": api_key("ELEVENLABS_API_KEY"), "p": params}


def _multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + data + \
        f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def run(state: dict, item: dict, out: Path) -> dict:
    src = Path(item["inputs"]["audio"])
    tmp_in = out.with_suffix(".in.wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                    "-c:a", "pcm_s16le", str(tmp_in)], check=True)
    with wave.open(str(tmp_in)) as w:
        minutes = w.getnframes() / w.getframerate() / 60
    body, ctype = _multipart("audio", src.name, tmp_in.read_bytes())
    tmp_in.unlink(missing_ok=True)
    data = None
    for attempt in range(5):
        req = urllib.request.Request(URL, data=body, method="POST",
                                     headers={"xi-api-key": state["key"], "Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=float(state["p"].get("timeout_s", 300))) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {exc.read()[:300]!r}") from None
    raw = out.with_suffix(".api")
    raw.write_bytes(data or b"")
    wav = out.with_suffix(".wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                    "-c:a", "pcm_s16le", str(wav)], check=True)
    raw.unlink(missing_ok=True)
    return {"files": {"audio": str(wav)},
            "_cost_usd": round(minutes * float(state["p"].get("usd_per_min", 0.12)), 6)}


def describe(state: dict) -> dict:
    return {"api": URL, "params": json.dumps(state["p"])}


if __name__ == "__main__":
    serve(load, run, describe)
