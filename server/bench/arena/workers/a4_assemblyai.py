"""A4/A6 worker: AssemblyAI async transcription.

``POST https://api.assemblyai.com/v2/upload`` (raw bytes) → ``upload_url``; ``POST /v2/transcript``
``{audio_url, speech_models: [...], language_code, speaker_labels?}``; poll ``GET
/v2/transcript/{id}`` until ``status`` is completed|error. ``words: [{text, start, end,
confidence, speaker}]`` in **milliseconds**. Header ``authorization: <key>``.
Checked 2026-09-28 (assemblyai.com/docs/api-reference/transcripts/submit, /pricing):
universal-3-5-pro $0.21/h (18 languages incl. hi, ja), universal-2 $0.15/h (99 languages),
speaker labels +$0.02/h.

params: speech_model, speaker_labels, usd_per_hour.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, poll, request

BASE = "https://api.assemblyai.com/v2"
PRICE = {"universal-3-5-pro": 0.21, "universal-2": 0.15}


def load(params: dict, lang: str):
    return {"key": api_key("ASSEMBLYAI_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang, hdr = state["params"], state["lang"], {"authorization": state["key"]}
    audio = item["inputs"]["audio"]
    up = request("POST", f"{BASE}/upload", data=Path(audio).read_bytes(),
                 headers={**hdr, "Content-Type": "application/octet-stream"})
    model = p.get("speech_model", "universal-3-5-pro")
    body = {"audio_url": up["upload_url"], "speech_models": [model], "language_code": lang,
            "speaker_labels": bool(p.get("speaker_labels", False))}
    job = request("POST", f"{BASE}/transcript", json_body=body, headers=hdr)

    def check():
        r = request("GET", f"{BASE}/transcript/{job['id']}", headers=hdr)
        if r["status"] == "error":
            raise RuntimeError(r.get("error", "transcription failed"))
        return r if r["status"] == "completed" else None

    res = poll(check)
    words = [{"start": w["start"] / 1000, "end": w["end"] / 1000, "word": w["text"],
              "speaker": w.get("speaker")} for w in res.get("words") or []]
    text = (res.get("text") or "").strip()
    rate = float(p.get("usd_per_hour", PRICE.get(model, 0.21))) + (
        0.02 if p.get("speaker_labels") else 0.0)
    payload = {"text": text, "language": lang,
               "segments": [{"start": words[0]["start"], "end": words[-1]["end"], "text": text,
                             "words": [{k: w[k] for k in ("start", "end", "word")}
                                       for w in words]}] if words else [],
               "_cost_usd": audio_seconds(audio) / 3600 * rate}
    if p.get("speaker_labels"):
        payload["turns"] = [{"start": u["start"] / 1000, "end": u["end"] / 1000,
                             "speaker": str(u["speaker"])} for u in res.get("utterances") or []]
    return payload


def describe(state: dict) -> dict:
    return {"speech_model": state["params"].get("speech_model", "universal-3-5-pro")}


if __name__ == "__main__":
    serve(load, run, describe)
