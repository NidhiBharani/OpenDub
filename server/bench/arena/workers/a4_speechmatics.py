"""A4/A6 worker: Speechmatics batch API.

``POST https://asr.api.speechmatics.com/v2/jobs`` (multipart ``data_file`` + ``config`` JSON
``{"type": "transcription", "transcription_config": {"language", "operating_point":
"enhanced", "diarization": "speaker"?}}``, ``Authorization: Bearer``); poll ``GET
/v2/jobs/{id}`` until ``job.status == done``; ``GET /v2/jobs/{id}/transcript?format=json-v2`` →
``results[{type: word|punctuation, start_time, end_time, alternatives[{content, speaker}]}]``.
Docs: docs.speechmatics.com (diarization page read 2026-09-28; endpoint shape from the batch
API reference, not re-verified). Price param default $0.30/h (Pro batch per third-party
sources; the pricing page showed "$0.129" — unclear, 2026-09-28).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import api_key, serve
from a4_http import audio_seconds, poll, post_form, request, words_to_turns

BASE = "https://asr.api.speechmatics.com/v2"


def load(params: dict, lang: str):
    return {"key": api_key("SPEECHMATICS_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    hdr = {"Authorization": f"Bearer {state['key']}"}
    audio = item["inputs"]["audio"]
    tc = {"language": lang, "operating_point": p.get("operating_point", "enhanced")}
    if p.get("diarize"):
        tc["diarization"] = "speaker"
    job = post_form(f"{BASE}/jobs", {"config": {"type": "transcription",
                                                "transcription_config": tc}},
                    {"data_file": audio}, hdr)

    def check():
        r = request("GET", f"{BASE}/jobs/{job['id']}", headers=hdr)["job"]
        if r["status"] in ("rejected", "deleted", "expired"):
            raise RuntimeError(f"job {r['status']}: {r.get('errors')}")
        return r if r["status"] == "done" else None

    poll(check)
    res = request("GET", f"{BASE}/jobs/{job['id']}/transcript?format=json-v2", headers=hdr)
    words = []
    for r in res.get("results") or []:
        alt = (r.get("alternatives") or [{}])[0]
        if r.get("type") == "word":
            words.append({"start": r["start_time"], "end": r["end_time"],
                          "word": alt.get("content", ""), "speaker": alt.get("speaker")})
        elif r.get("type") == "punctuation" and words:
            words[-1]["word"] += alt.get("content", "")
    sep = "" if lang in ("ja", "zh") else " "
    text = sep.join(w["word"] for w in words).strip()
    payload = {"text": text, "language": lang,
               "segments": [{"start": words[0]["start"], "end": words[-1]["end"], "text": text,
                             "words": [{k: w[k] for k in ("start", "end", "word")}
                                       for w in words]}] if words else [],
               "_cost_usd": audio_seconds(audio) / 3600 * float(p.get("usd_per_hour", 0.30))}
    if p.get("diarize"):
        payload["turns"] = words_to_turns(words)
    return payload


def describe(state: dict) -> dict:
    return {"operating_point": state["params"].get("operating_point", "enhanced")}


if __name__ == "__main__":
    serve(load, run, describe)
