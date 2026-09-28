"""D1/D2/D4 worker: Fish Audio hosted API (S2 family).

POST https://api.fish.audio/v1/tts, ``Authorization: Bearer``, header ``model`` ∈ {s1, s2-pro,
s2.1-pro, …} (docs.fish.audio, read 2026-09-28). Inline zero-shot ``references`` (audio bytes +
transcript) are accepted only as MessagePack, so the body is msgpack
{text, references: [{audio, text}], format: wav, temperature, top_p, normalize}.

params: model (s2.1-pro), temperature (0.7), top_p (0.7), emotion_tags ("[<emotion>]" prefix),
usd_per_million_bytes (15 — the Fish API list price per million UTF-8 bytes; NOT re-verified).
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import api_key, serve


def load(params: dict, lang: str):
    return {"key": api_key("FISH_AUDIO_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    emo = dv.emotion_of(item)
    if p.get("emotion_tags") and emo:
        text = f"[{emo.lower()}] {text}"
    ref, ref_text = dv.ref_of(item)
    req = {"text": text, "format": "wav", "normalize": True,
           "temperature": float(p.get("temperature", 0.7)), "top_p": float(p.get("top_p", 0.7)),
           "references": [{"audio": Path(ref).read_bytes(), "text": ref_text}]}
    raw, _ = dv.http("POST", "https://api.fish.audio/v1/tts", body=dv.msgpack_dumps(req),
                     headers={"Authorization": f"Bearer {state['key']}",
                              "Content-Type": "application/msgpack",
                              "model": p.get("model", "s2.1-pro")}, timeout=300)
    wav = dv.transcode_to_wav(raw, dv.wav_path(out), ".wav")
    cost = round(len(text.encode()) * float(p.get("usd_per_million_bytes", 15.0)) / 1e6, 6)
    return {**dv.audio_payload(wav), "_cost_usd": cost}


if __name__ == "__main__":
    serve(load, run)
