"""D2/D3/D4/D5 worker: Azure AI Speech neural / HD / MAI-Voice-2 prebuilt voices over REST SSML.

POST https://{region}.tts.speech.microsoft.com/cognitiveservices/v1, headers
``Ocp-Apim-Subscription-Key``, ``Content-Type: application/ssml+xml``,
``X-Microsoft-OutputFormat: riff-24khz-16bit-mono-pcm``. Verified against the MAI-Voice doc
(learn.microsoft.com …/speech-service/mai-voices, updated 2026-07-23): MAI voices are named
``<locale>-<Name>:MAI-Voice-2`` (e.g. en-US-Harper:MAI-Voice-2, hi-IN-Kavya:MAI-Voice-2) on the
same endpoint; styles via ``<mstts:express-as style=…>``. MAI-Voice-2 has NO ja-JP voice (use a
ja-JP neural voice). Instant cloning is gated (Limited Access) and not used here.

params: voices {lang: {female, male}}, style (express-as from inputs.emotion when set: D2),
rate_to_target (D3: SSML <prosody rate> from a first pass so the take fits inputs.target_s),
usd_per_million_chars (22 MAI-Voice-2; 16 standard neural per the Azure pricing page as
summarised 2026-09-28 — estimates).
"""
from __future__ import annotations

import os
from pathlib import Path
from xml.sax.saxutils import escape

import d_voice as dv
from _sdk import api_key, serve


def load(params: dict, lang: str):
    region = params.get("region") or os.environ.get("AZURE_SPEECH_REGION") or "eastus"
    return {"key": api_key("AZURE_SPEECH_KEY"), "region": region, "params": params, "lang": lang}


def _ssml(voice: str, locale: str, text: str, style: str = "", rate: float | None = None) -> str:
    body = escape(text)
    if rate:
        body = f'<prosody rate="{(rate - 1) * 100:+.0f}%">{body}</prosody>'
    if style:
        body = f'<mstts:express-as style="{escape(style)}">{body}</mstts:express-as>'
    return (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='{locale}'>"
            f"<voice name='{escape(voice)}'>{body}</voice></speak>")


def _synth(state: dict, ssml: str, wav: Path) -> None:
    raw, _ = dv.http("POST", f"https://{state['region']}.tts.speech.microsoft.com/"
                     "cognitiveservices/v1", body=ssml.encode(),
                     headers={"Ocp-Apim-Subscription-Key": state["key"],
                              "Content-Type": "application/ssml+xml",
                              "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                              "User-Agent": "opendub-arena"}, timeout=300)
    wav.write_bytes(raw)


def run(state: dict, item: dict, out: Path) -> dict:
    p, lang = state["params"], state["lang"]
    text = dv.text_of(item)
    voice = dv.pick_voice(p.get("voices"), lang, item)
    locale = voice.split("-")[0] + "-" + voice.split("-")[1] if voice.count("-") >= 2 else \
        dv.LOCALES.get(lang, "en-US")
    style = dv.emotion_of(item).lower() if p.get("style") else ""
    wav = dv.wav_path(out)
    _synth(state, _ssml(voice, locale, text, style), wav)
    calls, rate = 1, None
    tgt = dv.target_s(item)
    if tgt and p.get("rate_to_target"):
        rate = max(0.5, min(2.0, dv.duration_s(wav) / tgt))
        if abs(rate - 1) > 0.03:
            _synth(state, _ssml(voice, locale, text, style, rate), wav)
            calls = 2
    usd = float(p.get("usd_per_million_chars", 22.0 if "MAI-Voice" in voice else 16.0))
    return {**dv.audio_payload(wav, voice=voice, rate=rate),
            "_cost_usd": dv.char_cost(text, usd) * calls}


if __name__ == "__main__":
    serve(load, run)
