"""C5 worker (``<lang>.g2p``): espeak-ng IPA, called as a subprocess (GPL-3.0: shelled out,
never linked). Requires the ``espeak-ng`` binary on PATH; stdlib only (``server`` env).

params: voices ({en: en-us, hi: hi, ja: ja}), binary (espeak-ng).
item.inputs: {text}. payload: {text (IPA phones, space-separated)}.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import c_speech_rate as rate
from _sdk import Unsupported, serve

VOICES = {"en": "en-us", "hi": "hi", "ja": "ja"}


def load(params: dict, lang: str):
    base = rate.base_lang(lang)
    voice = {**VOICES, **params.get("voices", {})}.get(base)
    if not voice:
        raise ValueError(f"no espeak-ng voice for {base!r}")
    return {"voice": voice, "bin": params.get("binary", "espeak-ng"), "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    if not state["lang"].endswith(".g2p"):
        raise Unsupported("espeak-ng is used for G2P only (.g2p packs)")
    proc = subprocess.run([state["bin"], "-q", f"-v{state['voice']}", "--ipa", "--sep= ",
                           item["inputs"]["text"]], capture_output=True, text=True, timeout=60,
                          check=True)
    ipa = " ".join(proc.stdout.replace("ˈ", "").replace("ˌ", "").split())
    return {"text": ipa}


def describe(state: dict) -> dict:
    ver = subprocess.run([state["bin"], "--version"], capture_output=True, text=True,
                         check=False).stdout.strip()
    return {"espeak_ng": ver, "voice": state["voice"]}


if __name__ == "__main__":
    serve(load, run, describe)
