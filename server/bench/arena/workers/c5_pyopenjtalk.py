"""C5 worker (``ja.kana``): pyopenjtalk (OpenJTalk + NAIST-jdic, BSD/Modified-BSD) katakana
readings — the reference Japanese front end most ja TTS stacks use.

item.inputs: {text}. payload: {text (katakana), phones (OpenJTalk phone string)}.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve


def load(params: dict, lang: str):
    import pyopenjtalk

    return {"pj": pyopenjtalk, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    if not state["lang"].startswith("ja") or not state["lang"].endswith(".kana"):
        raise Unsupported("pyopenjtalk reading output is scored on ja.kana packs")
    text = item["inputs"]["text"]
    return {"text": state["pj"].g2p(text, kana=True), "phones": state["pj"].g2p(text)}


if __name__ == "__main__":
    serve(load, run)
