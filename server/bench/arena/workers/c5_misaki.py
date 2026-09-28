"""C5 worker (``en.g2p``, ``ja.g2p``): misaki G2P (Apache-2.0; the Kokoro front end) with its
espeak fallback disabled, so out-of-lexicon words show up as errors instead of being hidden.

misaki writes a few non-IPA symbols (diphthong capitals A I O W Y, ``ᵊ``); they are mapped to
IPA before output so WikiPron references compare fairly.

params: british (false), trf (false: no transformer POS tagger).
item.inputs: {text (a word)}. payload: {text (IPA phones, space-separated), raw}.
"""
from __future__ import annotations

from pathlib import Path

import c_speech_rate as rate
from _sdk import Unsupported, serve

MISAKI_TO_IPA = {"A": "eɪ", "I": "aɪ", "O": "oʊ", "W": "aʊ", "Y": "ɔɪ", "Q": "əʊ", "ᵊ": "ə",
                 "ʤ": "dʒ", "ʧ": "tʃ", "ᵻ": "ɨ"}


def to_ipa(phonemes: str) -> str:
    phones = []
    for ch in phonemes.replace(" ", ""):
        if ch in "ˈˌ":
            continue
        mapped = MISAKI_TO_IPA.get(ch, ch)
        if ch == "ː" and phones:
            phones[-1] += ch
            continue
        phones.append(mapped)
    return " ".join(phones)


def load(params: dict, lang: str):
    base = rate.base_lang(lang)
    if base == "en":
        from misaki import en

        g2p = en.G2P(trf=bool(params.get("trf", False)), british=bool(params.get("british")),
                     fallback=None)
    elif base == "ja":
        from misaki import ja

        g2p = ja.JAG2P()
    else:
        raise ValueError(f"misaki has no G2P for {base!r}")
    return {"g2p": g2p, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    if not state["lang"].endswith(".g2p"):
        raise Unsupported("misaki is a G2P front end (.g2p packs)")
    phonemes, _tokens = state["g2p"](item["inputs"]["text"])
    if not phonemes or "❓" in phonemes:
        raise ValueError(f"out of lexicon: {item['inputs']['text']!r}")
    return {"text": to_ipa(phonemes), "raw": phonemes}


if __name__ == "__main__":
    serve(load, run)
