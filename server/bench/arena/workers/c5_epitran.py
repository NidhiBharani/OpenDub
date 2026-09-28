"""C5 worker (``hi.g2p``, ``ja.g2p``): Epitran rule-based transliteration to IPA (MIT; 190+
language-script pairs; Hindi via hin-Deva, Japanese kana via jpn-Hira / jpn-Kana).

params: codes ({hi: hin-Deva, ja: jpn-Hira}).
item.inputs: {text (a word)}. payload: {text (IPA phones, space-separated)}.
"""
from __future__ import annotations

from pathlib import Path

import c_speech_rate as rate
from _sdk import Unsupported, serve

CODES = {"hi": "hin-Deva", "ja": "jpn-Hira"}


def load(params: dict, lang: str):
    import epitran

    base = rate.base_lang(lang)
    code = {**CODES, **params.get("codes", {})}.get(base)
    if not code:
        raise ValueError(f"no Epitran code for {base!r}")
    return {"epi": epitran.Epitran(code), "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    if not state["lang"].endswith(".g2p"):
        raise Unsupported("Epitran is word-level G2P (.g2p packs)")
    phones = state["epi"].trans_list(item["inputs"]["text"])
    return {"text": " ".join(p for p in phones if p.strip())}


if __name__ == "__main__":
    serve(load, run)
