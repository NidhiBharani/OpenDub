"""C5 worker (``<lang>.g2p``): CharsiuG2P multilingual ByT5 (charsiu/g2p_multilingual_byT5_*;
~100 languages incl. Hindi, where misaki has nothing). Input is ``"<code>: word"`` with the
Charsiu language code; output is an IPA string (the spec splits it into phones).

params: model (charsiu/g2p_multilingual_byT5_small_100), tokenizer (google/byt5-small),
codes ({en: eng-us, hi: hin, ja: jpn}), num_beams (1), device (cuda).
item.inputs: {text (a word)}. payload: {text}.
"""
from __future__ import annotations

from pathlib import Path

import c_speech_rate as rate
from _sdk import Unsupported, serve

CODES = {"en": "eng-us", "hi": "hin", "ja": "jpn"}


def load(params: dict, lang: str):
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    base = rate.base_lang(lang)
    code = {**CODES, **params.get("codes", {})}.get(base)
    if not code:
        raise ValueError(f"no CharsiuG2P code for {base!r}")
    tok = AutoTokenizer.from_pretrained(params.get("tokenizer", "google/byt5-small"))
    model = T5ForConditionalGeneration.from_pretrained(
        params.get("model", "charsiu/g2p_multilingual_byT5_small_100"),
        revision=params.get("revision"))
    model.to(params.get("device", "cuda")).eval()
    return {"tok": tok, "model": model, "code": code, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    import torch

    if not state["lang"].endswith(".g2p"):
        raise Unsupported("CharsiuG2P is word-level G2P (.g2p packs)")
    enc = state["tok"]([f"<{state['code']}>: {item['inputs']['text']}"], return_tensors="pt",
                       padding=True, add_special_tokens=False).to(state["model"].device)
    with torch.inference_mode():
        ids = state["model"].generate(**enc, num_beams=int(state["params"].get("num_beams", 1)),
                                      max_length=64)
    return {"text": state["tok"].batch_decode(ids, skip_special_tokens=True)[0].strip()}


if __name__ == "__main__":
    serve(load, run)
