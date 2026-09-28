"""C5 worker (``<lang>.tn`` packs): NVIDIA nemo-text-processing WFST text normalization.

Written → spoken form with the Sparrowhawk-style grammars (Apache-2.0; 1.2.0 adds Hindi TN).
CPU only. Grammars compile on first use (minutes) into ``cache_dir``.

params: input_case (cased), cache_dir (~/.opendub/cache/nemo_tn), punct_post_process (true).
item.inputs: {text}. payload: {text}.
"""
from __future__ import annotations

import os
from pathlib import Path

import c_speech_rate as rate
from _sdk import Unsupported, serve


def load(params: dict, lang: str):
    from nemo_text_processing.text_normalization.normalize import Normalizer

    base = rate.base_lang(lang)
    cache = os.path.expanduser(params.get("cache_dir", "~/.opendub/cache/nemo_tn"))
    os.makedirs(cache, exist_ok=True)
    norm = Normalizer(input_case=params.get("input_case", "cased"), lang=base, cache_dir=cache)
    return {"norm": norm, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    if not state["lang"].endswith(".tn"):
        raise Unsupported("nemo-text-processing only does text normalization (.tn packs)")
    text = state["norm"].normalize(item["inputs"]["text"], verbose=False,
                                   punct_post_process=bool(state["params"].get(
                                       "punct_post_process", True)))
    return {"text": text}


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"nemo_text_processing": version("nemo_text_processing")}


if __name__ == "__main__":
    serve(load, run, describe)
