"""A4 worker: Meta Omnilingual ASR (``omnilingual-asr``, Apache-2.0; fairseq2 backend).

params: model_card (omniASR_LLM_1B_v2, omniASR_LLM_7B_v2, omniASR_CTC_1B_v2,
omniASR_LLM_Unlimited_7B_v2 …), batch_size. Language forced with the card's
``<iso639-3>_<script>`` codes. Non-"Unlimited" cards only accept clips < 40 s. No timestamps.
fairseq2 has no confirmed aarch64 wheels, hence x86_64 only.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve

CODES = {"en": "eng_Latn", "hi": "hin_Deva", "ja": "jpn_Jpan", "zh": "cmn_Hans",
         "ko": "kor_Hang", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn",
         "ta": "tam_Taml", "te": "tel_Telu", "bn": "ben_Beng", "pt": "por_Latn"}


def load(params: dict, lang: str):
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    pipe = ASRInferencePipeline(model_card=params.get("model_card", "omniASR_LLM_1B_v2"))
    return {"pipe": pipe, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    code = CODES.get(state["lang"])
    if code is None:
        raise Unsupported(f"no omniASR language code for {state['lang']!r}")
    card = state["params"].get("model_card", "")
    dur = float(item.get("meta", {}).get("duration_s") or 0)
    if "Unlimited" not in card and dur >= 40:
        raise Unsupported("clip ≥ 40 s needs an Unlimited model card")
    text = state["pipe"].transcribe([item["inputs"]["audio"]], lang=[code],
                                    batch_size=int(state["params"].get("batch_size", 1)))[0]
    return {"text": str(text).strip(), "segments": [], "language": state["lang"]}


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"model_card": state["params"].get("model_card"),
            "omnilingual_asr": version("omnilingual-asr")}


if __name__ == "__main__":
    serve(load, run, describe)
