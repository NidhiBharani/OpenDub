"""D1/D4 worker: IndicF5 (ai4bharat/IndicF5, MIT, gated: accept the terms on HF and set HF_TOKEN).
The model card's API: ``AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)``
then ``audio = model(text, ref_audio_path=…, ref_text=…)`` → numpy (int16 or float) at 24 kHz.
The remote code imports the ``f5_tts`` fork installed from github.com/ai4bharat/IndicF5, so it
gets its own env (it would clash with the server's f5-tts).

params: model, revision, seed via the item (torch/numpy seeded before each call).
Languages: as bn gu hi kn ml mr or pa ta te. The reference transcript should be in the same
script family as the checkpoint's vocab; a Japanese/English reference transcript is passed as-is.
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import serve

LANGS = ("as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te")


def load(params: dict, lang: str):
    import torch
    from transformers import AutoModel

    dv.require_lang(lang, LANGS, "IndicF5")
    model = AutoModel.from_pretrained(params.get("model", "ai4bharat/IndicF5"),
                                      revision=params.get("revision"), trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return {"model": model, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np

    text = dv.text_of(item)
    ref, ref_text = dv.ref_of(item)
    seed = dv.item_seed(item, state["params"])
    dv.seed_all(seed)
    audio = state["model"](text, ref_audio_path=ref, ref_text=ref_text)
    audio = np.asarray(audio)
    wav = dv.write_wav(dv.wav_path(out), audio, 24000)
    return dv.audio_payload(wav, seed=seed)


def describe(state: dict) -> dict:
    return {"model": state["params"].get("model", "ai4bharat/IndicF5"),
            "revision": state["params"].get("revision")}


if __name__ == "__main__":
    serve(load, run, describe)
