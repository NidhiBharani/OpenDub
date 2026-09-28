"""D8 worker (evaluate-only): UniSS (cmots/UniSS, ICLR 2026, CC-BY-4.0), expressive S2ST with
voice/emotion preservation. Code copied from the model card usage (2026-09-28):
``UniSSTokenizer.from_pretrained``, ``tokenize(wav)``, ``process_input(glm4, bicodec, mode,
tgt_lang)``, ``model.generate(...)``, ``process_output(...)`` → (audio 16 kHz, translation,
transcription).

Only English ↔ Chinese (``<|eng|>`` / ``<|cmn|>``): registered disabled for the en/hi/ja scope
(ja/hi sources are Unsupported); kept so a zh pack can use it.
params: model_path (default ~/.opendub/src/UniSS/pretrained_models/UniSS), mode (Quality |
Performance), max_new_tokens (1500), temperature (0.7), top_p (0.8), repetition_penalty (1.1).
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve

TGT = {"en": "<|eng|>", "zh": "<|cmn|>"}
SRC = {"en", "zh"}


def load(params: dict, lang: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dv.require_lang(lang, TGT, "UniSS")
    repo = dv.add_repo_to_path("UniSS")
    from uniss import UniSSTokenizer

    path = params.get("model_path") or str(repo / "pretrained_models" / "UniSS")
    device = torch.device("cuda")
    return {"model": AutoModelForCausalLM.from_pretrained(path, device_map=device),
            "tok": AutoTokenizer.from_pretrained(path),
            "speech_tok": UniSSTokenizer.from_pretrained(path, device=device),
            "device": device, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    from uniss import process_input, process_output

    p = state["params"]
    src = (item.get("inputs") or {}).get("audio")
    if not src or (dv.src_lang(item) or "en") not in SRC:
        raise Unsupported("UniSS translates only between English and Chinese")
    dv.seed_all(dv.item_seed(item, p))
    mode = p.get("mode", "Quality")
    glm4, bicodec = state["speech_tok"].tokenize(src)
    prompt = process_input(glm4, bicodec, mode, TGT[state["lang"]])
    ids = state["tok"].encode(prompt, return_tensors="pt").to(state["device"])
    gen = state["model"].generate(ids, max_new_tokens=int(p.get("max_new_tokens", 1500)),
                                  temperature=float(p.get("temperature", 0.7)),
                                  top_p=float(p.get("top_p", 0.8)),
                                  repetition_penalty=float(p.get("repetition_penalty", 1.1)))
    text_out = state["tok"].batch_decode(gen, skip_special_tokens=True)[0]
    audio, translation, transcription = process_output(text_out, prompt, state["speech_tok"],
                                                       mode, state["device"])
    wav = dv.write_wav(dv.wav_path(out), audio, 16000)
    return {**dv.audio_payload(wav), "text": translation, "source_text": transcription}


if __name__ == "__main__":
    serve(load, run)
