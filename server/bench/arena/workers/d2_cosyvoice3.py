"""D1/D2 worker: Fun-CosyVoice3-0.5B-2512 (FunAudioLLM, Apache-2.0) from the CosyVoice repo
(``~/.opendub/src/CosyVoice``). Calls copied from the repo's example.py (2026-09-28):

  AutoModel(model_dir=…)
  inference_zero_shot(text, 'You are a helpful assistant.<|endofprompt|>' + ref_text, ref_wav)
  inference_cross_lingual('You are a helpful assistant.<|endofprompt|>' + text, ref_wav)
  inference_instruct2(text, 'You are a helpful assistant. <instruction><|endofprompt|>', ref_wav)

params:
  model_dir   pretrained_models/Fun-CosyVoice3-0.5B (relative to the repo)
  mode        auto (zero_shot when the reference is in the target language and has a transcript,
              else cross_lingual) | zero_shot | cross_lingual | instruct
  instruction (instruct) e.g. "Please speak in a {emotion} tone." ({emotion} from the item)
  ja_katakana convert Japanese text to katakana with pyopenjtalk (the example says ja input
              must be katakana); default true

Languages: zh en ja ko de es fr it ru (+ Chinese dialects); no Hindi. Output 24 kHz.
CAM++-family speaker embedding inside: never judge CosyVoice with a CAM++ SIM encoder.
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import serve

LANGS = ("zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru")
SYS = "You are a helpful assistant."


def load(params: dict, lang: str):
    dv.require_lang(lang, LANGS, "CosyVoice 3")
    repo = dv.add_repo_to_path("CosyVoice", "third_party/Matcha-TTS")
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(model_dir=str(repo / params.get("model_dir",
                                                      "pretrained_models/Fun-CosyVoice3-0.5B")))
    kana = None
    if lang == "ja" and params.get("ja_katakana", True):
        import pyopenjtalk

        kana = pyopenjtalk
    return {"model": model, "params": params, "lang": lang, "kana": kana}


def run(state: dict, item: dict, out: Path) -> dict:
    import torch

    p, model = state["params"], state["model"]
    text = dv.text_of(item)
    if state["kana"] is not None:
        text = state["kana"].g2p(text, kana=True)
    ref, ref_text = dv.ref_of(item)
    seed = dv.item_seed(item, p)
    dv.seed_all(seed)
    mode = p.get("mode", "auto")
    if mode == "auto":
        same = (dv.src_lang(item) or state["lang"]) == state["lang"]
        mode = "zero_shot" if same and ref_text else "cross_lingual"
    if mode == "zero_shot":
        gen = model.inference_zero_shot(text, f"{SYS}<|endofprompt|>{ref_text}", ref,
                                        stream=False)
    elif mode == "instruct":
        instr = p.get("instruction", "Please speak in a {emotion} tone.").format(
            emotion=dv.emotion_of(item) or "natural")
        gen = model.inference_instruct2(text, f"{SYS} {instr}<|endofprompt|>", ref, stream=False)
    else:
        gen = model.inference_cross_lingual(f"{SYS}<|endofprompt|>{text}", ref, stream=False)
    speech = torch.cat([j["tts_speech"] for j in gen], dim=1)
    wav = dv.write_wav(dv.wav_path(out), speech, model.sample_rate)
    return dv.audio_payload(wav, seed=seed, mode=mode)


def describe(state: dict) -> dict:
    return {"model_dir": state["params"].get("model_dir"), "mode": state["params"].get("mode")}


if __name__ == "__main__":
    serve(load, run, describe)
