"""D1/D4/D5/D6 worker: Resemble AI Chatterbox (MIT) via ``chatterbox-tts``.
Verified against github.com/resemble-ai/chatterbox README (2026-09-28).

params:
  mode          tts (ChatterboxMultilingualTTS) | vc (ChatterboxVC: inputs.audio → voice of
                inputs.ref_audio) | turbo (ChatterboxTurboTTS, English, paralinguistic tags)
  t3_model      multilingual T3 version ("v3" per the README); omitted = package default
  exaggeration (0.5), cfg_weight (0.5), temperature (0.8)
  stock_voices  (D5) {lang: {female: /abs/ref.wav, male: …}}: a licensed reference library
                instead of the item's reference (the research "licensed reference library" use)

Multilingual language ids (23): ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr
zh. Output 24 kHz (``model.sr``).
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import serve

MTL_LANGS = ("ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms",
             "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh")


def load(params: dict, lang: str):
    import torch

    device = params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    mode = params.get("mode", "tts")
    if mode == "vc":
        from chatterbox.vc import ChatterboxVC

        model = ChatterboxVC.from_pretrained(device=device)
    elif mode == "turbo":
        dv.require_lang(lang, ("en",), "Chatterbox Turbo")
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS.from_pretrained(device=device)
    else:
        dv.require_lang(lang, MTL_LANGS, "Chatterbox Multilingual")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        kw = {"t3_model": params["t3_model"]} if params.get("t3_model") else {}
        model = ChatterboxMultilingualTTS.from_pretrained(device=device, **kw)
    return {"model": model, "params": params, "lang": lang, "mode": mode}


def run(state: dict, item: dict, out: Path) -> dict:
    import torchaudio

    p, model, mode = state["params"], state["model"], state["mode"]
    seed = dv.item_seed(item, p)
    dv.seed_all(seed)
    if mode == "vc":
        src = (item.get("inputs") or {}).get("audio")
        ref, _ = dv.ref_of(item)
        wav = model.generate(audio=src, target_voice_path=ref)
    else:
        text = dv.text_of(item)
        if p.get("stock_voices"):
            ref = dv.pick_voice(p["stock_voices"], state["lang"], item)
        else:
            ref, _ = dv.ref_of(item)
        kw = {"audio_prompt_path": ref}
        if mode == "tts":
            kw.update(language_id=state["lang"], exaggeration=float(p.get("exaggeration", 0.5)),
                      cfg_weight=float(p.get("cfg_weight", 0.5)),
                      temperature=float(p.get("temperature", 0.8)))
        wav = model.generate(text, **kw)
    path = dv.wav_path(out)
    torchaudio.save(str(path), wav.detach().cpu().reshape(1, -1), model.sr)
    return dv.audio_payload(path, seed=seed, mode=mode)


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"mode": state["mode"], "t3_model": state["params"].get("t3_model"),
            "chatterbox_tts": version("chatterbox-tts")}


if __name__ == "__main__":
    serve(load, run, describe)
