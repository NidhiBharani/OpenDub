"""D5 worker: Kokoro-82M v1.0 (hexgrad, Apache-2.0) stock voices via ``kokoro`` (0.9.4) + misaki
G2P (Japanese needs ``misaki[ja]``). ``KPipeline(lang_code=…, repo_id="hexgrad/Kokoro-82M")``;
``pipeline(text, voice=…, speed=…)`` yields (graphemes, phonemes, audio) chunks at 24 kHz.

params:
  voices  {lang: {female: …, male: …}} — defaults below (VOICES.md of the model repo):
          en: af_heart / am_michael, hi: hf_alpha / hm_omega, ja: jf_alpha / jm_kumo
  speed   1.0
Language codes: a (American English), b (British), j (Japanese), h (Hindi), z, e, f, i, p.
Runs on CPU as well; no cloning (the D5 point: documented, licensed voices).
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import serve

LANG_CODES = {"en": "a", "hi": "h", "ja": "j", "zh": "z", "es": "e", "fr": "f", "it": "i",
              "pt": "p"}
DEFAULT_VOICES = {"en": {"female": "af_heart", "male": "am_michael"},
                  "hi": {"female": "hf_alpha", "male": "hm_omega"},
                  "ja": {"female": "jf_alpha", "male": "jm_kumo"}}


def load(params: dict, lang: str):
    from kokoro import KPipeline

    dv.require_lang(lang, LANG_CODES, "Kokoro-82M")
    pipe = KPipeline(lang_code=LANG_CODES[lang], repo_id=params.get("model", "hexgrad/Kokoro-82M"))
    return {"pipe": pipe, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np

    p = state["params"]
    voice = dv.pick_voice(p.get("voices", DEFAULT_VOICES), state["lang"], item)
    chunks = [np.asarray(audio) for _, _, audio in
              state["pipe"](dv.text_of(item), voice=voice, speed=float(p.get("speed", 1.0)))]
    wav = dv.write_wav(dv.wav_path(out), np.concatenate(chunks), 24000)
    return dv.audio_payload(wav, voice=voice)


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"model": "hexgrad/Kokoro-82M", "kokoro": version("kokoro")}


if __name__ == "__main__":
    serve(load, run, describe)
