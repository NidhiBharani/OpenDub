"""D6 worker: Vevo2 voice conversion (Amphion; checkpoints RMSnow/Vevo2, CC-BY-NC-ND-4.0).
Pipeline construction and calls copied from Amphion ``models/svc/vevo2/infer_vevo2_fm.py`` and
``infer_vevo2_ar.py`` (commit 26f6883, 2026-09-28).

params:
  mode        fm (default; timbre conversion, style preserved: ``inference_fm(src_wav_path,
              timbre_ref_wav_path, use_pitch_shift, flow_matching_steps)``) |
              ar_fm (style-converted VC: ``inference_ar_and_fm(target_text=<source transcript>,
              prosody_wav_path=src, style_ref_wav_path=ref, timbre_ref_wav_path=ref,
              use_prosody_code=True)``; needs inputs.text)
  ckpt_dir    local snapshot of RMSnow/Vevo2 (default ~/.opendub/src/Amphion/ckpts/Vevo2)
  flow_matching_steps (32), use_pitch_shift (true)
Declared languages: en zh ja ko de fr (card). ~3 GB VRAM. Output 24 kHz.
"""
from __future__ import annotations

import os
from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve


def load(params: dict, lang: str):
    import torch

    repo = dv.add_repo_to_path("Amphion")
    os.chdir(repo)  # Amphion resolves configs relative to the repo root
    from models.svc.vevo2.vevo2_utils import Vevo2InferencePipeline

    ck = Path(params.get("ckpt_dir") or repo / "ckpts" / "Vevo2")
    fm = ck / "acoustic_modeling/fm_emilia101k_singnet7k_repa"
    kw = {"content_style_tokenizer_ckpt_path": str(ck / "tokenizer/contentstyle_fvq16384_12.5hz"),
          "fmt_cfg_path": str(fm / "config.json"), "fmt_ckpt_path": str(fm),
          "vocoder_cfg_path": str(ck / "vocoder/config.json"),
          "vocoder_ckpt_path": str(ck / "vocoder"),
          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    if params.get("mode", "fm") == "ar_fm":
        kw.update(prosody_tokenizer_ckpt_path=str(ck / "tokenizer/prosody_fvq512_6.25hz"),
                  ar_cfg_path=str(ck / "contentstyle_modeling/posttrained/amphion_config.json"),
                  ar_ckpt_path=str(ck / "contentstyle_modeling/posttrained"))
    return {"pipe": Vevo2InferencePipeline(**kw), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    from models.svc.vevo2.vevo2_utils import save_audio

    p, pipe = state["params"], state["pipe"]
    src = (item.get("inputs") or {}).get("audio")
    ref, _ = dv.ref_of(item)
    if not src:
        raise Unsupported("VC item has no inputs.audio")
    seed = dv.item_seed(item, p)
    dv.seed_all(seed)
    if p.get("mode", "fm") == "ar_fm":
        text = (item.get("inputs") or {}).get("text")
        if not text:
            raise Unsupported("Vevo2 AR+FM needs the source transcript (inputs.text)")
        audio = pipe.inference_ar_and_fm(target_text=text, prosody_wav_path=src,
                                         style_ref_wav_path=ref, timbre_ref_wav_path=ref,
                                         use_prosody_code=True)
    else:
        audio = pipe.inference_fm(src_wav_path=src, timbre_ref_wav_path=ref,
                                  use_pitch_shift=bool(p.get("use_pitch_shift", True)),
                                  flow_matching_steps=int(p.get("flow_matching_steps", 32)))
    wav = dv.wav_path(out)
    save_audio(audio, output_path=str(wav))
    return dv.audio_payload(wav, seed=seed, mode=p.get("mode", "fm"))


def describe(state: dict) -> dict:
    return {"model": "RMSnow/Vevo2", "mode": state["params"].get("mode", "fm")}


if __name__ == "__main__":
    serve(load, run, describe)
