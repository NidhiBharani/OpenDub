"""D1/D2/D3 worker: IndexTTS2 and IndexTTS 2.5 (Bilibili, bilibili Model Use License) from the
index-tts repo (``~/.opendub/src/index-tts``, installed into this env). Verified against the repo
README (2026-09-28): 2.5 loads ``indextts.infer_v2_5.IndexTTS2`` with a ``lang`` argument
(ZH/EN/JA/ES/AR) and ``duration_factor``; 2 loads ``indextts.infer_v2.IndexTTS2`` (zh/en only).

params:
  version        "2.5" | "2"
  model_dir      checkpoints dir (default ~/.opendub/src/index-tts/checkpoints-<version>)
  use_bf16 (2.5) / use_fp16 (2), use_cuda_kernel (false), use_deepspeed (false)
  emotion        audio (use inputs.ref_audio as emo prompt: D2 source-line emotion transfer) |
                 text (use_emo_text with inputs.emotion) | none (default)
  emo_alpha      0..1 (default 0.8)
  use_target_duration  (2.5) fit inputs.target_s: one pass at factor 1.0, then a second pass at
                 duration_factor = d0 / target (README: factor = speaking speed, 0.5–2.0)
Output 22.05 kHz (the repo's BigVGAN rate).
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import serve

LANGS = {"2.5": {"zh": "ZH", "en": "EN", "ja": "JA", "es": "ES", "ar": "AR"},
         "2": {"zh": "ZH", "en": "EN"}}


def load(params: dict, lang: str):
    version = str(params.get("version", "2.5"))
    dv.require_lang(lang, LANGS[version], f"IndexTTS {version}")
    repo = dv.add_repo_to_path("index-tts")
    model_dir = params.get("model_dir") or str(repo / f"checkpoints-{version}")
    if version == "2.5":
        from indextts.infer_v2_5 import IndexTTS2

        tts = IndexTTS2(cfg_path=f"{model_dir}/config.yaml", model_dir=model_dir,
                        use_bf16=bool(params.get("use_bf16", True)),
                        use_cuda_kernel=bool(params.get("use_cuda_kernel", False)))
    else:
        from indextts.infer_v2 import IndexTTS2

        tts = IndexTTS2(cfg_path=f"{model_dir}/config.yaml", model_dir=model_dir,
                        use_fp16=bool(params.get("use_fp16", True)),
                        use_cuda_kernel=bool(params.get("use_cuda_kernel", False)),
                        use_deepspeed=bool(params.get("use_deepspeed", False)))
    return {"tts": tts, "params": params, "lang": lang, "version": version}


def _infer(state: dict, item: dict, wav: Path, factor: float | None) -> None:
    p = state["params"]
    ref, _ = dv.ref_of(item)
    kw: dict = {"spk_audio_prompt": ref, "text": dv.text_of(item), "output_path": str(wav),
                "verbose": False}
    if state["version"] == "2.5":
        kw["lang"] = LANGS["2.5"][state["lang"]]
        if factor:
            kw["duration_factor"] = max(0.5, min(2.0, factor))
    emo = p.get("emotion", "none")
    if emo == "audio":
        kw.update(emo_audio_prompt=ref, emo_alpha=float(p.get("emo_alpha", 0.8)))
    elif emo == "text" and dv.emotion_of(item):
        kw.update(use_emo_text=True, emo_text=dv.emotion_of(item),
                  emo_alpha=float(p.get("emo_alpha", 0.8)))
    state["tts"].infer(**kw)


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    seed = dv.item_seed(item, p)
    dv.seed_all(seed)
    wav = dv.wav_path(out)
    _infer(state, item, wav, None)
    tgt = dv.target_s(item)
    factor = None
    if tgt and state["version"] == "2.5" and p.get("use_target_duration", True):
        factor = dv.duration_s(wav) / tgt
        if abs(factor - 1.0) > 0.03:
            dv.seed_all(seed)
            _infer(state, item, wav, factor)
    return dv.audio_payload(wav, seed=seed, duration_factor=factor)


def describe(state: dict) -> dict:
    return {"version": state["version"], "model_dir": state["params"].get("model_dir")}


if __name__ == "__main__":
    serve(load, run, describe)
