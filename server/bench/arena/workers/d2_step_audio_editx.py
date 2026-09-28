"""D1/D2/D7 worker: Step-Audio-EditX (StepFun, 3B, Apache-2.0) from its repo
(``~/.opendub/src/Step-Audio-EditX``). Load and call signatures copied from the repo's
tts_infer.py (commit a652e87, 2026-09-28): ``StepAudioTokenizer(path, model_source=…)``,
``StepAudioTTS(model_path, tokenizer, …vLLM args…)``, ``.clone(prompt_wav_path, prompt_text,
target_text)`` and ``.edit(prompt_wav_path, prompt_text, edit_type, edit_info, target_text)``.

params:
  model_path / tokenizer_path  default ~/.opendub/src/Step-Audio-EditX/{Step-Audio-EditX,
                               Step-Audio-Tokenizer} (cloned by the env recipe)
  mode             clone | clone+emotion (clone, then edit the take with inputs.emotion; D2) |
                   clone+paralinguistic (clone text without tags, then an edit pass whose
                   target_text keeps the [Laughter]-style tags; D7)
  edit_iterations  edit passes (README: accuracy rises to iteration 3); default 1
  gpu_memory_utilization (0.5), max_model_len (3072), dtype bfloat16, quantization (None|awq)

Languages: zh, en, ja, ko (+ Sichuanese, Cantonese). No Hindi. 12–16 GB VRAM (AWQ 4-bit 6–8 GB).
"""
from __future__ import annotations

import re
from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve

LANGS = ("zh", "en", "ja", "ko")
EMOTIONS = {"happy", "angry", "sad", "fear", "surprised", "confusion", "empathy", "embarrass",
            "excited", "depressed", "admiration", "coldness", "disgusted", "humour"}
TAG = re.compile(r"\[[^\]]+\]")


def load(params: dict, lang: str):
    dv.require_lang(lang, LANGS, "Step-Audio-EditX")
    repo = dv.add_repo_to_path("Step-Audio-EditX")
    from tokenizer import StepAudioTokenizer
    from tts import StepAudioTTS

    tok_path = Path(params.get("tokenizer_path", repo / "Step-Audio-Tokenizer")).expanduser()
    model_path = Path(params.get("model_path", repo / "Step-Audio-EditX")).expanduser()
    tok = StepAudioTokenizer(str(tok_path), model_source=params.get("model_source", "auto"))
    model = StepAudioTTS(str(model_path), tok,
                         model_source=params.get("model_source", "auto"),
                         tts_model_id=params.get("tts_model_id"),
                         quantization=params.get("quantization"), tensor_parallel_size=1,
                         gpu_memory_utilization=float(params.get("gpu_memory_utilization", 0.5)),
                         max_model_len=int(params.get("max_model_len", 3072)),
                         enforce_eager=bool(params.get("enforce_eager", False)),
                         dtype=params.get("dtype", "bfloat16"), kv_cache_dtype=None,
                         max_num_seqs=1, max_num_batched_tokens=None,
                         cosyvoice_dtype=params.get("cosyvoice_dtype", "bfloat16"),
                         cosyvoice_cuda_graph=bool(params.get("cosyvoice_cuda_graph", True)))
    return {"model": model, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    import torchaudio

    p, model = state["params"], state["model"]
    text = dv.text_of(item)
    ref, ref_text = dv.ref_of(item)
    if not ref_text:
        raise Unsupported("Step-Audio-EditX cloning needs the reference transcript")
    seed = dv.item_seed(item, p)
    dv.seed_all(seed)
    mode = p.get("mode", "clone")
    plain = TAG.sub("", text).strip() if mode == "clone+paralinguistic" else text
    audio, sr = model.clone(prompt_wav_path=ref, prompt_text=ref_text, target_text=plain)
    wav = dv.wav_path(out)
    torchaudio.save(str(wav), audio.cpu(), sr)
    edits = []
    if mode == "clone+emotion":
        emo = dv.emotion_of(item).lower()
        if emo in EMOTIONS:
            edits = [("emotion", emo, plain)]
    elif mode == "clone+paralinguistic" and TAG.search(text):
        edits = [("paralinguistic", "", text)]
    for edit_type, info, tgt_text in edits * int(p.get("edit_iterations", 1)):
        audio, sr = model.edit(prompt_wav_path=str(wav), prompt_text=plain, edit_type=edit_type,
                               edit_info=info, target_text=tgt_text)
        torchaudio.save(str(wav), audio.cpu(), sr)
    return dv.audio_payload(wav, seed=seed, mode=mode, edits=len(edits))


def describe(state: dict) -> dict:
    return {"model_path": state["params"].get("model_path"), "mode": state["params"].get("mode")}


if __name__ == "__main__":
    serve(load, run, describe)
