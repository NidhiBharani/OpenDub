"""D1/D2/D3/D5 worker: MOSS-TTS family (OpenMOSS, Apache-2.0) through transformers remote code.
Verified against the OpenMOSS-Team/MOSS-TTS-v1.5 model card quick start (2026-09-28).

params:
  model     OpenMOSS-Team/MOSS-TTS-v1.5 (8B, 31 languages) |
            OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 (4B) | OpenMOSS-Team/MOSS-VoiceGenerator
  revision  HF commit
  mode      clone (reference=[ref wav]) | design (instruction text, VoiceGenerator) | plain
  dtype     bfloat16; attn: sdpa (flash_attention_2 only if installed and useful)
  use_target_duration  pass ``tokens = round(target_s * token_rate)`` (the card's duration
            control; the audio tokenizer runs at 12.5 frames/s) when inputs.target_s is set
  token_rate 12.5; max_new_tokens 4096; generate_kwargs {…} (temperature, top_p …)
  instruction (design) voice description; "{gender}"/"{emotion}" filled from the item

``build_user_message(text=, reference=[...], language=, tokens=, instruction=)``; decoded message
``.audio_codes_list[0]`` is the waveform at ``processor.model_config.sampling_rate``.
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import serve

LANGS_V15 = ("zh", "yue", "en", "ar", "cs", "da", "de", "nl", "es", "fr", "fi", "el", "he", "hi",
             "hu", "ja", "it", "ko", "mk", "ms", "ru", "fa", "pl", "pt", "sv", "ro", "sw", "tl",
             "th", "tr", "vi")


def load(params: dict, lang: str):
    import torch
    from transformers import AutoModel, AutoProcessor

    dv.require_lang(lang, params.get("languages", LANGS_V15), "this MOSS-TTS checkpoint")
    name = params.get("model", "OpenMOSS-Team/MOSS-TTS-v1.5")
    rev = params.get("revision")
    device = params.get("device", "cuda")
    torch.backends.cuda.enable_cudnn_sdp(False)
    processor = AutoProcessor.from_pretrained(name, revision=rev, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(name, revision=rev, trust_remote_code=True,
                                      attn_implementation=params.get("attn", "sdpa"),
                                      torch_dtype=getattr(torch, params.get("dtype", "bfloat16"))
                                      ).to(device)
    model.eval()
    return {"model": model, "processor": processor, "params": params, "lang": lang,
            "device": device}


def run(state: dict, item: dict, out: Path) -> dict:
    import torch
    import torchaudio

    p, proc = state["params"], state["processor"]
    text = dv.text_of(item)
    seed = dv.item_seed(item, p)
    dv.seed_all(seed)
    msg = {"text": text, "language": dv.LANG_NAMES.get(state["lang"], state["lang"])}
    mode = p.get("mode", "clone")
    if mode == "clone":
        ref, _ = dv.ref_of(item)
        msg["reference"] = [ref]
    elif mode == "design":
        msg["instruction"] = p.get("instruction", "A natural, clear {gender} voice.").format(
            gender=dv.gender_of(item) or "female", emotion=dv.emotion_of(item) or "calm")
    tgt = dv.target_s(item)
    tokens = None
    if tgt and p.get("use_target_duration", True):
        tokens = max(1, round(tgt * float(p.get("token_rate", 12.5))))
        msg["tokens"] = tokens
    batch = proc([[proc.build_user_message(**msg)]], mode="generation")
    with torch.no_grad():
        outputs = state["model"].generate(
            input_ids=batch["input_ids"].to(state["device"]),
            attention_mask=batch["attention_mask"].to(state["device"]),
            max_new_tokens=int(p.get("max_new_tokens", 4096)), **(p.get("generate_kwargs") or {}))
    audio = next(iter(proc.decode(outputs))).audio_codes_list[0]
    wav = dv.wav_path(out)
    torchaudio.save(str(wav), audio.detach().float().cpu().unsqueeze(0),
                    proc.model_config.sampling_rate)
    return dv.audio_payload(wav, seed=seed, tokens=tokens, mode=mode)


def describe(state: dict) -> dict:
    import transformers

    p = state["params"]
    return {"model": p.get("model"), "revision": p.get("revision"),
            "transformers": transformers.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
