"""G1 worker: the "strict" CTC intelligibility judge (wav2vec2-family CTC, no language model).

A CTC model with greedy decoding and no LM cannot "repair" a misread word the way an attention
decoder with a strong prior does, so it is the round-trip judge that keeps mispronunciations
visible (docs/plans/model-ranking.md §4.7). Also usable as an A4-style ASR worker.

item.inputs: {audio, text?}. Payload: ``{"text", "segments": [], "language"}`` like the A4
workers; in ``mode: forced`` (needs ``inputs.text``, the line the take was meant to say) also
``defect_score`` = CTC negative log-likelihood of the intended text per target token (higher =
the audio supports the intended text less), computed with ``torch.nn.functional.ctc_loss`` over
all alignments, plus ``metrics: {ctc_nll: …}``. Forced scoring cannot hallucinate: it never
free-decodes.

params: model (default facebook/mms-1b-all, CC-BY-NC-4.0, 1162 languages via adapters selected
by ISO 639-3 code), revision, adapters (true for MMS: ``target_lang=<iso3>``), mode
(greedy|forced), dtype, device. Generic CTC checkpoints (e.g. an English wav2vec2-960h) set
``adapters: false``. 16 kHz input.
"""
from __future__ import annotations

import math
import unicodedata
from pathlib import Path

from _sdk import Unsupported, serve
from judge_audio import device_of, load_mono

SR = 16000
ISO3 = {"en": "eng", "hi": "hin", "ja": "jpn", "zh": "cmn-script_simplified", "ko": "kor",
        "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "pt": "por", "ru": "rus",
        "ta": "tam", "te": "tel", "bn": "ben", "ar": "ara"}


def load(params: dict, lang: str):
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    model_id = params.get("model", "facebook/mms-1b-all")
    adapters = bool(params.get("adapters", "mms" in model_id))
    kwargs: dict = {"revision": params.get("revision")}
    target = None
    if adapters:
        if lang not in ISO3:
            raise ValueError(f"no MMS adapter code for {lang!r}")
        target = params.get("target_lang", ISO3[lang])
        processor = AutoProcessor.from_pretrained(model_id, target_lang=target, **kwargs)
        model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=target,
                                               ignore_mismatched_sizes=True, **kwargs)
    else:
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        model = Wav2Vec2ForCTC.from_pretrained(model_id, **kwargs)
    device = device_of(params)
    dtype = getattr(torch, params.get("dtype", "float16" if device.startswith("cuda") else
                                      "float32"))
    model = model.to(device=device, dtype=dtype).eval()
    return {"processor": processor, "model": model, "device": device, "dtype": dtype,
            "mode": params.get("mode", "greedy"), "lang": lang, "target": target,
            "model_id": model_id, "vocab": processor.tokenizer.get_vocab()}


def _target_ids(state: dict, text: str) -> list[int]:
    """Intended text -> CTC label ids; characters outside the vocabulary are dropped (the
    vocabulary is per adapter, e.g. lower-case Latin for eng, Devanagari for hin)."""
    tok = state["processor"].tokenizer
    vocab = state["vocab"]
    text = unicodedata.normalize("NFKC", text)
    if not any(ch.isupper() for ch in vocab if len(ch) == 1):
        text = text.lower()
    kept = "".join(ch if (ch in vocab or ch == " ") else " " for ch in text)
    kept = " ".join(kept.split())
    ids = tok(kept).input_ids
    specials = {tok.pad_token_id, getattr(tok, "unk_token_id", None),
                getattr(tok, "bos_token_id", None), getattr(tok, "eos_token_id", None)}
    return [i for i in ids if i not in specials]


def run(state: dict, item: dict, out: Path) -> dict:
    import torch

    wav = load_mono(item["inputs"]["audio"], SR)
    proc, model = state["processor"], state["model"]
    feats = proc(wav, sampling_rate=SR, return_tensors="pt")
    values = feats.input_values.to(state["device"], dtype=state["dtype"])
    with torch.inference_mode():
        logits = model(values).logits.float()
    ids = torch.argmax(logits, dim=-1)[0]
    text = proc.decode(ids).strip()
    payload: dict = {"text": text, "segments": [], "language": state["lang"]}
    if state["mode"] == "forced":
        intended = item["inputs"].get("text")
        if not intended:
            raise Unsupported("forced mode needs inputs.text (the intended line)")
        target = _target_ids(state, intended)
        if not target:
            raise Unsupported("intended text has no characters in this model's vocabulary")
        logp = torch.log_softmax(logits, dim=-1)[0]            # (T, C)
        if logp.shape[0] < len(target):
            nll = 1e3  # fewer frames than labels: the take cannot contain the line
        else:
            blank = proc.tokenizer.pad_token_id  # wav2vec2 CTC uses <pad> as the blank
            loss = torch.nn.functional.ctc_loss(
                logp.unsqueeze(1), torch.tensor([target]), torch.tensor([logp.shape[0]]),
                torch.tensor([len(target)]), blank=blank, reduction="sum", zero_infinity=False)
            nll = float(loss) / len(target)
            if not math.isfinite(nll):
                nll = 1e3
        payload["defect_score"] = nll
        payload["metrics"] = {"ctc_nll": nll}
    return payload


def describe(state: dict) -> dict:
    import transformers

    return {"model": state["model_id"], "target_lang": state["target"], "mode": state["mode"],
            "transformers": transformers.__version__, "device": state["device"]}


if __name__ == "__main__":
    serve(load, run, describe)
