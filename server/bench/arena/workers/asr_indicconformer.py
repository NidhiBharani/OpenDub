"""A4 worker: AI4Bharat IndicConformer-600M multilingual (ONNX + TorchScript), CTC decoding.

params: model (HF repo id), revision, decoding (ctc), threads (onnxruntime intra-op threads).

The model card loads ``model_onnx.py`` via ``trust_remote_code``; that class builds a session for
all 22 RNNT joint heads and silently falls back to CPU when the CUDA execution provider fails to
load. This worker reproduces the card's CTC path (model_onnx.py, git blob a9814bd at repo commit
e9b71b3): TorchScript preprocessor -> encoder.onnx -> ctc_decoder.onnx -> per-language vocab
mask -> greedy collapse, plus the card's CTC word timestamps. It asserts CUDAExecutionProvider.

onnxruntime-gpu dlopens the CUDA/cuDNN libs by bare soname; ``import torch`` first puts torch's
pip-installed nvidia libs in the process so the CUDA EP can load (the env pairs a cu130 torch
with the CUDA 13 onnxruntime-gpu wheel).
"""
from __future__ import annotations

import json
from pathlib import Path

from _sdk import serve

FRAME_S = 0.08  # 8x subsampled 10 ms frames; model_onnx.py default (config has no override)
BLANK_ID = 256


def load(params: dict, lang: str):
    import torch  # must precede onnxruntime so the CUDA libs resolve

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to torch")
    import onnxruntime as ort
    from huggingface_hub import snapshot_download

    if params.get("decoding", "ctc") != "ctc":
        raise ValueError("only CTC decoding is implemented")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"onnxruntime has no CUDA EP: {ort.get_available_providers()}")

    root = Path(snapshot_download(params.get("model", "ai4bharat/indic-conformer-600m-multilingual"),
                                  revision=params.get("revision"),
                                  allow_patterns=["assets/*", "config.json"])) / "assets"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(params.get("threads", 4))  # explicit: avoids affinity errors
    opts.inter_op_num_threads = 1
    cuda = ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"})
    sessions = {}
    for name in ("encoder", "ctc_decoder"):
        sess = ort.InferenceSession(str(root / f"{name}.onnx"), sess_options=opts,
                                    providers=[cuda, "CPUExecutionProvider"])
        if sess.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{name}.onnx fell back to {sess.get_providers()}; "
                               "CUDA EP failed to load (check CUDA/cuDNN libs)")
        sessions[name] = sess

    vocab = json.loads((root / "vocab.json").read_text())
    masks = json.loads((root / "language_masks.json").read_text())
    if lang not in vocab or lang not in masks:
        raise ValueError(f"IndicConformer has no vocabulary for {lang!r}")
    device = torch.device("cuda")
    pre = torch.jit.load(str(root / "preprocessor.ts"), map_location=device)
    return {"pre": pre, "sessions": sessions, "vocab": vocab[lang], "mask": masks[lang],
            "device": device, "lang": lang, "params": params, "root": str(root)}


def _audio(path: str):
    import numpy as np
    import soundfile as sf
    import torch

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(wav.mean(axis=1)))[None]
    if sr != 16000:
        import torchaudio.functional as F

        wav = F.resample(wav, sr, 16000)
    return wav


def run(state: dict, item: dict, out: Path) -> dict:
    import torch

    wav = _audio(item["inputs"]["audio"]).to(state["device"])
    with torch.inference_mode():
        feats, length = state["pre"](input_signal=wav,
                                     length=torch.tensor([wav.shape[-1]], device=state["device"]))
    enc, enc_len = state["sessions"]["encoder"].run(
        ["outputs", "encoded_lengths"],
        {"audio_signal": feats.cpu().numpy(), "length": length.cpu().numpy()})
    logits = state["sessions"]["ctc_decoder"].run(["logprobs"], {"encoder_output": enc})[0]
    logprobs = torch.from_numpy(logits[:, :, state["mask"]]).log_softmax(dim=-1)

    vocab = state["vocab"]
    n = int(enc_len[0])
    path = logprobs[0, :n].argmax(dim=-1).tolist()
    text = "".join(vocab[t] for t in _collapse(path) if t != BLANK_ID).replace("▁", " ").strip()
    words = _words(path, vocab, n)
    segments = ([{"start": words[0]["start"], "end": words[-1]["end"], "text": text,
                  "words": words}] if words else [])
    return {"text": " ".join(text.split()), "segments": segments, "language": state["lang"]}


def _collapse(path: list[int]) -> list[int]:
    return [t for i, t in enumerate(path) if i == 0 or t != path[i - 1]]


def _words(path: list[int], vocab: list[str], n: int) -> list[dict]:
    """Greedy-CTC token spans merged at '▁' word starts (model_onnx.compute_timestamps 'w')."""
    spans, cur, start = [], None, 0
    for f, tok in enumerate(path):
        if tok == BLANK_ID:
            if cur is not None:
                spans.append((vocab[cur], start * FRAME_S, f * FRAME_S))
                cur = None
        elif tok != cur:
            if cur is not None:
                spans.append((vocab[cur], start * FRAME_S, f * FRAME_S))
            cur, start = tok, f
    if cur is not None:
        spans.append((vocab[cur], start * FRAME_S, n * FRAME_S))

    words: list[dict] = []
    for tok, t0, t1 in spans:
        if "▁" in tok or not words:
            piece = tok.replace("▁", "")
            words.append({"start": round(t0, 3), "end": round(t1, 3), "word": piece})
        else:
            words[-1]["word"] += tok
            words[-1]["end"] = round(t1, 3)
    return [w for w in words if w["word"]]


def describe(state: dict) -> dict:
    import onnxruntime as ort

    p = state["params"]
    return {"model": p.get("model"), "revision": p.get("revision"), "decoding": "ctc",
            "onnxruntime": ort.__version__,
            "providers": state["sessions"]["encoder"].get_providers(),
            "snapshot": state["root"]}


if __name__ == "__main__":
    serve(load, run, describe)
