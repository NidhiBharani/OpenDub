"""Judge worker: speaker similarity (cosine of speaker embeddings) between an output and a reference.

item.inputs: {audio, ref_audio}; ``ref_audio`` may be a list (a reference set: the embeddings
are L2-normalised and averaged). Payload: ``{"metrics": {"sim_<encoder>": cos}, "score": cos}``.
``score`` is the uniform prediction the G2 meta-evaluation ranks by.

params: encoder (below), device, plus per-encoder options:

- ``wavlm-sv``: microsoft/wavlm-base-plus-sv (transformers WavLMForXVector, 16 kHz; the card's
  same-speaker threshold is 0.86). ``model`` overrides the repo id.
- ``eres2netv2``: 3D-Speaker ERes2NetV2 (ModelScope iic/speech_eres2netv2_sv_zh-cn_16k-common,
  checkpoint pretrained_eres2netv2.ckpt, 80-dim FBank with mean norm, 192-dim embedding). Needs
  the 3D-Speaker repo (``repo``, default ~/.opendub/src/3D-Speaker) for ``speakerlab``.
- ``redimnet2``: torch.hub PalabraAI/redimnet2 (``model_name`` b0–b6, ``train_type`` ptn|lm|dis,
  ``dataset`` vox2 | vb2+vox2_v0 | vb2+vox2+cnc2_v0, ``hub_ref`` = pinned commit). 16 kHz raw
  waveform (batch, samples) in, (batch, dim) out.
- ``ecapa``: speechbrain/spkrec-ecapa-voxceleb (EncoderClassifier, 16 kHz, 192-dim), the encoder
  OpenDub's pipeline already names (app/pipeline/quality.py).
- ``wespeaker-resnet293``: WeSpeaker ResNet293-LM (``model`` = local dir or HF repo with
  avg_model.pt + config.yaml; ``wespeaker.load_model``; extract_embedding_from_pcm at 16 kHz).
- ``resemblyzer``: GE2E d-vector (Resemblyzer VoiceEncoder), the hobby-pipeline floor.

Every encoder here takes 16 kHz; audio is resampled by ``judge_audio.load_mono``. Reference
embeddings are cached per path within the job.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from _sdk import serve
from judge_audio import RefCache, as_list, cosine, device_of, load_mono, mean_embedding

SR = 16000
ENCODERS = ("wavlm-sv", "eres2netv2", "redimnet2", "ecapa", "wespeaker-resnet293", "resemblyzer")
REDIMNET2_REF = "c5bbe0b76e37df698c403f8844e41304ceab6307"  # PalabraAI/redimnet2 main, 2026-08-28
ERES2NETV2_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"


def _wavlm(params: dict, device: str):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    repo = params.get("model", "microsoft/wavlm-base-plus-sv")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(repo, revision=params.get("revision"))
    model = WavLMForXVector.from_pretrained(repo, revision=params.get("revision")).to(device).eval()

    def embed(path: str):
        wav = load_mono(path, SR)
        inputs = fe(wav, sampling_rate=SR, return_tensors="pt").to(device)
        with torch.inference_mode():
            emb = model(**inputs).embeddings
        return torch.nn.functional.normalize(emb, dim=-1)[0].float().cpu().numpy()
    return embed, {"model": repo}


def _eres2netv2(params: dict, device: str):
    import torch

    repo = Path(os.path.expanduser(params.get("repo", "~/.opendub/src/3D-Speaker")))
    if not (repo / "speakerlab").is_dir():
        raise RuntimeError(f"3D-Speaker repo not found at {repo} (see bench/envs/judge_speaker.yaml)")
    sys.path.insert(0, str(repo))
    from modelscope.hub.snapshot_download import snapshot_download
    from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
    from speakerlab.process.processor import FBank

    ckpt_dir = Path(snapshot_download(params.get("model", ERES2NETV2_ID),
                                      revision=params.get("revision")))
    model = ERes2NetV2(feat_dim=80, embedding_size=192, baseWidth=26, scale=2, expansion=2)
    state = torch.load(ckpt_dir / "pretrained_eres2netv2.ckpt", map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    fbank = FBank(80, sample_rate=SR, mean_nor=True)

    def embed(path: str):
        wav = torch.from_numpy(load_mono(path, SR)).unsqueeze(0)  # (1, T) like infer_sv.py
        feat = fbank(wav).unsqueeze(0).to(device)
        with torch.inference_mode():
            return model(feat)[0].float().cpu().numpy()
    return embed, {"model": params.get("model", ERES2NETV2_ID)}


def _redimnet2(params: dict, device: str):
    import torch

    ref = params.get("hub_ref", REDIMNET2_REF)
    kwargs = {"model_name": params.get("model_name", "b6"),
              "train_type": params.get("train_type", "lm"), "pretrained": True}
    if params.get("dataset"):
        kwargs["dataset"] = params["dataset"]
    model = torch.hub.load(f"PalabraAI/redimnet2:{ref}", "redimnet2", trust_repo=True, **kwargs)
    model = model.to(device).eval()

    def embed(path: str):
        wav = torch.from_numpy(load_mono(path, SR)).unsqueeze(0).to(device)
        with torch.inference_mode():
            return model(wav)[0].float().cpu().numpy()
    return embed, {"model": f"redimnet2-{kwargs['model_name']}-{kwargs['train_type']}",
                   "hub_ref": ref, "dataset": kwargs.get("dataset", "vox2")}


def _ecapa(params: dict, device: str):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    repo = params.get("model", "speechbrain/spkrec-ecapa-voxceleb")
    save = Path.home() / ".cache" / "opendub" / "speechbrain" / repo.replace("/", "_")
    clf = EncoderClassifier.from_hparams(source=repo, savedir=str(save),
                                         run_opts={"device": device})

    def embed(path: str):
        wav = torch.from_numpy(load_mono(path, SR)).unsqueeze(0)
        with torch.inference_mode():
            return clf.encode_batch(wav).flatten().float().cpu().numpy()
    return embed, {"model": repo}


def _wespeaker(params: dict, device: str):
    import torch
    import wespeaker

    src = params.get("model", "Wespeaker/wespeaker-voxceleb-resnet293-LM")
    if not Path(os.path.expanduser(src)).is_dir():
        from huggingface_hub import snapshot_download

        src = snapshot_download(src, revision=params.get("revision"))
    model = wespeaker.load_model(os.path.expanduser(src))
    model.set_device(device)

    def embed(path: str):
        pcm = torch.from_numpy(load_mono(path, SR)).unsqueeze(0)
        return model.extract_embedding_from_pcm(pcm, SR).float().cpu().numpy()
    return embed, {"model": params.get("model", "Wespeaker/wespeaker-voxceleb-resnet293-LM")}


def _resemblyzer(params: dict, device: str):
    from resemblyzer import VoiceEncoder, preprocess_wav

    enc = VoiceEncoder(device=device)

    def embed(path: str):
        return enc.embed_utterance(preprocess_wav(load_mono(path, SR), source_sr=SR))
    return embed, {"model": "resemblyzer-ge2e"}


_LOADERS = {"wavlm-sv": _wavlm, "eres2netv2": _eres2netv2, "redimnet2": _redimnet2,
            "ecapa": _ecapa, "wespeaker-resnet293": _wespeaker, "resemblyzer": _resemblyzer}


def load(params: dict, lang: str):
    encoder = params.get("encoder", "wavlm-sv")
    if encoder not in _LOADERS:
        raise ValueError(f"unknown encoder {encoder!r}; one of {', '.join(ENCODERS)}")
    device = device_of(params)
    embed, ident = _LOADERS[encoder](params, device)
    return {"encoder": encoder, "embed": embed, "ref": RefCache(embed), "ident": ident,
            "device": device}


def run(state: dict, item: dict, out: Path) -> dict:
    inputs = item["inputs"]
    refs = as_list(inputs.get("ref_audio"))
    if not inputs.get("audio") or not refs:
        raise ValueError("speaker similarity needs inputs.audio and inputs.ref_audio")
    ref = state["ref"](refs[0]) if len(refs) == 1 else mean_embedding(refs, state["ref"])
    cos = cosine(state["embed"](inputs["audio"]), ref)
    return {"metrics": {f"sim_{state['encoder']}": cos}, "score": cos,
            "encoder": state["encoder"]}


def describe(state: dict) -> dict:
    return {"encoder": state["encoder"], "device": state["device"], **state["ident"]}


if __name__ == "__main__":
    serve(load, run, describe)
