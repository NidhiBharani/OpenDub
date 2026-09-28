"""A8 worker: transformers speech-emotion models.

params: family = classifier (``AutoModelForAudioClassification``: superb/hubert-large-superb-er,
the English IEMOCAP model OpenDub uses today — labels neu/hap/ang/sad) | msp_dim
(audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim, CC BY-NC-SA 4.0: the card's custom
RegressionHead over mean-pooled hidden states → arousal, dominance, valence ≈ [0, 1]); model,
revision. 16 kHz.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a8_llm import duration


def _msp_model(mid: str, rev: str | None):
    import torch
    from torch import nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            return self.out_proj(self.dropout(x))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return hidden, self.classifier(hidden)

    return EmotionModel.from_pretrained(mid, revision=rev)


def load(params: dict, lang: str):
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    fam = params.get("family", "classifier")
    mid, rev = params["model"], params.get("revision")
    fe = AutoFeatureExtractor.from_pretrained(mid, revision=rev)
    model = (_msp_model(mid, rev) if fam == "msp_dim"
             else AutoModelForAudioClassification.from_pretrained(mid, revision=rev))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {"model": model.to(dev).eval(), "fe": fe, "dev": dev, "family": fam}


def run(state: dict, item: dict, out: Path) -> dict:
    import librosa
    import torch

    audio = item["inputs"]["audio"]
    y, _ = librosa.load(audio, sr=16000, mono=True)
    x = state["fe"](y, sampling_rate=16000, return_tensors="pt").to(state["dev"])
    dur = duration(audio)
    with torch.no_grad():
        if state["family"] == "msp_dim":
            _, pred = state["model"](x["input_values"])
            a, d, v = (float(t) for t in pred[0].cpu())
            return {"segments": [], "dims": {"arousal": a, "dominance": d, "valence": v}}
        probs = state["model"](**x).logits.softmax(-1)[0].cpu().tolist()
    labels = state["model"].config.id2label
    ranked = sorted(((labels[k], pr) for k, pr in enumerate(probs)), key=lambda t: -t[1])
    return {"segments": [{"start": 0.0, "end": dur, "label": lab, "score": pr}
                         for lab, pr in ranked], "emotion": ranked[0][0]}


if __name__ == "__main__":
    serve(load, run)
