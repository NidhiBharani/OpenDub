"""Judge worker: emotion consistency between an output and its source line (cross-lingual).

item.inputs: {audio, ref_audio} (``ref_audio`` = the source-language line). Payload:
``{"metrics": {...}, "score": <similarity>, "avd": {...}}``; ``score`` is the uniform prediction
the G4 meta-evaluation ranks by (higher = same feeling).

params.model:

- ``emotion2vec`` (default): emotion2vec+ large through FunASR (``AutoModel(model=
  "iic/emotion2vec_plus_large")``, ``generate(..., granularity="utterance",
  extract_embedding=True)``; 16 kHz; 9 classes angry, disgusted, fearful, happy, neutral, other,
  sad, surprised, unknown). Metrics ``emo_sim_emotion2vec`` (utterance-embedding cosine),
  ``emo_post_sim_emotion2vec`` (cosine of the class posteriors), ``emo_label_match``.
  With ``avd: true`` (the default, so ``judgelib.emotion_consistency()`` gets both) the Odyssey
  model below runs in the same job and adds its metrics.
- ``odyssey-avd``: 3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes (MIT; transformers
  AutoModelForAudioClassification with trust_remote_code; input normalised with the config's
  mean/std; outputs arousal, dominance, valence in ~0..1). Metrics: absolute differences
  ``emo_arousal_d``, ``emo_valence_d``, ``emo_dominance_d`` and the Euclidean ``emo_avd_dist``.

- ``hf-classifier``: any transformers audio-classification SER head (``classifier``, default
  superb/hubert-large-superb-er, the model OpenDub's pipeline uses today in
  app/pipeline/emotion.py). Metrics ``emo_post_sim_hf`` (cosine of the class posteriors) and
  ``emo_label_match``.

A 2026 audit found emotion2vec cosine near chance across a language change; dimensional A/V/D
deltas are the more defensible signal. Both are kept so G4 can say which one tracks humans.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

from _sdk import serve
from judge_audio import RefCache, cosine, device_of, load_mono, write_wav

SR = 16000
E2V = "iic/emotion2vec_plus_large"
ODYSSEY = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"


def _emotion2vec(params: dict, device: str):
    import numpy as np
    from funasr import AutoModel

    kwargs = {"model": params.get("e2v_model", E2V), "device": device, "disable_update": True}
    if params.get("hub"):
        kwargs["hub"] = params["hub"]  # "hf" -> emotion2vec/emotion2vec_plus_large on HF
    model = AutoModel(**kwargs)
    tmp = Path(tempfile.mkdtemp(prefix="opendub-e2v-"))

    def analyse(path: str) -> dict:
        wav16 = write_wav(tmp / "in.wav", load_mono(path, SR), SR)
        res = model.generate(wav16, granularity="utterance", extract_embedding=True)[0]
        scores = np.asarray(res.get("scores") or [], dtype=np.float64)
        return {"feats": np.asarray(res["feats"], dtype=np.float64).ravel(), "scores": scores,
                "labels": [str(x).split("/")[-1] for x in res.get("labels") or []]}
    return analyse


def _odyssey(params: dict, device: str):
    import torch
    from transformers import AutoModelForAudioClassification

    repo = params.get("avd_model", ODYSSEY)
    model = AutoModelForAudioClassification.from_pretrained(
        repo, trust_remote_code=True, revision=params.get("avd_revision")).to(device).eval()
    mean, std, sr = model.config.mean, model.config.std, int(model.config.sampling_rate)

    def analyse(path: str) -> list[float]:
        wav = (load_mono(path, sr) - mean) / (std + 1e-6)
        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.ones_like(x)
        with torch.inference_mode():
            pred = model(x, mask)
        pred = getattr(pred, "logits", pred)
        a, d, v = (float(t) for t in pred.flatten()[:3])  # card: 0 arousal, 1 dominance, 2 valence
        return [a, v, d]
    return analyse


def _hf_classifier(params: dict, device: str):
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    repo = params.get("classifier", "superb/hubert-large-superb-er")
    fe = AutoFeatureExtractor.from_pretrained(repo, revision=params.get("classifier_revision"))
    model = AutoModelForAudioClassification.from_pretrained(
        repo, revision=params.get("classifier_revision")).to(device).eval()
    sr = int(getattr(fe, "sampling_rate", SR))
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    def analyse(path: str) -> dict:
        inputs = fe(load_mono(path, sr), sampling_rate=sr, return_tensors="pt").to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits.float(), dim=-1)[0].cpu().numpy()
        return {"scores": np.asarray(probs, dtype=np.float64), "labels": labels}
    return analyse


def load(params: dict, lang: str):
    model = params.get("model", "emotion2vec")
    if model not in ("emotion2vec", "odyssey-avd", "hf-classifier"):
        raise ValueError(f"unknown emotion model {model!r}")
    device = device_of(params)
    state = {"model": model, "device": device, "params": params, "e2v": None, "avd": None,
             "hf": None}
    if model == "emotion2vec":
        state["e2v"] = RefCache(_emotion2vec(params, device))
    if model == "hf-classifier":
        state["hf"] = RefCache(_hf_classifier(params, device))
    if model == "odyssey-avd" or (model == "emotion2vec" and params.get("avd", True)):
        state["avd"] = RefCache(_odyssey(params, device))
    default = {"emotion2vec": "emo_sim_emotion2vec", "hf-classifier": "emo_post_sim_hf"}.get(
        model, "neg_avd_dist")
    state["score_key"] = params.get("score", default)
    return state


def run(state: dict, item: dict, out: Path) -> dict:
    audio, ref = item["inputs"].get("audio"), item["inputs"].get("ref_audio")
    if not audio or not ref:
        raise ValueError("emotion judge needs inputs.audio and inputs.ref_audio")
    if isinstance(ref, list):
        ref = ref[0]
    metrics: dict[str, float] = {}
    payload: dict = {}
    if state["e2v"] is not None:
        a, r = state["e2v"](audio), state["e2v"](ref)
        metrics["emo_sim_emotion2vec"] = cosine(a["feats"], r["feats"])
        if len(a["scores"]) and len(a["scores"]) == len(r["scores"]):
            metrics["emo_post_sim_emotion2vec"] = cosine(a["scores"], r["scores"])
            la, lr = int(a["scores"].argmax()), int(r["scores"].argmax())
            metrics["emo_label_match"] = float(la == lr)
            payload["labels"] = {"out": a["labels"][la] if a["labels"] else la,
                                 "ref": r["labels"][lr] if r["labels"] else lr}
    if state["hf"] is not None:
        a, r = state["hf"](audio), state["hf"](ref)
        metrics["emo_post_sim_hf"] = cosine(a["scores"], r["scores"])
        la, lr = int(a["scores"].argmax()), int(r["scores"].argmax())
        metrics["emo_label_match"] = float(la == lr)
        payload["labels"] = {"out": a["labels"][la], "ref": r["labels"][lr]}
    if state["avd"] is not None:
        o, s = state["avd"](audio), state["avd"](ref)
        d = [abs(x - y) for x, y in zip(o, s)]
        metrics.update(emo_arousal_d=d[0], emo_valence_d=d[1], emo_dominance_d=d[2],
                       emo_avd_dist=math.sqrt(sum(x * x for x in d)))
        payload["avd"] = {"out": dict(zip(("arousal", "valence", "dominance"), o)),
                          "ref": dict(zip(("arousal", "valence", "dominance"), s))}
    key = state["score_key"]
    score = -metrics["emo_avd_dist"] if key == "neg_avd_dist" else metrics.get(key)
    return {"metrics": metrics, "score": score, **payload}


def describe(state: dict) -> dict:
    p = state["params"]
    return {"model": state["model"], "device": state["device"], "score": state["score_key"],
            "emotion2vec": p.get("e2v_model", E2V) if state["e2v"] else None,
            "avd_model": p.get("avd_model", ODYSSEY) if state["avd"] else None,
            "classifier": p.get("classifier") if state["hf"] else None}


if __name__ == "__main__":
    serve(load, run, describe)
