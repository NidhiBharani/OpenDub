"""Judge worker: reference-free naturalness / quality (MOS predictors).

item.inputs: {audio}. Payload: ``{"metrics": {"mos_<model>": x, ...}, "score": <primary>}``.
``score`` is the uniform prediction the G3 meta-evaluation ranks by (``params.score`` picks a
sub-score, e.g. ``aes_pq``).

params.model:

- ``utmosv2``: sarulab-speech/UTMOSv2 (``utmosv2.create_model(pretrained=True)``,
  ``model.predict(data=..., sr=16000)``). VMC2024 track 1 winner. Compare within a language only.
- ``distillmos``: microsoft/Distill-MOS (``distillmos.ConvTransformerSQAModel()``, 16 kHz
  waveform [batch, samples] -> MOS 1..5). 4.3M params, XLS-R-SQA student.
- ``audiobox``: facebookresearch/audiobox-aesthetics (``initialize_predictor().forward([{path}])``
  -> CE, CU, PC, PQ on a 1..10 scale); metrics ``aes_ce``, ``aes_cu``, ``aes_pc``, ``aes_pq``;
  default score = ``aes_pq`` (production quality, the artefact axis).
- ``nisqa``: NISQA v2.0 via torchmetrics (``non_intrusive_speech_quality_assessment(preds, fs)``
  -> [mos, noisiness, discontinuity, coloration, loudness]); metrics ``mos_nisqa`` + ``nisqa_noi``,
  ``nisqa_dis``, ``nisqa_col``, ``nisqa_loud``. Weights fetched by torchmetrics from the NISQA
  repo (CC BY-NC-SA 4.0 weights).
- ``nisqa-tts``: the NISQA-TTS naturalness head from the NISQA repo (``repo`` default
  ~/.opendub/src/NISQA, weights/nisqa_tts.tar) through its ``nisqaModel`` predict_file mode.
- ``dnsmos``: DNSMOS via torchmetrics (``deep_noise_suppression_mean_opinion_score``, ONNX,
  [p808_mos, mos_sig, mos_bak, mos_ovr]); metrics ``mos_dnsmos`` (= P.808) + ``dnsmos_sig``,
  ``dnsmos_bak``, ``dnsmos_ovr``. Built for noise suppression, not TTS: kept as a floor.
- ``xls-r-sqa``: lcn-kul XLS-R-SQA (the Distill-MOS teacher; ``size`` 300m|1b|2b, ``layers``,
  ``variant`` full|subset; ``E2EModel(config, xlsr_layers, dataset_variant, auto_download=True)``).
"""
from __future__ import annotations

import os
from pathlib import Path

from _sdk import serve
from judge_audio import device_of, load_mono

SR = 16000
DEFAULT_SCORE = {"utmosv2": "mos_utmosv2", "distillmos": "mos_distillmos", "audiobox": "aes_pq",
                 "nisqa": "mos_nisqa", "nisqa-tts": "mos_nisqa-tts", "dnsmos": "mos_dnsmos",
                 "xls-r-sqa": "mos_xls-r-sqa"}


def _utmosv2(params: dict, device: str):
    import torch
    import utmosv2

    try:  # newer releases take device=; the README only documents pretrained=True
        model = utmosv2.create_model(pretrained=True, device=device)
    except TypeError:
        model = utmosv2.create_model(pretrained=True)
        if hasattr(model, "to"):
            model = model.to(device)

    def score(path: str) -> dict:
        wav = torch.from_numpy(load_mono(path, SR))
        mos = model.predict(data=wav, sr=SR)
        return {"mos_utmosv2": float(mos)}
    return score


def _distillmos(params: dict, device: str):
    import distillmos
    import torch

    model = distillmos.ConvTransformerSQAModel().to(device).eval()

    def score(path: str) -> dict:
        wav = torch.from_numpy(load_mono(path, SR)).unsqueeze(0).to(device)
        with torch.inference_mode():
            return {"mos_distillmos": float(model(wav).flatten()[0])}
    return score


def _audiobox(params: dict, device: str):
    from audiobox_aesthetics.infer import initialize_predictor

    predictor = initialize_predictor()

    def score(path: str) -> dict:
        res = predictor.forward([{"path": path}])[0]
        return {f"aes_{k.lower()}": float(v) for k, v in res.items() if k in ("CE", "CU", "PC", "PQ")}
    return score


def _nisqa(params: dict, device: str):
    import torch
    from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment

    fs = int(params.get("fs", 48000))

    def score(path: str) -> dict:
        wav = torch.from_numpy(load_mono(path, fs))
        mos, noi, dis, col, loud = (float(x) for x in
                                    non_intrusive_speech_quality_assessment(wav, fs).flatten())
        return {"mos_nisqa": mos, "nisqa_noi": noi, "nisqa_dis": dis, "nisqa_col": col,
                "nisqa_loud": loud}
    return score


def _nisqa_tts(params: dict, device: str):
    import sys

    repo = Path(os.path.expanduser(params.get("repo", "~/.opendub/src/NISQA")))
    weights = repo / "weights" / params.get("weights", "nisqa_tts.tar")
    if not weights.exists():
        raise RuntimeError(f"NISQA-TTS weights not found at {weights}")
    sys.path.insert(0, str(repo))
    from nisqa.NISQA_model import nisqaModel

    def score(path: str) -> dict:
        args = {"mode": "predict_file", "pretrained_model": str(weights), "deg": path,
                "output_dir": None, "ms_channel": None, "tr_bs_val": 1, "tr_num_workers": 0,
                "tr_parallel": False}
        df = nisqaModel(args).predict()
        return {"mos_nisqa-tts": float(df["mos_pred"].iloc[0])}
    return score


def _dnsmos(params: dict, device: str):
    import torch
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

    personalized = bool(params.get("personalized", False))
    ort_device = params.get("ort_device", "cpu")  # ONNX on CPU: no CUDA-lib preload needed

    def score(path: str) -> dict:
        wav = torch.from_numpy(load_mono(path, SR))
        p808, sig, bak, ovr = (float(x) for x in deep_noise_suppression_mean_opinion_score(
            wav, SR, personalized, device=ort_device).flatten())
        return {"mos_dnsmos": p808, "dnsmos_sig": sig, "dnsmos_bak": bak, "dnsmos_ovr": ovr}
    return score


def _xlsr_sqa(params: dict, device: str):
    import torch
    from xls_r_sqa import config as cfgs
    from xls_r_sqa.e2e_model import E2EModel

    size = str(params.get("size", "2b")).upper()
    cfg = getattr(cfgs, f"XLSR_{size}_TRANSFORMER_32DEEP_CONFIG")
    layers = params.get("layers", 10 if size in ("1B", "2B") else 21)
    model = E2EModel(cfg, layers, dataset_variant=params.get("variant", "full"),
                     auto_download=True).to(device).eval()

    def score(path: str) -> dict:
        wav = torch.from_numpy(load_mono(path, SR)).to(device)
        with torch.inference_mode():
            return {"mos_xls-r-sqa": float(model.forward(wav).item())}
    return score


_LOADERS = {"utmosv2": _utmosv2, "distillmos": _distillmos, "audiobox": _audiobox,
            "nisqa": _nisqa, "nisqa-tts": _nisqa_tts, "dnsmos": _dnsmos, "xls-r-sqa": _xlsr_sqa}


def load(params: dict, lang: str):
    model = params.get("model", "utmosv2")
    if model not in _LOADERS:
        raise ValueError(f"unknown MOS model {model!r}; one of {', '.join(_LOADERS)}")
    device = device_of(params)
    return {"model": model, "fn": _LOADERS[model](params, device), "device": device,
            "score_key": params.get("score", DEFAULT_SCORE[model]), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    audio = item["inputs"].get("audio")
    if not audio:
        raise ValueError("MOS judge needs inputs.audio")
    metrics = state["fn"](audio)
    return {"metrics": metrics, "score": metrics.get(state["score_key"]),
            "model": state["model"]}


def describe(state: dict) -> dict:
    p = state["params"]
    return {"model": state["model"], "device": state["device"], "score": state["score_key"],
            **{k: p[k] for k in ("size", "layers", "variant") if k in p}}


if __name__ == "__main__":
    serve(load, run, describe)
