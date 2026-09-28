"""Reusable model judges. Capability specs compose these; judge workers live in ``workers/``.

Every factory returns :class:`~bench.arena.judges.ModelJudge` objects with fixed metric names, so
specs written in parallel agree on what they rank by. A judge whose worker or env is missing on
this machine is skipped with a note (``arena plan`` lists what each judge needs).

| factory              | worker              | worker item inputs                    | metrics                              |
|----------------------|---------------------|---------------------------------------|--------------------------------------|
| asr_roundtrip        | asr_* (A4 workers)  | {audio}                               | rt_cer_<asr>, rt_wer_<asr>           |
| speaker_similarity   | judge_speaker_sim   | {audio, ref_audio}                    | sim_<encoder>                        |
| naturalness          | judge_mos           | {audio}                               | mos_<model> (audiobox: aes_ce/cu/pc/pq; nisqa_*, dnsmos_* sub-scores) |
| emotion_consistency  | judge_emotion       | {audio, ref_audio}                    | emo_sim_emotion2vec, emo_post_sim_emotion2vec, emo_label_match, emo_arousal_d, emo_valence_d, emo_dominance_d, emo_avd_dist |
| lipsync_score        | judge_lipsync       | {video, audio?}                       | synchformer: sync_offset_ms_synchformer, sync_conf_synchformer; syncnet: lse_c, lse_d, sync_offset_ms_syncnet; peavs: peavs |
| mt_qe                | judge_mt_qe         | {source, hypothesis, reference?, src_lang, tgt_lang} | qe_<model>            |

Judge workers return ``{"metrics": {name: value}, "weights": {name: w}?, "score": x}`` except
the ASR judges, which are the A4 workers themselves and return a transcript. ``score`` is the
uniform per-item prediction the G-phase meta-evaluation ranks judges by (G2–G5 run these same
workers as candidates). Lip-sync offsets use one sign convention: positive = audio lags picture.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .hardware import Requires
from .judges import ModelJudge, Row, error_rows, output_file
from .packs import Item

TextOf = Callable[[Item, dict[str, Any]], str | None]


def spoken_text(item: Item, out: dict[str, Any]) -> str | None:
    """What a TTS/VC output was supposed to say: the input text, else the reference text."""
    return item.inputs.get("text") or item.refs.get("text")


def _metrics_rows(payload: dict[str, Any], rename: Callable[[str], str] = str) -> list[Row]:
    weights = payload.get("weights") or {}
    rows: list[Row] = []
    for name, value in (payload.get("metrics") or {}).items():
        if value is None:
            continue
        rows.append((rename(name), float(value), float(weights.get(name, 1.0))))
    return rows


# ---------------------------------------------------------------- ASR round-trip (G1)

# Two ASR families per language, so a candidate tuned against one ASR cannot game the score
# (docs/plans/model-ranking.md §4.7). Worker/env/params mirror the A4 candidates.
ASR_JUDGES: dict[str, dict[str, Any]] = {
    "whisper": {"worker": "asr_faster_whisper", "env": "server",
                "params": {"model": "large-v3", "compute_type": "float16", "beam_size": 5,
                           "word_timestamps": False, "condition_on_previous_text": False},
                "requires": {"gpu": True, "vram_gb": 4.5}, "languages": "*"},
    "qwen3": {"worker": "asr_qwen3", "env": "qwen3asr",
              "params": {"model": "Qwen/Qwen3-ASR-1.7B",
                         "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
                         "dtype": "bfloat16", "attn": "sdpa"},
              "requires": {"gpu": True, "vram_gb": 5}, "languages": ["en", "hi", "ja", "zh", "ko",
                                                                    "de", "fr", "es", "it", "pt",
                                                                    "ru"]},
    # The CTC family with no LM (plan §4.7): greedy decoding cannot "repair" a misread word.
    # MMS-1B-all is CC-BY-NC-4.0 (NC allowed as a judge, decision 2026-09-27). G1 ranks it.
    # https://huggingface.co/facebook/mms-1b-all
    "mms": {"worker": "g1_ctc", "env": "server",
            "params": {"model": "facebook/mms-1b-all", "mode": "greedy", "dtype": "float16"},
            "requires": {"gpu": True, "vram_gb": 5},
            "languages": ["en", "hi", "ja", "zh", "ko", "de", "fr", "es", "it", "pt", "ru", "ta",
                          "te", "bn", "ar"]},
}
ASR_PANEL: dict[str, list[str]] = {"*": ["whisper", "qwen3", "mms"]}


def asr_roundtrip(lang: str, *, text_of: TextOf = spoken_text, audio_key: str = "audio",
                  panel: list[str] | None = None, version: int = 1) -> list[ModelJudge]:
    """Re-transcribe the output audio and compare with what it should say (intelligibility)."""
    judges = []
    for name in panel or ASR_PANEL.get(lang, ASR_PANEL["*"]):
        cfg = ASR_JUDGES[name]

        def inputs(item: Item, out: dict[str, Any], prefix: Path, _k=audio_key):
            audio = output_file(out, _k)
            return {"audio": audio} if audio and text_of(item, out) else None

        def to_rows(item: Item, payload: dict[str, Any], out: dict[str, Any], _n=name):
            return error_rows(item.lang, text_of(item, out) or "", payload.get("text", ""),
                              prefix=f"rt_{_n}_")

        judges.append(ModelJudge(
            id=f"asr_roundtrip.{name}@{version}", worker=cfg["worker"], env=cfg["env"],
            params=cfg["params"], requires=Requires.parse(cfg["requires"]),
            languages=cfg["languages"], inputs=inputs, to_rows=to_rows))
    return [j for j in judges if j.supports(lang)]


# ---------------------------------------------------------------- speaker similarity (G2)

SIM_ENCODERS = {  # encoder -> (env, vram)
    # https://huggingface.co/microsoft/wavlm-base-plus-sv
    "wavlm-sv": ("judge_speaker", 1.5),
    # https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common (3D-Speaker)
    "eres2netv2": ("judge_speaker", 1.0),
    # https://github.com/PalabraAI/redimnet2 (B6, lm; torch.hub pinned in the worker)
    "redimnet2": ("judge_speaker", 1.0),
    # https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb (continuity with the pipeline)
    "ecapa": ("judge_speaker", 1.0),
    # https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM
    "wespeaker-resnet293": ("judge_speaker", 1.5),
    # https://github.com/resemble-ai/Resemblyzer (GE2E d-vector: the floor, not for ranking)
    "resemblyzer": ("judge_speaker", 0.5),
}


def speaker_similarity(encoder: str = "wavlm-sv", *, ref_key: str = "ref_audio",
                       audio_key: str = "audio", version: int = 1) -> ModelJudge:
    env, vram = SIM_ENCODERS[encoder]

    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        audio, ref = output_file(out, audio_key), item.inputs.get(ref_key) or item.refs.get(ref_key)
        return {"audio": audio, "ref_audio": ref} if audio and ref else None

    return ModelJudge(
        id=f"speaker_sim.{encoder}@{version}", worker="judge_speaker_sim", env=env,
        params={"encoder": encoder}, requires=Requires.parse({"gpu": True, "vram_gb": vram}),
        lang_param=False, inputs=inputs,
        to_rows=lambda item, payload, out: _metrics_rows(payload))


# ---------------------------------------------------------------- naturalness (G3)

# utmosv2 https://github.com/sarulab-speech/UTMOSv2 · distillmos https://github.com/microsoft/
# Distill-MOS · audiobox https://huggingface.co/facebook/audiobox-aesthetics · nisqa / nisqa-tts
# https://github.com/gabrielmittag/NISQA · dnsmos https://github.com/microsoft/DNS-Challenge ·
# xls-r-sqa https://github.com/lcn-kul/xls-r-analysis-sqa
MOS_MODELS = {"utmosv2": 2.5, "distillmos": 0.5, "audiobox": 1.5, "nisqa": 0.5,
              "nisqa-tts": 0.5, "dnsmos": 0.5, "xls-r-sqa": 5.0}


def naturalness(model: str = "utmosv2", *, audio_key: str = "audio",
                version: int = 1) -> ModelJudge:
    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        audio = output_file(out, audio_key)
        return {"audio": audio} if audio else None

    return ModelJudge(
        id=f"mos.{model}@{version}", worker="judge_mos", env="judge_mos",
        params={"model": model}, requires=Requires.parse({"gpu": True,
                                                          "vram_gb": MOS_MODELS[model]}),
        lang_param=False, inputs=inputs,
        to_rows=lambda item, payload, out: _metrics_rows(payload))


# ---------------------------------------------------------------- emotion (G4)

# emotion2vec https://huggingface.co/emotion2vec/emotion2vec_plus_large · odyssey-avd
# https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes
def emotion_consistency(model: str = "emotion2vec", *, ref_key: str = "ref_audio",
                        audio_key: str = "audio", with_avd: bool = True,
                        version: int = 1) -> ModelJudge:
    """Emotion similarity of the output to the source line (cross-lingual).

    ``model="emotion2vec"`` gives the emotion2vec+ large cosine and, with ``with_avd`` (default),
    the Odyssey-2024 WavLM arousal/valence/dominance deltas from the same job;
    ``model="odyssey-avd"`` gives only the A/V/D deltas."""
    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        audio, ref = output_file(out, audio_key), item.inputs.get(ref_key) or item.refs.get(ref_key)
        return {"audio": audio, "ref_audio": ref} if audio and ref else None

    return ModelJudge(
        id=f"emotion.{model}@{version}", worker="judge_emotion", env="judge_emotion",
        params={"model": model, "avd": bool(with_avd) or model == "odyssey-avd"},
        requires=Requires.parse({"gpu": True, "vram_gb": 3 if with_avd else 2}),
        lang_param=False, inputs=inputs,
        to_rows=lambda item, payload, out: _metrics_rows(payload))


# ---------------------------------------------------------------- lip sync (G5)

LIPSYNC_MODELS = {  # model -> requires (one env each: judge_lipsync_<model>)
    "synchformer": {"gpu": True, "vram_gb": 4},  # https://github.com/v-iashin/Synchformer
    "syncnet": {"gpu": True, "vram_gb": 3},      # https://github.com/joonson/syncnet_python
    "peavs": {"gpu": True, "vram_gb": 6},        # https://github.com/amazon-science/avgen-eval-toolkit
}


def lipsync_score(model: str = "synchformer", *, video_key: str = "video",
                  version: int = 1) -> ModelJudge:
    """Sync of a talking-face video. synchformer = signed offset (ms) + confidence; syncnet =
    LSE-C/LSE-D (comparability only; never rank SyncNet-supervised generators with it); peavs =
    human-calibrated 1..5 score."""
    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        video = output_file(out, video_key)
        return {"video": video, "audio": output_file(out, "audio")} if video else None

    return ModelJudge(
        id=f"lipsync.{model}@{version}", worker="judge_lipsync", env=f"judge_lipsync_{model}",
        params={"model": model}, requires=Requires.parse(LIPSYNC_MODELS[model]),
        lang_param=False, inputs=inputs,
        to_rows=lambda item, payload, out: _metrics_rows(payload))


# ---------------------------------------------------------------- MT quality estimation (C4)

QE_MODELS = {"metricx-24-hybrid-xl": 9.0, "metricx-24-hybrid-large": 4.0,
             "metricx-24-hybrid-xxl": 28.0, "cometkiwi-da-xl": 8.0, "xcomet-xl": 8.0,
             "xcomet-xxl": 24.0}


def mt_qe(model: str = "metricx-24-hybrid-xl", *, src_key: str = "text",
          use_reference: bool = False, version: int = 1) -> ModelJudge:
    """Reference-free (or hybrid, with ``use_reference``) translation quality of ``out["text"]``.
    MetricX is an error score (lower is better); COMET-family scores are higher-is-better."""
    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        src, hyp = item.inputs.get(src_key), out.get("text")
        if not src or not hyp:
            return None
        d = {"source": src, "hypothesis": hyp, "src_lang": item.meta.get("src_lang"),
             "tgt_lang": item.meta.get("tgt_lang", item.lang)}
        if use_reference and item.refs.get("text"):
            d["reference"] = item.refs["text"]
        return d

    return ModelJudge(
        id=f"mt_qe.{model}{'+ref' if use_reference else ''}@{version}", worker="judge_mt_qe",
        env="judge_mt_qe", params={"model": model},
        requires=Requires.parse({"gpu": True, "vram_gb": QE_MODELS[model]}),
        lang_param=False, inputs=inputs,
        to_rows=lambda item, payload, out: _metrics_rows(payload))
