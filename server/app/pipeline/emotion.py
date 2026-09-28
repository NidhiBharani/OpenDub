"""Conservative speech emotion and delivery measurements.

The optional SUPERB HuBERT classifier was trained for English IEMOCAP speech. Its labels
must not be projected onto arbitrary dubbed languages. Audio measurements are available
without ML packages and describe sound, not psychological emotion.
"""
from __future__ import annotations

import importlib.util
import math
import os
import statistics
import tempfile
import wave
from array import array
from pathlib import Path
from typing import Any

from ..providers._runtime import manager

MODEL_ID = "superb/hubert-large-superb-er"
MIN_EMOTION_CONFIDENCE = 0.65
MIN_EMOTION_MARGIN = 0.15
# Classifiers live in the process-wide model manager (app.providers._runtime), keyed by model.
_FUNASR_CLASSIFIER: Any = None  # unused; kept so older tests that reset it still import cleanly


def _pcm(path: Path) -> tuple[list[float], int]:
    """Read PCM WAV with the standard library. Pipeline inputs are standardized WAVs."""
    with wave.open(str(path), "rb") as wav:
        channels, width, rate, count = (
            wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        )
        raw = wav.readframes(count)
    if rate <= 0 or channels < 1 or width not in (1, 2, 4):
        raise ValueError("expected 8-, 16-, or 32-bit PCM WAV")
    if width == 1:
        values = [(v - 128) / 128.0 for v in raw]
    else:
        fmt = "h" if width == 2 else "i"
        values_array = array(fmt)
        values_array.frombytes(raw)
        if values_array.itemsize != width:
            raise ValueError("unsupported PCM sample width")
        if __import__("sys").byteorder != "little":
            values_array.byteswap()
        scale = float(1 << (width * 8 - 1))
        values = [v / scale for v in values_array]
    if channels > 1:
        values = [sum(values[i:i + channels]) / channels for i in range(0, len(values), channels)]
    return values, rate


def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-8))


def _pitch(samples: list[float], rate: int) -> float | None:
    """Coarse autocorrelation pitch on a short center excerpt, adequate for relative comparison."""
    if rate < 8000 or len(samples) < int(rate * 0.12):
        return None
    step = max(1, rate // 4000)
    excerpt = samples[max(0, len(samples) // 2 - rate // 8):len(samples) // 2 + rate // 8:step]
    if not excerpt:
        return None
    effective_rate = rate / step
    mean = statistics.fmean(excerpt)
    centered = [v - mean for v in excerpt]
    power = sum(v * v for v in centered)
    if power < 1e-4:
        return None
    lo, hi = max(1, int(effective_rate / 450)), min(len(centered) // 2, int(effective_rate / 75))
    best_lag, best_corr = 0, 0.0
    for lag in range(lo, hi + 1):
        corr = sum(centered[i] * centered[i + lag] for i in range(len(centered) - lag))
        if corr > best_corr:
            best_lag, best_corr = lag, corr
    if not best_lag or best_corr / power < 0.30:
        return None
    return round(effective_rate / best_lag, 1)


def _pitch_contour(samples: list[float], rate: int) -> tuple[float | None, float | None]:
    """Median and within-line semitone spread from several voiced windows."""
    window = max(1, int(rate * 0.30))
    if len(samples) < window:
        pitch = _pitch(samples, rate)
        return pitch, None
    count = min(16, max(1, len(samples) // max(1, int(rate * 0.2))))
    starts = [round(i * (len(samples) - window) / max(1, count - 1)) for i in range(count)]
    pitches = [p for start in starts if (p := _pitch(samples[start:start + window], rate))]
    if not pitches:
        return None, None
    median = statistics.median(pitches)
    semitones = sorted(12 * math.log2(p / median) for p in pitches)
    lo = semitones[round((len(semitones) - 1) * 0.1)]
    hi = semitones[round((len(semitones) - 1) * 0.9)]
    return round(median, 1), round(hi - lo, 2)


def acoustic_prosody(path: str | Path) -> dict[str, float | None]:
    samples, rate = _pcm(Path(path))
    if not samples:
        raise ValueError("empty audio")
    window = max(1, int(rate * 0.025))
    rms = [math.sqrt(sum(v * v for v in samples[i:i + window]) / len(samples[i:i + window]))
           for i in range(0, len(samples), window)]
    peak_db = _db(max(abs(v) for v in samples))
    active = [v for v in rms if _db(v) > max(-42.0, peak_db - 30.0)]
    pause_fraction = 1.0 - len(active) / len(rms)
    active_db = sorted(_db(v) for v in active)
    energy_variation = statistics.pstdev(active_db) if len(active_db) > 1 else 0.0
    energy_range = (active_db[round((len(active_db) - 1) * 0.9)]
                    - active_db[round((len(active_db) - 1) * 0.1)]) if len(active_db) > 1 else 0.0
    threshold = max(-42.0, peak_db - 30.0)
    active_flags = [_db(v) > threshold for v in rms]
    pause_runs: list[int] = []
    run = 0
    bursts = 0
    prev_active = False
    for flag in active_flags + [True]:
        if not flag:
            run += 1
        elif run:
            if run * 0.025 >= 0.1:
                pause_runs.append(run)
            run = 0
        if flag and not prev_active:
            bursts += 1
        prev_active = flag
    pitch, pitch_spread = (
        _pitch_contour(samples, rate) if active and pause_fraction < 0.95 else (None, None)
    )
    return {
        "duration_seconds": round(len(samples) / rate, 3),
        "rms_dbfs": round(_db(math.sqrt(sum(v * v for v in samples) / len(samples))), 2),
        "peak_dbfs": round(peak_db, 2),
        "pause_fraction": round(pause_fraction, 3),
        "pause_count": len(pause_runs),
        "mean_pause_seconds": round(statistics.fmean(pause_runs) * 0.025, 3)
        if pause_runs else 0.0,
        "activity_bursts_per_second": round(
            max(0, bursts - (not active_flags[-1])) / (len(samples) / rate), 3
        ),
        "energy_variation_db": round(energy_variation, 2),
        "energy_range_db": round(energy_range, 2),
        "pitch_hz": pitch,
        "pitch_spread_semitones": pitch_spread,
    }


def _classify(path: Path, model: str) -> list[dict[str, Any]]:
    def load() -> Any:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline

        extractor = AutoFeatureExtractor.from_pretrained(model, local_files_only=True)
        classifier = AutoModelForAudioClassification.from_pretrained(model, local_files_only=True)
        # No device argument: the Transformers pipeline runs on CPU.
        return pipeline("audio-classification", model=classifier, feature_extractor=extractor)

    with manager.acquire(("quality.emotion", "transformers", model, "cpu"), load,
                         est_vram_gb=0.0) as classifier:
        return classifier(str(path), top_k=None)


def _classify_funasr(path: Path, model_dir: str) -> list[dict[str, Any]]:
    """Use a preinstalled emotion2vec+ directory through its native FunASR interface.

    Requiring a local directory prevents FunASR's implicit network download during a job.
    """
    def load() -> Any:
        from funasr import AutoModel

        return AutoModel(model=model_dir)

    samples, rate = _pcm(path)
    # emotion2vec+ documents 16 kHz input; project WAVs are normally 48 kHz.
    if rate != 16000:
        length = max(1, round(len(samples) * 16000 / rate))
        samples = [samples[min(len(samples) - 1, round(i * rate / 16000))]
                   for i in range(length)]
    with tempfile.TemporaryDirectory(prefix="opendub-emotion-") as temp_dir:
        input_path = Path(temp_dir) / "input.wav"
        with wave.open(str(input_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            pcm = array("h", [max(-32768, min(32767, round(v * 32767))) for v in samples])
            if __import__("sys").byteorder != "little":
                pcm.byteswap()
            wav.writeframes(pcm.tobytes())
        # FunASR picks CUDA when available; emotion2vec+ base is small (~0.4 GB).
        with manager.acquire(("quality.emotion", "funasr", model_dir), load,
                             est_vram_gb=0.4) as classifier:
            rows = classifier.generate(
                input=str(input_path), granularity="utterance", extract_embedding=False
            )
    if not rows or not isinstance(rows[0], dict):
        return []
    row = rows[0]
    labels, scores = row.get("labels"), row.get("scores")
    if not isinstance(labels, list) or not isinstance(scores, list) or len(labels) != len(scores):
        return []
    return [{"label": str(label).split("/")[-1], "score": float(score)}
            for label, score in zip(labels, scores)]


def analyze_audio(
    path: str | Path, *, language: str = "en", model: str = "auto",
    supported_languages: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Return measured prosody and a confidence-gated model label, if supported.

    `auto` uses locally cached English SUPERB. For other languages, supply a local Transformers
    audio-classification model and its documented supported_languages, or set
    OPENDUB_EMOTION_MODEL=funasr:/local/path/to/emotion2vec_plus_base and
    OPENDUB_EMOTION_LANGUAGES=ja,hi (only after validating those languages on your material).
    There is no implicit model download. Model scores are left unknown without language evidence.
    """
    path = Path(path)
    prosody = acoustic_prosody(path)
    model = os.environ.get("OPENDUB_EMOTION_MODEL", model) if model == "auto" else model
    if supported_languages is None:
        configured = os.environ.get("OPENDUB_EMOTION_LANGUAGES", "")
        supported_languages = tuple(v.strip().lower() for v in configured.split(",") if v.strip())
    default_model = model == "auto"
    model_id = MODEL_ID if default_model else model
    result: dict[str, Any] = {
        "emotion": "", "confidence": 0.0, "model_status": "unavailable",
        "model_reason": "", "model_id": model_id,
        "prosody": prosody,
    }
    if model in ("", "off", "none"):
        result["model_reason"] = "emotion classifier disabled"
        return result
    lang = language.split("-")[0].split("_")[0].lower()
    allowed = {"en"} if default_model else {str(v).lower() for v in supported_languages}
    if lang not in allowed:
        result.update(model_status="unsupported_language", model_reason=(
            f"no documented or configured emotion model support for language '{lang}'"
        ))
        return result
    if prosody["duration_seconds"] < 0.5 or prosody["pause_fraction"] > 0.8:
        result.update(model_status="insufficient_audio", model_reason="clip too short or mostly silent")
        return result
    funasr_model = model.startswith("funasr:")
    dependency = "funasr" if funasr_model else "transformers"
    if importlib.util.find_spec(dependency) is None:
        result["model_reason"] = f"{dependency} is not installed"
        return result
    try:
        if funasr_model:
            local_dir = model.partition(":")[2]
            if not Path(local_dir).is_dir():
                raise FileNotFoundError(f"local FunASR model directory missing: {local_dir}")
            predictions = _classify_funasr(path, local_dir)
        else:
            predictions = _classify(path, result["model_id"])
    except Exception as exc:  # noqa: BLE001 - optional third-party model failure is nonfatal
        result["model_reason"] = f"emotion model unavailable: {type(exc).__name__}: {exc}"
        return result
    ranked = sorted(predictions, key=lambda item: float(item.get("score", 0)), reverse=True)
    if not ranked:
        result.update(model_status="uncertain", model_reason="classifier returned no labels")
        return result
    confidence = float(ranked[0]["score"])
    margin = confidence - (float(ranked[1]["score"]) if len(ranked) > 1 else 0.0)
    result["confidence"] = round(confidence, 3)
    if (str(ranked[0]["label"]).lower() in {"other", "unknown", "<unk>"}
            or confidence < MIN_EMOTION_CONFIDENCE or margin < MIN_EMOTION_MARGIN):
        result.update(model_status="uncertain", model_reason="emotion confidence below threshold")
        return result
    label = str(ranked[0]["label"]).split("/")[-1].lower()
    label = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad",
             "happiness": "happy", "anger": "angry", "sadness": "sad"}.get(label, label)
    if label.startswith("label_"):
        result.update(model_status="uncertain", model_reason="classifier has no semantic label mapping")
        return result
    result.update(emotion=label, model_status="ready", model_reason="")
    return result


def delivery_hint(analysis: dict[str, Any]) -> str:
    """Only pass a categorical hint when the model has made a reliable prediction."""
    return str(analysis.get("emotion") or "") if analysis.get("model_status") == "ready" else ""


def compare_delivery(source: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    """Compare audible dynamics and pitch; semantic emotion requires two reliable model labels."""
    a, b = source.get("prosody") or {}, output.get("prosody") or {}
    if not a or not b:
        return {"status": "unavailable", "score": None, "reason": "missing audio measurements"}
    pause_delta = abs(float(a.get("pause_fraction") or 0) - float(b.get("pause_fraction") or 0))
    variation_delta = abs(float(a.get("energy_variation_db") or 0) - float(b.get("energy_variation_db") or 0))
    range_delta = abs(float(a.get("energy_range_db") or 0) - float(b.get("energy_range_db") or 0))
    burst_delta = abs(float(a.get("activity_bursts_per_second") or 0)
                      - float(b.get("activity_bursts_per_second") or 0))
    # Absolute loudness is normally changed during mastering and is deliberately excluded.
    components = [max(0.0, 1.0 - pause_delta / 0.5),
                  max(0.0, 1.0 - variation_delta / 12.0),
                  max(0.0, 1.0 - range_delta / 18.0),
                  max(0.0, 1.0 - burst_delta / 3.0)]
    pitch_spread_delta: float | None = None
    if a.get("pitch_spread_semitones") is not None and b.get("pitch_spread_semitones") is not None:
        pitch_spread_delta = abs(float(b["pitch_spread_semitones"])
                                 - float(a["pitch_spread_semitones"]))
        components.append(max(0.0, 1.0 - pitch_spread_delta / 10.0))
    emotion_match: bool | None = None
    if (source.get("model_status") == output.get("model_status") == "ready"
            and source.get("model_id") == output.get("model_id")):
        emotion_match = source.get("emotion") == output.get("emotion")
    return {
        "status": "measured", "score": round(statistics.fmean(components), 3),
        "pause_delta": round(pause_delta, 3),
        "activity_burst_delta_per_second": round(burst_delta, 3),
        "energy_variation_delta_db": round(variation_delta, 2),
        "energy_range_delta_db": round(range_delta, 2),
        "pitch_spread_delta_semitones": round(pitch_spread_delta, 2)
        if pitch_spread_delta is not None else None,
        "emotion_match": emotion_match,
        "reason": "acoustic similarity only; different languages limit emotion comparison",
    }


def confident_emotion_mismatch(source: dict[str, Any], output: dict[str, Any]) -> bool:
    """Only gate a take when both clips have comparable, confidence-gated model labels."""
    return compare_delivery(source, output).get("emotion_match") is False
