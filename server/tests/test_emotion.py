from __future__ import annotations

import asyncio
import math
import sys
import types
import wave
from pathlib import Path

import pytest

from app.models import TTSRequest
from app.pipeline import emotion
from app.providers.tts.elevenlabs import _v3_text
from app.providers.tts.f5_tts import F5TTSProvider


def _tone(path: Path, frequency: float = 180.0, seconds: float = 1.0) -> None:
    rate = 16000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(
            int(0.25 * math.sin(2 * math.pi * frequency * i / rate) * 32767).to_bytes(
                2, "little", signed=True
            ) for i in range(int(rate * seconds))
        ))


def test_acoustic_measurements_and_unsupported_language(tmp_path: Path) -> None:
    path = tmp_path / "speech.wav"
    _tone(path)
    analyzed = emotion.analyze_audio(path, language="ja")
    assert analyzed["model_status"] == "unsupported_language"
    assert analyzed["emotion"] == ""
    assert analyzed["prosody"]["duration_seconds"] == 1.0
    assert analyzed["prosody"]["rms_dbfs"] > -20
    assert 160 <= analyzed["prosody"]["pitch_hz"] <= 200
    assert analyzed["prosody"]["pitch_spread_semitones"] is not None
    assert emotion.delivery_hint(analyzed) == ""


def test_classifier_gates_uncertain_and_accepts_confident(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "speech.wav"
    _tone(path)
    monkeypatch.setattr(emotion.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(emotion, "_classify", lambda *_: [
        {"label": "angry", "score": 0.58}, {"label": "sad", "score": 0.42}
    ])
    low = emotion.analyze_audio(path)
    assert low["model_status"] == "uncertain" and not emotion.delivery_hint(low)
    monkeypatch.setattr(emotion, "_classify", lambda *_: [
        {"label": "angry", "score": 0.88}, {"label": "sad", "score": 0.12}
    ])
    high = emotion.analyze_audio(path)
    assert high["model_status"] == "ready"
    assert emotion.delivery_hint(high) == "angry"
    assert emotion.compare_delivery(high, high)["emotion_match"] is True


def test_custom_model_requires_explicit_language_scope(tmp_path: Path) -> None:
    path = tmp_path / "speech.wav"
    _tone(path)
    analyzed = emotion.analyze_audio(path, language="hi", model="local/custom-model")
    assert analyzed["model_status"] == "unsupported_language"


def test_configured_multilingual_model_uses_confidence_gate(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "speech.wav"
    model_dir = tmp_path / "emotion2vec_plus_base"
    model_dir.mkdir()
    _tone(path)
    monkeypatch.setattr(emotion.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(emotion, "_classify_funasr", lambda *_: [
        {"label": "生气/angry", "score": 0.91},
        {"label": "中立/neutral", "score": 0.09},
    ])
    result = emotion.analyze_audio(
        path, language="ja", model=f"funasr:{model_dir}", supported_languages=["ja", "hi"]
    )
    assert result["model_status"] == "ready"
    assert result["emotion"] == "angry"  # normalized below for multilingual labels


def test_funasr_receives_16khz_pcm(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    _tone(path, seconds=0.6)
    # Rewrite at 48 kHz, the project's standard rate.
    with wave.open(str(path), "rb") as wav:
        data = wav.readframes(wav.getnframes())
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(data * 3)

    class FakeAutoModel:
        def __init__(self, model):
            self.model_path = model

        def generate(self, *, input, granularity, extract_embedding):
            with wave.open(input, "rb") as wav:
                assert wav.getframerate() == 16000
                assert wav.getnchannels() == 1
            assert granularity == "utterance" and extract_embedding is False
            return [{"labels": ["生气/angry", "中立/neutral"], "scores": [0.9, 0.1]}]

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=FakeAutoModel))
    monkeypatch.setattr(emotion, "_FUNASR_CLASSIFIER", None)
    rows = emotion._classify_funasr(path, str(tmp_path))
    assert rows[0] == {"label": "angry", "score": 0.9}


def test_delivery_comparison_uses_pitch_movement_not_absolute_pitch(tmp_path: Path) -> None:
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    _tone(first, 180)
    _tone(second, 260)
    a = emotion.analyze_audio(first, language="ja")
    b = emotion.analyze_audio(second, language="hi")
    result = emotion.compare_delivery(a, b)
    assert result["score"] >= 0.9
    assert result["emotion_match"] is None
    assert result["pitch_spread_delta_semitones"] is not None
    assert emotion.confident_emotion_mismatch(a, b) is False


def test_eleven_v3_tag_mapping_is_bounded() -> None:
    assert _v3_text("Hello", "angry, shouting") == "[shouts] Hello"
    assert _v3_text("Hello", "surprised") == "Hello"


def test_f5_uses_speaker_reference_transcript(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    _tone(reference)
    reference.with_suffix(".txt").write_text("the original line")
    request = TTSRequest(text="translation", language="hi", speaker_reference=str(reference))
    picked = asyncio.run(F5TTSProvider({})._pick_reference(request))
    assert picked == (str(reference), "the original line")


def test_f5_rejects_low_score_and_rechecks_trim(monkeypatch, tmp_path: Path) -> None:
    from app.providers.tts import f5_tts

    async def run_thread(func, *args):
        if func is f5_tts._run_infer:
            args[4].write_bytes(b"candidate")
            return None
        return func(*args)

    monkeypatch.setattr(f5_tts.asyncio, "to_thread", run_thread)
    monkeypatch.setattr(f5_tts, "_verify_take", lambda *_: (0.1, 0.0))
    provider = F5TTSProvider({"verify": True})
    with pytest.raises(ValueError, match="failed transcript verification"):
        asyncio.run(provider._synthesize_verified(
            object(), "ref.wav", "", "Hello", tmp_path / "out.wav", 1.0, "en", lambda *_: None
        ))
    assert not list(tmp_path.glob("*.wav"))


def test_f5_rejects_take_after_failed_post_trim_check(monkeypatch, tmp_path: Path) -> None:
    from app.providers.tts import f5_tts

    async def run_thread(func, *args):
        if func is f5_tts._run_infer:
            args[4].write_bytes(b"candidate")
            return None
        return func(*args)

    async def trim_start(source, output, _seconds):
        output.write_bytes(source.read_bytes())

    checks = iter([(0.9, 0.2), (0.2, 0.0)])
    monkeypatch.setattr(f5_tts.asyncio, "to_thread", run_thread)
    monkeypatch.setattr(f5_tts, "_verify_take", lambda *_: next(checks))
    monkeypatch.setattr(f5_tts.ffmpeg, "trim_start", trim_start)
    provider = F5TTSProvider({"verify": True})
    with pytest.raises(ValueError, match="trimmed take failed"):
        asyncio.run(provider._synthesize_verified(
            object(), "ref.wav", "", "Hello", tmp_path / "out.wav", 1.0, "en", lambda *_: None
        ))
    assert not list(tmp_path.glob("*.wav"))


def test_superb_abbreviations_become_usable_emotion_hints(tmp_path, monkeypatch):
    from app.pipeline import emotion
    path = tmp_path / 'speech.wav'
    monkeypatch.setattr(emotion, 'acoustic_prosody', lambda _: {
        'duration_seconds': 2.0, 'pause_fraction': .1})
    monkeypatch.setattr(emotion.importlib.util, 'find_spec', lambda _: object())
    monkeypatch.setattr(emotion, '_classify', lambda *_: [
        {'label': 'ang', 'score': .9}, {'label': 'neu', 'score': .1}])
    monkeypatch.delenv('OPENDUB_EMOTION_MODEL', raising=False)
    assert emotion.delivery_hint(emotion.analyze_audio(path, language='en')) == 'angry'
