"""Quality helper tests use tiny PCM fixtures and no optional model downloads."""
from __future__ import annotations

import math
import struct
import wave
from types import SimpleNamespace

import pytest

from app.pipeline.quality import (
    audio_stats,
    coverage_diagnostics,
    mix_statistics,
    speaker_similarity,
    text_error_rates,
    verify_speech,
)


def _tone(path, seconds=0.2):
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b''.join(struct.pack('<h', int(10000 * math.sin(i / 8)))
                                 for i in range(int(8000 * seconds))))


def test_audio_stats_explicit_missing_unreadable_and_signal(tmp_path):
    assert audio_stats(tmp_path / 'missing.wav')['status'] == 'missing'
    (tmp_path / 'bad.wav').write_text('not audio')
    assert audio_stats(tmp_path / 'bad.wav')['status'] == 'unreadable'
    path = tmp_path / 'tone.wav'
    _tone(path)
    stats = audio_stats(path)
    assert stats['status'] == 'ok'
    assert stats['duration_s'] == pytest.approx(0.2)
    assert -12 < stats['peak_dbfs'] < -9
    assert stats['clipping_fraction'] == 0
    assert stats['peak_kind'] == 'sample'


def test_unicode_text_rates_and_interval_union():
    assert text_error_rates('Hello, WORLD!', 'hello world') == {'wer': 0, 'cer': 0}
    assert text_error_rates('a b', 'a c')['wer'] == 0.5
    assert text_error_rates('', 'a') == {'wer': None, 'cer': None}
    result = coverage_diagnostics(iter([(0, 2), (1, 3), (4, 5)]), 10)
    assert result['coverage_ratio'] == pytest.approx(0.4)
    assert result['overlap_s'] == pytest.approx(1)
    assert result['overlap_pairs'] == 1


@pytest.mark.asyncio
async def test_optional_verifiers_do_not_download_and_accept_injected_models(tmp_path):
    path = tmp_path / 'tone.wav'
    _tone(path)
    assert (await verify_speech(path, 'hello'))['status'] == 'unavailable'
    assert (await speaker_similarity(path, path))['status'] == 'unavailable'

    class LocalModel:
        def transcribe(self, path, language=None):
            assert language == 'en'
            return [SimpleNamespace(text=' Hello, world!')], None

    checked = await verify_speech(path, 'hello world', LocalModel(), language='en')
    assert checked['status'] == 'ok'
    assert checked['wer'] == 0
    similarity = await speaker_similarity(path, path, embedder=lambda _: [1.0, 0.0])
    assert similarity['cosine'] == pytest.approx(1)


@pytest.mark.asyncio
async def test_mix_statistics_uses_same_window_and_reports_estimate(tmp_path, monkeypatch):
    from app.media import ffmpeg

    mix = tmp_path / 'mix.wav'
    voice = tmp_path / 'voice.wav'
    bed = tmp_path / 'bed.wav'
    for path in (mix, voice, bed):
        _tone(path)

    async def lufs(_):
        return -17.0

    async def true_peak(_):
        return -1.7

    monkeypatch.setattr(ffmpeg, 'measure_loudness', lufs)
    monkeypatch.setattr(ffmpeg, 'measure_true_peak', true_peak)
    result = await mix_statistics(mix, voice, bed, [(0, .2)], gains={
        'vocal_gain_db': 2.0, 'background_gain_db': 0.0}, target_lufs=-16)
    assert result['status'] == 'measured'
    assert result['loudness_error_lu'] == 1.0
    assert result['true_peak_dbtp'] == -1.7
    assert result['estimated_stem_margin_db'] == pytest.approx(2.0)
    assert result['stem_margin_status'] == 'estimated'


def test_truncated_wav_is_not_a_valid_take(tmp_path):
    path = tmp_path / 'truncated.wav'
    _tone(path)
    raw = path.read_bytes()
    path.write_bytes(raw[:len(raw) // 2])
    assert audio_stats(path)['status'] == 'unreadable'


def test_stereo_bed_energy_does_not_cancel_on_mono_downmix(tmp_path):
    from app.pipeline.quality import window_rms
    path = tmp_path / 'stereo.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(struct.pack('<hh', 10000, -10000) * 800)
    assert window_rms(path, 0, .1) == pytest.approx(10000 / 32768)
