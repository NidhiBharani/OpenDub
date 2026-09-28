"""Regression checks for timing and timeline mixing."""
from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

import pytest

from app.media import ffmpeg
from app.pipeline.audio import TimingOverflowError, assemble_track, fit_to_duration


def _tone(path: Path, seconds: float, amplitude: int = 6000) -> Path:
    frames = round(seconds * 48000)
    samples = array("h")
    for i in range(frames):
        sample = round(amplitude * math.sin(2 * math.pi * 440 * i / 48000))
        samples.extend((sample, sample))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(samples)
    return path


def _sample(path: Path, seconds: float) -> int:
    with wave.open(str(path), "rb") as wf:
        wf.setpos(round(seconds * 48000))
        values = array("h")
        values.frombytes(wf.readframes(1))
    return values[0]


async def test_overlapping_voices_sum_and_do_not_overwrite(tmp_path: Path):
    first = _tone(tmp_path / "first.wav", 0.5)
    second = _tone(tmp_path / "second.wav", 0.5)
    out = tmp_path / "track.wav"
    await assemble_track([(0.0, first), (0.25, second)], 0.75, out)
    # At the same phase in the overlap, the two voices add rather than replace.
    single = _sample(first, 0.301)
    overlap = _sample(out, 0.301)
    assert overlap == pytest.approx(2 * single, abs=2)
    assert _sample(out, 0.101) == pytest.approx(_sample(first, 0.101), abs=2)


async def test_short_take_is_padded_without_slowing(tmp_path: Path):
    take = _tone(tmp_path / "short.wav", 0.5)
    out = tmp_path / "fitted.wav"
    assert await fit_to_duration(take, out, 1.0) == 1.0
    assert await ffmpeg.wav_duration(out) == pytest.approx(1.0, abs=0.002)
    assert _sample(out, 0.75) == 0


async def test_long_take_requests_retry_instead_of_cutting_words(tmp_path: Path):
    take = _tone(tmp_path / "long.wav", 1.5)
    out = tmp_path / "fitted.wav"
    with pytest.raises(TimingOverflowError, match="requires"):
        await fit_to_duration(take, out, 1.0)
    assert not out.exists()


async def test_final_render_is_checked_after_aac_encoding(tmp_path: Path):
    take = _tone(tmp_path / "take.wav", 0.5)
    render = tmp_path / "dubbed.mp4"
    await ffmpeg.mux(take, take, render)
    result = await ffmpeg.validate_render(render, expected_duration=0.5)
    assert result["duration"] == pytest.approx(0.5, abs=0.05)
    assert await ffmpeg.measure_true_peak(render) < 0.0
    with pytest.raises(RuntimeError, match="duration"):
        await ffmpeg.validate_render(render, expected_duration=2.0)
