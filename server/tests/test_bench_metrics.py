"""Unit tests for the benchmark metric modules — CPU-only, synthetic fixtures, no GPU/network.

Each test builds a minimal Project + on-disk wavs and asserts the metric module returns the
expected named metrics and that the reference-free ones produce numbers (not None).
"""
from __future__ import annotations

import math
import struct
import tempfile
import wave
from pathlib import Path

import pytest

from app.models import MediaInfo, Project, Segment, Take
from bench.cases import BenchCase
from bench.metrics import mix as mix_metric
from bench.metrics import transcribe as transcribe_metric
from bench.metrics import translate as translate_metric
from bench.metrics import tts as tts_metric
from bench.metrics.base import BenchCase as BC
from bench.metrics.base import MetricContext


def _write_wav(path: Path, seconds: float, freq: float, amp: float = 0.3, sr: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            v = int(amp * 32767 * math.sin(2 * math.pi * freq * i / sr)) if freq else 0
            frames += struct.pack("<hh", v, v)
        w.writeframes(bytes(frames))


def _names(metrics) -> set[str]:
    return {m.name for m in metrics}


def _by_name(metrics, name):
    return next(m for m in metrics if m.name == name)


def _ctx(tmp: Path, project: Project, case: BenchCase | None = None) -> MetricContext:
    case = case or BC(name="t", source=tmp / "source.mp4", source_lang="ja")
    return MetricContext(project=project, project_dir=tmp, case=case)


def _project(segments, media_dur=16.0, stages=None) -> Project:
    return Project(
        name="t", media=MediaInfo(duration=media_dur),
        stages=stages or {}, segments=segments,
    )


@pytest.mark.asyncio
async def test_mix_metrics_on_healthy_and_silent_lead_in():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # 6s tone file with a silent lead-in. First segment starts at 3.0s, so the metric's
        # lead-in window is [0, min(2.9, 2.4)] = [0, 2.4] — zero past that to keep it silent.
        _write_wav(tmp / "audio/dub_mix.wav", 6.0, 220.0, amp=0.5)
        _zero_head(tmp / "audio/dub_mix.wav", 2.9)

        seg = Segment(start=3.0, end=5.0, source_text="x", translated_text="hi")
        ctx = _ctx(tmp, _project([seg], media_dur=6.0))
        metrics = await mix_metric.score(ctx)
        names = _names(metrics)
        assert "mix.integrated_lufs" in names
        assert "mix.loudness_error" in names
        assert "mix.lead_in_noise_floor" in names
        # lead-in is truly silent → very low floor
        floor = _by_name(metrics, "mix.lead_in_noise_floor").value
        assert floor is not None and floor < -40


@pytest.mark.asyncio
async def test_mix_missing_file_reports_null():
    with tempfile.TemporaryDirectory() as d:
        ctx = _ctx(Path(d), _project([Segment(start=0, end=1)], media_dur=6.0))
        metrics = await mix_metric.score(ctx)
        assert metrics[0].value is None
        assert "not found" in metrics[0].note


@pytest.mark.asyncio
async def test_tts_flags_degenerate_take():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # good take (1s tone) and a degenerate take (0.04s silence)
        good = Take(path="audio/segments/s1/take_1.wav", rate_factor=1.0)
        bad = Take(path="audio/segments/s2/take_1.wav", rate_factor=0.6)
        _write_wav(tmp / good.path, 1.0, 300.0, amp=0.5)
        _write_wav(tmp / bad.path, 0.04, 0.0)
        s1 = Segment(start=0, end=2, translated_text="hello", takes=[good], active_take_id=good.id)
        s2 = Segment(start=3, end=8, translated_text="world", takes=[bad], active_take_id=bad.id)
        ctx = _ctx(tmp, _project([s1, s2]))
        metrics = await tts_metric.score(ctx)
        dr = _by_name(metrics, "tts.degenerate_take_rate")
        assert dr.value == 0.5  # one of two takes degenerate
        cr = _by_name(metrics, "tts.duration_clamp_rate")
        assert cr.value == 0.5  # bad take pinned to 0.6 clamp


@pytest.mark.asyncio
async def test_transcribe_free_metrics_and_wer_with_gt():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        segs = [Segment(start=0, end=2, source_text="hello world"),
                Segment(start=3, end=5, source_text="good morning")]
        # ground-truth transcript with one word wrong
        ref = tmp / "ref_transcript.jsonl"
        ref.write_text('{"start":0,"end":2,"text":"hello world"}\n'
                       '{"start":3,"end":5,"text":"good evening"}\n')
        case = BC(name="t", source=tmp / "s.mp4", source_lang="en", ref_transcript=ref)
        ctx = _ctx(tmp, _project(segs), case)
        metrics = await transcribe_metric.score(ctx)
        names = _names(metrics)
        assert "transcribe.speech_coverage" in names
        assert "transcribe.segments_per_min" in names
        wer = _by_name(metrics, "transcribe.wer")
        assert wer.value is not None and 0 < wer.value < 1  # 1 of 4 words wrong → 0.25


@pytest.mark.asyncio
async def test_translate_budget_compliance():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # 2s slot → budget max(12, 30)=30 chars. One within, one way over.
        segs = [Segment(start=0, end=2, source_text="x", translated_text="short line"),
                Segment(start=3, end=5, source_text="y", translated_text="z" * 90)]
        # no ollama configured in test env → judge reports null, chrf null; budget still computes
        ctx = _ctx(tmp, _project(segs))
        metrics = await translate_metric.score(ctx)
        bc = _by_name(metrics, "translate.budget_compliance")
        assert bc.value == 0.5  # one of two within budget


def _zero_head(path: Path, seconds: float) -> None:
    """Zero the first `seconds` of a wav in place (make a silent lead-in)."""
    with wave.open(str(path), "rb") as r:
        params = r.getparams()
        frames = bytearray(r.readframes(r.getnframes()))
    head = int(seconds * params.framerate) * params.sampwidth * params.nchannels
    for i in range(min(head, len(frames))):
        frames[i] = 0
    with wave.open(str(path), "wb") as w:
        w.setparams(params)
        w.writeframes(bytes(frames))
