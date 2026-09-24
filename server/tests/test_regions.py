"""Skip ranges: normalization, the derived Segment.skipped flag, and the audio splice."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from app.models import Project, Segment, TimeRange
from app.pipeline.audio import mix_tracks, splice_ranges
from app.pipeline.regions import apply_skip_ranges, is_skipped, normalize_ranges


def _r(start: float, end: float, label: str = "") -> TimeRange:
    return TimeRange(start=start, end=end, label=label)


def test_normalize_clamps_sorts_and_merges():
    out = normalize_ranges([_r(9.0, 20.0, "b"), _r(1.0, 3.0, "a"), _r(2.5, 4.0), _r(5.0, 5.02)], 10.0)
    assert [(r.start, r.end) for r in out] == [(1.0, 4.0), (9.0, 10.0)]
    assert out[0].label == "a"  # merged: earlier range keeps its id/label
    assert out[1].label == "b"


def test_is_skipped_by_midpoint_or_half_overlap():
    ranges = [_r(10.0, 20.0)]
    inside = Segment(start=12.0, end=14.0)
    edge = Segment(start=19.0, end=23.0)  # midpoint 21 outside, overlap 1s of 4s
    half = Segment(start=18.0, end=22.0)  # overlap 2s of 4s
    assert is_skipped(inside, ranges)
    assert not is_skipped(edge, ranges)
    assert is_skipped(half, ranges)


def test_apply_reports_transitions():
    p = Project(name="x", segments=[Segment(id="a", start=1, end=2), Segment(id="b", start=5, end=6)])
    p.skip_ranges = [_r(0.0, 3.0)]
    assert apply_skip_ranges(p) == ({"a"}, set())
    p.skip_ranges = [_r(4.0, 7.0)]
    assert apply_skip_ranges(p) == ({"b"}, {"a"})
    assert [s.skipped for s in p.segments] == [False, True]


# ---- splice ---------------------------------------------------------------------------------


def _tone(path: Path, freq: int, db: float, seconds: float = 6.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f={freq}:r=48000",
         "-t", str(seconds), "-af", f"volume={db}dB", "-ac", "2", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


def _band_rms_db(path: Path, band: str, start: float, end: float) -> float:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-ss", str(start), "-t", str(end - start), "-i", str(path), "-af",
         f"{band},astats=measure_perchannel=none", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr
    return float(re.findall(r"RMS level dB:\s*(-?[\d.]+)", out)[-1])


async def test_splice_substitutes_overlay_inside_ranges(tmp_path: Path):
    base = _tone(tmp_path / "base.wav", 3000, -20)  # "dub"
    overlay = _tone(tmp_path / "orig.wav", 200, -20)  # "original"
    out = tmp_path / "out.wav"
    await splice_ranges(base, overlay, [(2.0, 4.0)], out)

    hi_in = _band_rms_db(out, "highpass=f=1000", 2.2, 3.8)
    lo_in = _band_rms_db(out, "lowpass=f=400", 2.2, 3.8)
    hi_out = _band_rms_db(out, "highpass=f=1000", 0.2, 1.8)
    lo_out = _band_rms_db(out, "lowpass=f=400", 0.2, 1.8)
    assert lo_in - hi_in > 20  # inside the range only the original's tone remains
    assert hi_out - lo_out > 20  # outside, only the dub


async def test_mix_keeps_original_in_skip_ranges(tmp_path: Path):
    background = _tone(tmp_path / "bg.wav", 100, -30)
    dub = _tone(tmp_path / "dub.wav", 3000, -20)
    original = _tone(tmp_path / "orig.wav", 500, -20)
    out = tmp_path / "mix.wav"
    await mix_tracks(dub, background, out, keep_original=[(1.0, 3.0)], original=original)

    mid_in = _band_rms_db(out, "bandpass=f=500:w=200", 1.2, 2.8)
    mid_out = _band_rms_db(out, "bandpass=f=500:w=200", 3.5, 5.5)
    assert mid_in - mid_out > 20


def test_zero_length_and_missing_original_are_noops(tmp_path: Path):
    p = Project(name="x")
    assert normalize_ranges([_r(1.0, 1.0)], None) == []
    assert apply_skip_ranges(p) == (set(), set())


@pytest.mark.parametrize("dur", [None, 0.0])
def test_normalize_without_duration_keeps_ranges(dur):
    assert len(normalize_ranges([_r(0.0, 100.0)], dur)) == 1
