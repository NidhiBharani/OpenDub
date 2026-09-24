"""Mix stage levelling: the background keeps its source balance, E4 sets the delivery target."""
import re
import subprocess
from pathlib import Path

import pytest

from app.media import ffmpeg
from app.pipeline.audio import mix_tracks
from app.pipeline.stages import _delivery_lufs


def _tone(path: Path, freq: int, db: float, seconds: float = 6.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f={freq}:r=48000",
         "-t", str(seconds), "-af", f"volume={db}dB", "-ac", "2", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


def _band_rms_db(path: Path, band: str) -> float:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af",
         f"{band},astats=measure_perchannel=none", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr
    return float(re.findall(r"RMS level dB:\s*(-?[\d.]+)", out)[-1])


async def test_dub_matches_source_dialogue_and_background_is_not_ducked(tmp_path: Path):
    # Tones a decade apart from the band filters so neither leaks into the other's measurement.
    background = _tone(tmp_path / "bg.wav", 100, -30)
    dub = _tone(tmp_path / "dub.wav", 3000, -40)  # quiet TTS take
    source_dialogue = _tone(tmp_path / "src.wav", 3000, -20)
    out = tmp_path / "mix.wav"

    await mix_tracks(dub, background, out, reference_vocals=source_dialogue, target_lufs=-23.0)

    voice = _band_rms_db(out, "highpass=f=1000")
    bed = _band_rms_db(out, "lowpass=f=300")
    # Source balance: dialogue -20 over bed -30 → the dub sits 10 dB over the bed, as it did.
    assert voice - bed == pytest.approx(10.0, abs=1.0)
    assert await ffmpeg.measure_loudness(out) == pytest.approx(-23.0, abs=0.5)


async def test_without_reference_falls_back_to_fixed_vocal_level(tmp_path: Path):
    background = _tone(tmp_path / "bg.wav", 200, -30)
    dub = _tone(tmp_path / "dub.wav", 1000, -40)
    out = tmp_path / "mix.wav"

    await mix_tracks(dub, background, out, reference_vocals=tmp_path / "missing.wav")

    assert await ffmpeg.measure_loudness(out) == pytest.approx(-16.0, abs=0.5)


@pytest.mark.parametrize(
    ("delivery", "lufs"),
    [("web -16", -16.0), ("EBU R128 -23", -23.0), ("ATSC A/85 -24", -24.0), ("", -16.0)],
)
def test_delivery_preset_sets_the_target(delivery: str, lufs: float):
    assert _delivery_lufs({"delivery": delivery}) == lufs
