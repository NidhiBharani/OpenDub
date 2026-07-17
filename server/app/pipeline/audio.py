"""Timeline assembly for the `mix` stage: fit a TTS take to its slot duration, place clips on a
full-length track, and mix vocals with background music/fx.

All ffmpeg work is delegated to `app/media/ffmpeg.py` (the only module allowed to invoke ffmpeg);
`assemble_track` is pure Python (the `wave` module) per spec, to avoid building giant ffmpeg filter
graphs for shows with hundreds of segments.
"""
from __future__ import annotations

import asyncio
import wave
from pathlib import Path

from ..media import ffmpeg

SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # bytes (s16)
_FRAME_SIZE = CHANNELS * SAMPLE_WIDTH


async def fit_to_duration(
    take: Path, out_wav: Path, target: float, min_rate: float = 0.6, max_rate: float = 1.6
) -> float:
    """Normalize `take` to 48k s16 stereo, atempo-stretch it to fill `target` seconds (rate clamped
    to [min_rate, max_rate]), then pad with silence or hard-trim to exactly `target`.

    Returns the applied (possibly clamped) rate factor, recorded on the Take for the editor UI.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    target = max(0.01, target)

    tmp_std = out_wav.parent / f".{out_wav.stem}.std.tmp.wav"
    tmp_stretched = out_wav.parent / f".{out_wav.stem}.rate.tmp.wav"
    try:
        await ffmpeg.to_std_wav(take, tmp_std)
        dur = await ffmpeg.wav_duration(tmp_std)
        if dur <= 0.0:
            dur = target

        rate = dur / target
        applied = min(max_rate, max(min_rate, rate))

        if abs(applied - 1.0) < 1e-6:
            stretched = tmp_std
        else:
            await ffmpeg.atempo(tmp_std, tmp_stretched, applied)
            stretched = tmp_stretched

        await ffmpeg.pad_or_trim(stretched, out_wav, target)
        return applied
    finally:
        tmp_std.unlink(missing_ok=True)
        tmp_stretched.unlink(missing_ok=True)


async def assemble_track(
    placements: list[tuple[float, Path]], total_duration: float, out_wav: Path
) -> None:
    """Build the full-length dub vocal track in pure Python.

    All `placements` clips are guaranteed 48k s16 stereo (i.e. `fit_to_duration` output). A silence
    buffer of `total_duration` seconds is preallocated and each clip's frames are copied in at its
    start offset, overwriting whatever was there — later entries in `placements` win on overlap.
    Clips extending past `total_duration` are clamped (truncated).

    The assembly (large allocation, per-clip reads, full-track disk write) is blocking CPU/IO
    work, so it runs in a worker thread rather than on the event loop.
    """
    await asyncio.to_thread(_assemble_track_sync, placements, total_duration, out_wav)


def _assemble_track_sync(
    placements: list[tuple[float, Path]], total_duration: float, out_wav: Path
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, round(max(0.0, total_duration) * SAMPLE_RATE))
    buf = bytearray(total_frames * _FRAME_SIZE)  # all-zero == silence for signed PCM

    for start, clip in placements:
        with wave.open(str(clip), "rb") as wf:
            if (
                wf.getframerate() != SAMPLE_RATE
                or wf.getnchannels() != CHANNELS
                or wf.getsampwidth() != SAMPLE_WIDTH
            ):
                raise ValueError(
                    f"{clip} is not 48k s16 stereo "
                    f"(got {wf.getframerate()}Hz {wf.getnchannels()}ch {wf.getsampwidth() * 8}bit)"
                )
            data = wf.readframes(wf.getnframes())

        offset_frames = max(0, round(start * SAMPLE_RATE))
        if offset_frames >= total_frames:
            continue  # placement starts at/after the end of the track: nothing to copy

        clip_frames = len(data) // _FRAME_SIZE
        avail_frames = total_frames - offset_frames
        copy_frames = min(clip_frames, avail_frames)  # clamp clips extending past the end
        if copy_frames <= 0:
            continue

        copy_bytes = copy_frames * _FRAME_SIZE
        byte_offset = offset_frames * _FRAME_SIZE
        buf[byte_offset : byte_offset + copy_bytes] = data[:copy_bytes]  # overwrite: later wins

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        # Write the bytearray directly — bytes(buf) would materialize a second full-track copy
        # and double peak memory (~GBs on feature-length sources).
        wf.writeframes(buf)


async def mix_tracks(
    vocals: Path, background: Path, out_wav: Path, background_gain_db: float = -8.0
) -> None:
    """Level the dub vocals, duck the background, mix, and master — all with STATIC gains.

    TTS output level varies wildly between providers (F5-TTS peaks around -15..-20 dB, far below
    a typical music/effects bed), so the vocal track is leveled to a speech-forward -18 LUFS via
    a measured, linear gain (R128 gating measures the spoken parts, ignoring the gaps). The
    background is ducked (default -8 dB, a normal dialogue-over-music margin), amixed
    (normalize=0), and the result mastered to -16 LUFS with a -1.5 dBTP limiter.

    Static gains only: one-pass loudnorm's dynamic ride pumps on sparse dialogue and lifts the
    noise floor of silent passages (audible hiss before the first line). Output is 48k s16
    stereo."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_vocals = out_wav.parent / f".{out_wav.stem}.voclevel.tmp.wav"
    tmp_premix = out_wav.parent / f".{out_wav.stem}.premix.tmp.wav"
    try:
        vocals_i = await ffmpeg.measure_loudness(vocals)
        if vocals_i > ffmpeg.SILENCE_LUFS + 1.0:
            # Static gain to -18 LUFS, capped at +30 dB so a barely-audible track can't be
            # blasted into pure amplified noise.
            gain = min(-18.0 - vocals_i, 30.0)
            await ffmpeg.apply_gain(vocals, tmp_vocals, gain, true_peak_db=-1.5)
            leveled = tmp_vocals
        else:
            leveled = vocals  # effectively silent: leave untouched (gain would only raise noise)

        await ffmpeg.amix(leveled, background, tmp_premix, background_gain_db=background_gain_db)

        premix_i = await ffmpeg.measure_loudness(tmp_premix)
        master_gain = 0.0 if premix_i <= ffmpeg.SILENCE_LUFS + 1.0 else -16.0 - premix_i
        await ffmpeg.apply_gain(
            tmp_premix, out_wav, max(-30.0, min(master_gain, 30.0)), true_peak_db=-1.5
        )
    finally:
        tmp_vocals.unlink(missing_ok=True)
        tmp_premix.unlink(missing_ok=True)
