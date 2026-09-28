"""Timeline assembly for the `mix` stage: fit a TTS take to its slot duration, place clips on a
full-length track, and mix vocals with background music/fx.

All ffmpeg work is delegated to `app/media/ffmpeg.py` (the only module allowed to invoke ffmpeg);
`assemble_track` is pure Python (the `wave` module) per spec, to avoid building giant ffmpeg filter
graphs for shows with hundreds of segments.
"""
from __future__ import annotations

import asyncio
import math
import mmap
import statistics
import tempfile
import wave
from array import array
from pathlib import Path

from ..media import ffmpeg

SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # bytes (s16)
_FRAME_SIZE = CHANNELS * SAMPLE_WIDTH


class TimingOverflowError(ValueError):
    """A spoken take cannot fit its slot without an excessive tempo change."""


async def fit_to_duration(
    take: Path, out_wav: Path, target: float, min_rate: float = 0.85, max_rate: float = 1.15
) -> float:
    """Fit a take while preserving natural pace and every spoken sample.

    Short takes retain their pace and are padded. Long takes speed up only as much as
    allowed; an unfit take raises ``TimingOverflowError`` for synthesis/rewrite retry.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    target = max(0.01, target)

    tmp_std = out_wav.parent / f".{out_wav.stem}.std.tmp.wav"
    tmp_stretched = out_wav.parent / f".{out_wav.stem}.rate.tmp.wav"
    tmp_trimmed = out_wav.parent / f".{out_wav.stem}.trim.tmp.wav"
    tmp_fitted = out_wav.parent / f".{out_wav.stem}.fitted.tmp.wav"
    try:
        await ffmpeg.to_std_wav(take, tmp_std)
        dur = await ffmpeg.wav_duration(tmp_std)
        if dur <= 0.0:
            raise ValueError(f"empty or unreadable TTS take: {take}")

        # Trim only substantial *outer silence*, leaving 50 ms on both edges.
        # This avoids stretching a long TTS tail or mistaking it for speech.
        if dur > target:
            silences = await ffmpeg.detect_silences(tmp_std, noise_db=-45, min_dur=0.12)
            lead = silences[0][1] if silences and silences[0][0] <= 0.02 else None
            tail = (
                silences[-1][0]
                if silences and (silences[-1][1] is None or silences[-1][1] >= dur - 0.02)
                else None
            )
            start = max(0.0, (lead or 0.0) - 0.05) if lead and lead > 0.15 else 0.0
            end = min(dur, tail + 0.05) if tail is not None and dur - tail > 0.15 else dur
            if end - start > 0.05 and (start > 0 or end < dur):
                await ffmpeg.slice_audio(tmp_std, tmp_trimmed, start, end)
                tmp_std.unlink(missing_ok=True)
                tmp_trimmed.replace(tmp_std)
                dur = await ffmpeg.wav_duration(tmp_std)

        rate = dur / target
        if rate > max_rate + 0.002:
            raise TimingOverflowError(
                f"{take} is {dur:.3f}s for a {target:.3f}s slot; "
                f"requires {rate:.3f}x tempo (limit {max_rate:.3f}x)"
            )
        # atempo may emit a few extra frames. A 1% margin avoids clipping or
        # false overflow for normally fitting lines; final silence is padded.
        applied = min(max_rate, max(1.0, rate * 1.01)) if rate > 1.0 else 1.0

        if abs(applied - 1.0) < 1e-6:
            stretched = tmp_std
        else:
            await ffmpeg.atempo(tmp_std, tmp_stretched, applied)
            stretched = tmp_stretched

        stretched_duration = await ffmpeg.wav_duration(stretched)
        if stretched_duration > target + 0.002:
            raise TimingOverflowError(
                f"{take} remains {stretched_duration:.3f}s after {applied:.3f}x tempo "
                f"for a {target:.3f}s slot"
            )
        if stretched_duration >= target:
            # Encoder rounding can leave at most two milliseconds extra. Keep the
            # whole take; never cut a final consonant to satisfy sample rounding.
            await ffmpeg.to_std_wav(stretched, tmp_fitted)
        else:
            await ffmpeg.pad_or_trim(stretched, tmp_fitted, target)
        tmp_fitted.replace(out_wav)
        return applied
    finally:
        tmp_std.unlink(missing_ok=True)
        tmp_stretched.unlink(missing_ok=True)
        tmp_trimmed.unlink(missing_ok=True)
        tmp_fitted.unlink(missing_ok=True)


async def assemble_track(
    placements: list[tuple[float, Path]], total_duration: float, out_wav: Path,
    level_outliers: bool = False,
) -> None:
    """Build the full-length dub vocal track in pure Python.

    Overlapping voices are summed in floating point before one final peak guard.
    A short fade removes splice clicks. Optional level correction only touches
    take-level outliers, based on speech frames rather than padded silence.
    Clips extending beyond the media duration are rejected.

    The assembly (large allocation, per-clip reads, full-track disk write) is blocking CPU/IO
    work, so it runs in a worker thread rather than on the event loop.
    """
    await asyncio.to_thread(_assemble_track_sync, placements, total_duration, out_wav, level_outliers)


def _assemble_track_sync(
    placements: list[tuple[float, Path]], total_duration: float, out_wav: Path,
    level_outliers: bool = False,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = out_wav.parent / f".{out_wav.stem}.assemble.tmp.wav"
    total_frames = max(1, round(max(0.0, total_duration) * SAMPLE_RATE))
    clips: list[tuple[float, Path, float]] = []
    levels: list[float] = []
    for start, clip in placements:
        with wave.open(str(clip), "rb") as wf:
            _check_std(wf, clip)
            frames = wf.getnframes()
        if not frames:
            continue
        offset_frames = round(start * SAMPLE_RATE)
        if offset_frames < 0 or offset_frames + frames > total_frames + 96:
            raise ValueError(f"{clip} extends outside the {total_duration:.3f}s media duration")
        rms_db = -100.0
        if level_outliers:
            energy = 0.0
            active = 0
            with wave.open(str(clip), "rb") as wf:
                while data := wf.readframes(SAMPLE_RATE):
                    samples = array("h")
                    samples.frombytes(data)
                    for value in samples:
                        if abs(value) > 104:  # -50 dBFS, ignoring padded silence
                            energy += float(value) * value
                            active += 1
            if active:
                rms_db = 20 * math.log10(math.sqrt(energy / active) / 32768)
                levels.append(rms_db)
        clips.append((start, clip, rms_db))

    median_db = statistics.median(levels) if levels else -30.0
    # Anonymous RAM would double the footprint of an already large feature-length
    # s16 track. Keep the float accumulation in a sparse disk-backed map instead.
    try:
        with tempfile.TemporaryFile(dir=out_wav.parent) as scratch:
            scratch.truncate(total_frames * CHANNELS * 4)
            with mmap.mmap(scratch.fileno(), 0) as mapped:
                buf = memoryview(mapped).cast("f")
                try:
                    _mix_placements(clips, buf, median_db, tmp_output)
                finally:
                    buf.release()
        tmp_output.replace(out_wav)
    finally:
        tmp_output.unlink(missing_ok=True)


def _mix_placements(
    clips: list[tuple[float, Path, float]], buf: memoryview, median_db: float, out_wav: Path
) -> None:
    fade_frames = round(0.008 * SAMPLE_RATE)
    for start, clip, rms_db in clips:
        offset = round(start * SAMPLE_RATE) * CHANNELS
        # Leave expressive dynamics intact. Correct only takes more than 6 dB
        # from the median, and never by more than 3 dB.
        correction_db = 0.0
        if rms_db > -60 and rms_db > median_db + 6:
            correction_db = max(-3.0, median_db + 6 - rms_db)
        elif rms_db > -60 and rms_db < median_db - 6:
            correction_db = min(3.0, median_db - 6 - rms_db)
        gain = 10 ** (correction_db / 20)
        with wave.open(str(clip), "rb") as wf:
            samples = array("h")
            samples.frombytes(wf.readframes(wf.getnframes()))
        clip_frames = len(samples) // CHANNELS
        for frame in range(clip_frames):
            fade_gain = min(1.0, (frame + 1) / max(1, fade_frames), (clip_frames - frame) / max(1, fade_frames))
            for ch in range(CHANNELS):
                k = frame * CHANNELS + ch
                pos = offset + k
                if pos < len(buf):
                    buf[pos] += samples[k] * gain * fade_gain

    peak = max((abs(v) for v in buf), default=0.0)
    peak_gain = min(1.0, 0.95 * 32767 / peak) if peak else 1.0
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        for start in range(0, len(buf), SAMPLE_RATE * CHANNELS):
            pcm = array("h", (round(v * peak_gain) for v in buf[start : start + SAMPLE_RATE * CHANNELS]))
            wf.writeframesraw(pcm)


async def splice_ranges(
    base: Path,
    overlay: Path,
    ranges: list[tuple[float, float]],
    out_wav: Path,
    fade: float = 0.05,
) -> None:
    """Write `base` with `overlay`'s audio substituted inside each (start, end) range, joined by
    linear crossfades of `fade` seconds on both edges. Both inputs must be 48k s16 stereo (the
    standard produced by `ffmpeg.to_std_wav` / `extract_audio`). Pure Python; runs in a thread."""
    await asyncio.to_thread(_splice_ranges_sync, base, overlay, ranges, out_wav, fade)


def _check_std(wf: wave.Wave_read, path: Path) -> None:
    if (
        wf.getframerate() != SAMPLE_RATE
        or wf.getnchannels() != CHANNELS
        or wf.getsampwidth() != SAMPLE_WIDTH
    ):
        raise ValueError(
            f"{path} is not 48k s16 stereo "
            f"(got {wf.getframerate()}Hz {wf.getnchannels()}ch {wf.getsampwidth() * 8}bit)"
        )


def _splice_ranges_sync(
    base: Path, overlay: Path, ranges: list[tuple[float, float]], out_wav: Path, fade: float
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(base), "rb") as wf:
        _check_std(wf, base)
        buf = bytearray(wf.readframes(wf.getnframes()))
    total_frames = len(buf) // _FRAME_SIZE
    fade_frames = max(0, round(fade * SAMPLE_RATE))

    with wave.open(str(overlay), "rb") as ov:
        _check_std(ov, overlay)
        ov_frames = ov.getnframes()
        limit = min(total_frames, ov_frames)

        def read(f0: int, f1: int) -> bytes:
            f0 = max(0, min(f0, limit))
            f1 = max(f0, min(f1, limit))
            if f1 <= f0:
                return b""
            ov.setpos(f0)
            return ov.readframes(f1 - f0)

        for start, end in sorted(ranges):
            s = max(0, round(start * SAMPLE_RATE))
            e = min(limit, round(end * SAMPLE_RATE))
            if e <= s:
                continue
            # inner: overlay replaces base outright
            data = read(s, e)
            buf[s * _FRAME_SIZE : s * _FRAME_SIZE + len(data)] = data
            # edges: linear crossfade base -> overlay (in) and overlay -> base (out)
            for f0, f1, rising in ((s - fade_frames, s, True), (e, e + fade_frames, False)):
                f0 = max(0, f0)
                f1 = min(limit, f1)
                n = f1 - f0
                if n <= 0:
                    continue
                a = array("h")
                a.frombytes(bytes(buf[f0 * _FRAME_SIZE : f1 * _FRAME_SIZE]))
                b = array("h")
                b.frombytes(read(f0, f1))
                if len(b) != len(a):
                    continue
                for i in range(n):
                    w = (i + 1) / (n + 1)
                    if not rising:
                        w = 1.0 - w
                    for ch in range(CHANNELS):
                        k = i * CHANNELS + ch
                        v = round(a[k] * (1.0 - w) + b[k] * w)
                        a[k] = max(-32768, min(32767, v))
                buf[f0 * _FRAME_SIZE : f1 * _FRAME_SIZE] = a.tobytes()

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(buf)


async def mix_tracks(
    vocals: Path,
    background: Path,
    out_wav: Path,
    reference_vocals: Path | None = None,
    target_lufs: float = -16.0,
    background_gain_db: float = 0.0,
    keep_original: list[tuple[float, float]] | None = None,
    original: Path | None = None,
    ducking_mix: float = 0.0,
) -> dict[str, float]:
    """Level the dub vocals, lay them over the background, and master — all with STATIC gains.

    The background is the source mix minus its dialogue, so it already carries the original
    music/effects balance (including any ducking the source editor did under speech). It goes in
    at unity by default; ducking it again would bury the music twice.

    TTS output level varies wildly between providers (F5-TTS peaks around -15..-20 dB), so the
    dub vocals get a measured, linear gain to match `reference_vocals` (the separated source
    dialogue), keeping the original voice-to-bed ratio. Without a usable reference they fall back
    to a speech-forward -18 LUFS. The premix is amixed (normalize=0) and moved as a whole to
    `target_lufs` with a -1.5 dBTP limiter, which preserves that balance.

    Static gains by default: one-pass loudnorm's dynamic ride pumps on sparse dialogue and lifts the
    noise floor of silent passages (audible hiss before the first line). Output is 48k s16
    stereo.

    `keep_original` ranges (with `original`, the full source mix at 48k s16 stereo) are spliced
    into the premix before mastering: the premix sits at source level (dub vocals matched to the
    original dialogue, bed at unity), so the original drops in at the same loudness and the one
    static master gain then moves the whole programme together. ``ducking_mix``
    enables restrained sidechain compression of the bed only when an un-ducked
    background masks the new speech; zero preserves the source bed balance."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_vocals = out_wav.parent / f".{out_wav.stem}.voclevel.tmp.wav"
    tmp_premix = out_wav.parent / f".{out_wav.stem}.premix.tmp.wav"
    tmp_spliced = out_wav.parent / f".{out_wav.stem}.splice.tmp.wav"
    tmp_headroom = out_wav.parent / f".{out_wav.stem}.headroom.tmp.wav"
    tmp_original = out_wav.parent / f".{out_wav.stem}.original.tmp.wav"
    tmp_master = out_wav.parent / f".{out_wav.stem}.master.tmp.wav"
    vocal_gain_db = 0.0
    splice_headroom_db = 0.0
    try:
        vocals_i = await ffmpeg.measure_loudness(vocals)
        if vocals_i > ffmpeg.SILENCE_LUFS + 1.0:
            vocal_target = -18.0
            if reference_vocals is not None and reference_vocals.exists():
                reference_i = await ffmpeg.measure_loudness(reference_vocals)
                if reference_i > ffmpeg.SILENCE_LUFS + 1.0:
                    vocal_target = reference_i
            # Capped at +30 dB so a barely-audible track can't be blasted into amplified noise.
            gain = min(vocal_target - vocals_i, 30.0)
            vocal_gain_db = gain
            await ffmpeg.apply_gain(vocals, tmp_vocals, gain, true_peak_db=-1.5)
            leveled = tmp_vocals
        else:
            leveled = vocals  # effectively silent: leave untouched (gain would only raise noise)

        await ffmpeg.amix(
            leveled, background, tmp_premix,
            background_gain_db=background_gain_db,
            float_output=True,
            ducking_mix=max(0.0, min(1.0, ducking_mix)),
        )

        premix = tmp_premix
        if keep_original and original is not None and original.exists():
            # The premix stays float until enough headroom is applied for the
            # s16 splice operation. Attenuate the original by the same amount
            # so skipped ranges retain their source balance.
            premix_peak_db = await ffmpeg.measure_true_peak(tmp_premix)
            splice_headroom_db = -max(9.0, premix_peak_db + 1.5)
            await ffmpeg.apply_gain(tmp_premix, tmp_headroom, splice_headroom_db)
            await ffmpeg.apply_gain(original, tmp_original, splice_headroom_db)
            await splice_ranges(tmp_headroom, tmp_original, keep_original, tmp_spliced)
            premix = tmp_spliced

        premix_i = await ffmpeg.measure_loudness(premix)
        master_gain = 0.0 if premix_i <= ffmpeg.SILENCE_LUFS + 1.0 else target_lufs - premix_i
        await ffmpeg.apply_gain(
            premix, tmp_master, max(-30.0, min(master_gain, 30.0)), true_peak_db=-1.5
        )
        tmp_master.replace(out_wav)
        return {
            "vocal_gain_db": vocal_gain_db,
            "background_gain_db": background_gain_db,
            "master_gain_db": max(-30.0, min(master_gain, 30.0)),
            "ducking_mix": max(0.0, min(1.0, ducking_mix)),
            "splice_headroom_db": splice_headroom_db,
        }
    finally:
        tmp_vocals.unlink(missing_ok=True)
        tmp_premix.unlink(missing_ok=True)
        tmp_spliced.unlink(missing_ok=True)
        tmp_headroom.unlink(missing_ok=True)
        tmp_original.unlink(missing_ok=True)
        tmp_master.unlink(missing_ok=True)
