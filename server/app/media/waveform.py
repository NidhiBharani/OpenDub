"""Waveform peak extraction for the editor's timeline lanes.

Streams ffmpeg's raw PCM output through a pipe in bounded chunks so multi-hour files never get
fully buffered in memory.
"""
from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

_SAMPLE_RATE = 48000  # fixed decode rate so bucket-size math is exact for common pairs_per_second values
_READ_SAMPLES = 65536  # samples per pipe read (128 KiB at s16)
_BYTES_PER_SAMPLE = 2


async def generate_peaks(audio: Path, out_json: Path, pairs_per_second: int = 50) -> None:
    """Decode `audio` to mono s16le PCM via an ffmpeg pipe and write min/max peak pairs as JSON.

    Output: {"version": 1, "sample_rate": pairs_per_second, "peaks": [min0, max0, min1, max1, ...]}
    with each value normalized to [-1, 1] and rounded to 3 decimals.
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    samples_per_bucket = max(1, _SAMPLE_RATE // max(1, pairs_per_second))

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(_SAMPLE_RATE),
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None

    # Drain stderr concurrently so a chatty ffmpeg process can't fill its pipe buffer and deadlock
    # while we're busy consuming stdout below.
    stderr_task = asyncio.create_task(proc.stderr.read())

    peaks: list[float] = []
    bucket_min = 32767
    bucket_max = -32768
    count_in_bucket = 0
    leftover = b""
    chunk_bytes = _READ_SAMPLES * _BYTES_PER_SAMPLE

    try:
        while True:
            chunk = await proc.stdout.read(chunk_bytes)
            if not chunk:
                break
            data = leftover + chunk
            usable_len = (len(data) // _BYTES_PER_SAMPLE) * _BYTES_PER_SAMPLE
            leftover = data[usable_len:]
            data = data[:usable_len]
            if not data:
                continue
            n = len(data) // _BYTES_PER_SAMPLE
            for s in struct.unpack(f"<{n}h", data):
                bucket_min = min(bucket_min, s)
                bucket_max = max(bucket_max, s)
                count_in_bucket += 1
                if count_in_bucket >= samples_per_bucket:
                    peaks.append(round(bucket_min / 32768.0, 3))
                    peaks.append(round(bucket_max / 32768.0, 3))
                    bucket_min = 32767
                    bucket_max = -32768
                    count_in_bucket = 0
    except asyncio.CancelledError:
        # Job cancellation: kill the decoder instead of leaving it blocked on a full pipe.
        # (The finally block then drains stderr and reaps the killed process quickly.)
        if proc.returncode is None:
            proc.kill()
        raise
    finally:
        stderr_data = await stderr_task
        returncode = await proc.wait()

    if returncode != 0:
        excerpt = stderr_data.decode("utf-8", "replace").strip()[-4000:]
        raise RuntimeError(f"ffmpeg failed (exit {returncode}): {excerpt}")

    if count_in_bucket > 0:
        peaks.append(round(bucket_min / 32768.0, 3))
        peaks.append(round(bucket_max / 32768.0, 3))

    if not peaks:
        peaks = [0.0, 0.0]

    out_json.write_text(
        json.dumps({"version": 1, "sample_rate": pairs_per_second, "peaks": peaks})
    )
