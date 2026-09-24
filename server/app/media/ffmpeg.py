"""ffmpeg/ffprobe subprocess helpers.

Per ARCHITECTURE.md, ffmpeg is invoked with ``-y -hide_banner -loglevel error`` and ONLY through the
helpers in this module (`app/media/ffmpeg.py`) — no other module should shell out to ffmpeg/ffprobe
directly. Every async helper here runs the subprocess via `asyncio.create_subprocess_exec` and raises
`RuntimeError` (with a trailing excerpt of stderr) on a nonzero exit code.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import uuid
from pathlib import Path

from ..models import MediaInfo

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

_STD_SAMPLE_RATE = 48000
_STD_CHANNELS = 2

# containers that ffmpeg happily faststart/remuxes as mp4 without re-muxing surprises
_MP4_FRIENDLY_FORMAT_TAGS = {"mp4", "mov", "m4a", "3gp", "3g2", "mj2"}


async def _run(binary: str, *args: str) -> bytes:
    """Run `binary -y -hide_banner -loglevel error <args>` and return stdout bytes."""
    proc = await asyncio.create_subprocess_exec(
        binary,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await proc.communicate()
    except asyncio.CancelledError:
        # Job cancellation: don't leave an orphaned ffmpeg/ffprobe writing artifacts.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise
    if proc.returncode != 0:
        excerpt = stderr.decode("utf-8", "replace").strip()[-4000:]
        raise RuntimeError(f"{binary} failed (exit {proc.returncode}): {excerpt}")
    return stdout


async def _ffmpeg(*args: str) -> None:
    await _run(FFMPEG_BIN, *args)


async def _ffprobe(*args: str) -> bytes:
    return await _run(FFPROBE_BIN, *args)


def _parse_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            num_f, den_f = float(num), float(den)
        except ValueError:
            return 0.0
        return num_f / den_f if den_f else 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


async def probe(path: Path) -> MediaInfo:
    """ffprobe -print_format json -show_format -show_streams -> MediaInfo."""
    out = await _ffprobe("-print_format", "json", "-show_format", "-show_streams", str(path))
    data = json.loads(out or b"{}")
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    if fmt.get("duration") is not None:
        try:
            duration = float(fmt["duration"])
        except (TypeError, ValueError):
            duration = 0.0
    if duration <= 0.0:
        for s in (video, audio):
            if s and s.get("duration"):
                try:
                    duration = max(duration, float(s["duration"]))
                except (TypeError, ValueError):
                    pass

    width = int((video or {}).get("width") or 0)
    height = int((video or {}).get("height") or 0)
    fps = _parse_fraction(video.get("r_frame_rate") or video.get("avg_frame_rate")) if video else 0.0

    return MediaInfo(
        duration=round(max(0.0, duration), 3),
        width=width,
        height=height,
        fps=round(fps, 3),
        has_audio=audio is not None,
    )


async def wav_duration(path: Path) -> float:
    out = await _ffprobe(
        "-print_format", "json", "-show_entries", "format=duration", str(path)
    )
    data = json.loads(out or b"{}")
    try:
        return round(float((data.get("format") or {}).get("duration", 0.0)), 6)
    except (TypeError, ValueError):
        return 0.0


async def extract_audio(
    video: Path, out_wav: Path, sample_rate: int = 48000, channels: int = 2
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    info = await probe(video)
    if not info.has_audio:
        await silent_wav(out_wav, info.duration or 0.0, sample_rate=sample_rate, channels=channels)
        return
    await _ffmpeg(
        "-i", str(video),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )


async def make_playback(video: Path, out_mp4: Path) -> None:
    """h264 + aac + faststart, scale to <=1080p; stream-copy video when already h264/mp4-friendly."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out = await _ffprobe("-print_format", "json", "-show_format", "-show_streams", str(video))
    data = json.loads(out or b"{}")
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    # Ignore attached_pic "video" streams (embedded cover art): they are not playable video and
    # copying their disposition into an mp4 makes ffmpeg fail outright.
    vstream = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")
        ),
        None,
    )
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if vstream is None:
        # audio-only source: nothing to make "playback video" from, but still produce a
        # faststart-able container so downstream code has something to serve.
        args = ["-i", str(video), "-vn"]
        args += ["-c:a", "aac", "-b:a", "192k"] if astream is not None else ["-an"]
        args += ["-movflags", "+faststart", str(out_mp4)]
        await _ffmpeg(*args)
        return

    format_tags = set((fmt.get("format_name") or "").split(","))
    is_h264 = vstream.get("codec_name") == "h264"
    mp4_friendly = bool(format_tags & _MP4_FRIENDLY_FORMAT_TAGS)
    # Browsers only decode 8-bit 4:2:0 H.264 — Hi10P/yuv444p (common anime encodes) must be
    # re-encoded or the playback.mp4 silently shows nothing in the editor's <video>.
    browser_safe = vstream.get("pix_fmt") == "yuv420p"
    height = int(vstream.get("height") or 0)

    args = ["-i", str(video)]
    if is_h264 and mp4_friendly and browser_safe:
        args += ["-c:v", "copy"]
    else:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        if height > 1080:
            args += ["-vf", "scale=-2:1080"]  # already mod-2 safe
        else:
            # libx264 + yuv420p rejects odd dimensions; force even width/height.
            args += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    args += ["-c:a", "aac", "-b:a", "192k"] if astream is not None else ["-an"]
    args += ["-movflags", "+faststart", str(out_mp4)]
    await _ffmpeg(*args)


async def slice_audio(
    wav: Path,
    out_wav: Path,
    start: float,
    end: float,
    pad: float = 0.0,
    sample_rate: int = _STD_SAMPLE_RATE,
    channels: int = _STD_CHANNELS,
) -> None:
    """Extract [start-pad, end+pad] (clamped to >=0) from `wav`, standardized to 48k s16 stereo
    by default (callers may override the rate/channels, e.g. for upload-size-limited APIs)."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    s = max(0.0, start - pad)
    dur = max(0.01, (end + pad) - s)
    await _ffmpeg(
        "-ss", f"{s:.3f}",
        "-i", str(wav),
        "-t", f"{dur:.3f}",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )


async def concat_audio(wavs: list[Path], out_wav: Path, gap: float = 0.15) -> None:
    """Re-encode inputs to 48k s16 stereo, insert `gap` seconds of silence between clips.

    A single filter_complex concat graph is used for up to ~40 inputs; larger lists are reduced in
    chunks first to keep the ffmpeg command line/graph a sane size.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if not wavs:
        await silent_wav(out_wav, 0.01)
        return
    if len(wavs) == 1:
        await to_std_wav(wavs[0], out_wav)
        return

    chunk_size = 40
    if len(wavs) > chunk_size:
        tmp_dir = out_wav.parent / f".concat_{uuid.uuid4().hex[:8]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            parts: list[Path] = []
            for i in range(0, len(wavs), chunk_size):
                part_out = tmp_dir / f"part_{i // chunk_size}.wav"
                await concat_audio(wavs[i : i + chunk_size], part_out, gap=gap)
                parts.append(part_out)
            await concat_audio(parts, out_wav, gap=gap)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    inputs: list[str] = []
    for w in wavs:
        inputs += ["-i", str(w)]

    filter_parts: list[str] = []
    labels: list[str] = []
    for idx in range(len(wavs)):
        filter_parts.append(
            f"[{idx}:a]aresample={_STD_SAMPLE_RATE},"
            f"aformat=sample_fmts=s16:channel_layouts=stereo[a{idx}]"
        )
        labels.append(f"[a{idx}]")
        if gap > 0 and idx != len(wavs) - 1:
            filter_parts.append(f"anullsrc=r={_STD_SAMPLE_RATE}:cl=stereo:d={gap:.3f}[g{idx}]")
            labels.append(f"[g{idx}]")

    filter_parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]")

    await _ffmpeg(
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[out]",
        "-ar", str(_STD_SAMPLE_RATE),
        "-ac", str(_STD_CHANNELS),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )


def _atempo_chain(factor: float) -> str:
    """Build a comma-chained `atempo=` filter string so each stage stays within ffmpeg's classic
    0.5-2.0 per-filter bounds, multiplying together to the requested overall `factor`."""
    if factor <= 0:
        raise ValueError("atempo factor must be positive")
    stages: list[float] = []
    f = factor
    if f > 2.0:
        while f > 2.0:
            stages.append(2.0)
            f /= 2.0
        stages.append(f)
    elif f < 0.5:
        while f < 0.5:
            stages.append(0.5)
            f /= 0.5
        stages.append(f)
    else:
        stages.append(f)
    return ",".join(f"atempo={x:.6f}" for x in stages)


async def atempo(wav: Path, out_wav: Path, factor: float) -> None:
    """factor >1 speeds up (shortens); chains multiple atempo filters when outside [0.5, 2.0]."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    await _ffmpeg(
        "-i", str(wav),
        "-filter:a", _atempo_chain(factor),
        "-ar", str(_STD_SAMPLE_RATE),
        "-ac", str(_STD_CHANNELS),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )


_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


async def detect_silences(
    audio: Path, noise_db: float = -35.0, min_dur: float = 0.45
) -> list[tuple[float, float | None]]:
    """Run silencedetect over `audio`, returning (start, end) silence intervals.

    `end` is `None` when the stream ends while still inside a silence run (ffmpeg never emits a
    matching `silence_end` then) — callers substitute the total duration. The filter reports at
    info level on stderr, so this bypasses `_run`'s `-loglevel error` like `measure_loudness`.
    """
    proc = await asyncio.create_subprocess_exec(
        FFMPEG_BIN,
        "-hide_banner", "-nostats", "-loglevel", "info",
        "-i", str(audio),
        "-af", f"silencedetect=noise={noise_db:g}dB:d={min_dur:g}",
        "-f", "null", "-",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await proc.communicate()
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise
    text = stderr.decode("utf-8", "replace")
    starts: list[float] = []
    ends: list[float] = []
    for line in text.splitlines():
        m = _SILENCE_START_RE.search(line)
        if m:
            starts.append(float(m.group(1)))
            continue
        m = _SILENCE_END_RE.search(line)
        if m:
            ends.append(float(m.group(1)))
    if proc.returncode not in (0, None) and not starts and not ends:
        raise RuntimeError(f"{FFMPEG_BIN} silencedetect failed (exit {proc.returncode}): {text[-500:]}")
    return [(start, ends[i] if i < len(ends) else None) for i, start in enumerate(starts)]


_PTS_TIME_RE = re.compile(r"pts_time:(\S+)")
_SCENE_SCORE_RE = re.compile(r"lavfi\.scene_score=(\S+)")


async def detect_scene_changes(video: Path, threshold: float = 0.3) -> list[tuple[float, float]]:
    """Shot boundaries as (time, score) pairs, score in 0..1, via the `scene` frame-difference
    metric. Frames are downscaled first (the metric is resolution-independent and ~10x faster)."""
    tmp = video.parent / f".scenes_{uuid.uuid4().hex[:8]}.txt"
    try:
        await _ffmpeg(
            "-i", str(video),
            "-an",
            "-vf", f"scale=320:-2,select='gt(scene,{threshold:g})',metadata=print:file={tmp}",
            "-f", "null", "-",
        )
        text = tmp.read_text() if tmp.exists() else ""
    finally:
        tmp.unlink(missing_ok=True)
    cuts: list[tuple[float, float]] = []
    t: float | None = None
    for line in text.splitlines():
        m = _PTS_TIME_RE.search(line)
        if m:
            try:
                t = float(m.group(1))
            except ValueError:
                t = None
            continue
        m = _SCENE_SCORE_RE.search(line)
        if m and t is not None:
            try:
                cuts.append((round(t, 3), min(1.0, max(0.0, float(m.group(1))))))
            except ValueError:
                pass
            t = None
    return cuts


async def overlay_ranges(
    base_video: Path, overlay_video: Path, ranges: list[tuple[float, float]], out_path: Path
) -> None:
    """Video-only output: `base_video` with the picture of `overlay_video` shown during `ranges`
    (used to put the original picture back over a lip-synced render inside skip ranges)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not ranges:
        await _ffmpeg("-i", str(base_video), "-an", "-c:v", "copy", str(out_path))
        return
    enable = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in ranges)
    graph = (
        "[1:v][0:v]scale2ref[ov][base];"
        f"[base][ov]overlay=eof_action=pass:enable='{enable}'[v]"
    )
    script = out_path.parent / f".overlay_{uuid.uuid4().hex[:8]}.txt"
    try:
        script.write_text(graph)
        await _ffmpeg(
            "-i", str(base_video),
            "-i", str(overlay_video),
            "-filter_complex_script", str(script),
            "-map", "[v]",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        )
    finally:
        script.unlink(missing_ok=True)


async def filmstrip(
    video: Path, out_jpg: Path, interval: float, cols: int, tile_w: int = 96, tile_h: int = 54
) -> int:
    """Contact sheet of one frame every `interval` seconds, `cols` tiles per row, each
    `tile_w`x`tile_h` (letterboxed on black). Returns the number of tiles written."""
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    info = await probe(video)
    count = max(1, int(info.duration // interval) + 1)
    rows = (count + cols - 1) // cols
    await _ffmpeg(
        "-i", str(video),
        "-an",
        "-vf",
        (
            f"fps=1/{interval:g},scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
            f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:black,tile={cols}x{rows}"
        ),
        "-frames:v", "1",
        "-q:v", "5",
        str(out_jpg),
    )
    return count


async def mux(video: Path, audio: Path, out_path: Path) -> None:
    """Replace the video's audio track with `audio`; -shortest.

    The video map is optional (``0:v:0?``) so audio-only sources — which make_playback
    deliberately supports — render to an audio-only mp4 instead of crashing the render stage."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    await _ffmpeg(
        "-i", str(video),
        "-i", str(audio),
        "-map", "0:v:0?",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    )


async def loudness_normalize(wav: Path, out_wav: Path, i: float = -16.0) -> None:
    """One-pass loudnorm to integrated loudness `i` LUFS.

    NOTE: one-pass loudnorm applies a *dynamic* gain ride, which pumps on sparse program
    material (long silences between dub lines) and lifts silent passages' noise floor. Prefer
    `measure_loudness` + `apply_gain` (static, linear) for mixing/mastering steps.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    await _ffmpeg(
        "-i", str(wav),
        "-af", f"loudnorm=I={i}:TP=-1.5:LRA=11",
        "-ar", str(_STD_SAMPLE_RATE),
        "-ac", str(_STD_CHANNELS),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )


_EBUR128_I_RE = re.compile(r"I:\s*(-?[0-9.]+)\s*LUFS")

# ebur128's gated integrated-loudness floor: a (near-)silent file measures at/below this.
SILENCE_LUFS = -70.0


async def measure_loudness(wav: Path) -> float:
    """Integrated loudness (LUFS, EBU R128 gated) of `wav`.

    Returns `SILENCE_LUFS` (-70.0) for silent/near-silent input. Runs ffmpeg's ebur128 filter,
    whose summary is printed to stderr at info level — so this bypasses `_run`'s
    `-loglevel error` and captures stderr itself.
    """
    proc = await asyncio.create_subprocess_exec(
        FFMPEG_BIN,
        "-hide_banner", "-nostats",
        "-i", str(wav),
        "-af", "ebur128",
        "-f", "null", "-",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await proc.communicate()
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise
    text = stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{FFMPEG_BIN} ebur128 failed (exit {proc.returncode}): {text[-4000:]}")
    matches = _EBUR128_I_RE.findall(text)
    if not matches:
        raise RuntimeError(f"could not parse ebur128 integrated loudness for {wav}")
    return max(float(matches[-1]), SILENCE_LUFS)  # summary block's I: is the last match


async def apply_gain(wav: Path, out_wav: Path, gain_db: float, true_peak_db: float | None = None) -> None:
    """Apply a static (linear, non-pumping) gain; optionally brick-limit true peaks.

    `true_peak_db` (e.g. -1.5) adds an alimiter after the gain so hot sections can't clip —
    unlike dynamic loudnorm this touches only peaks above the ceiling, not overall dynamics.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    filt = f"volume={gain_db:.2f}dB"
    if true_peak_db is not None:
        limit_linear = 10 ** (true_peak_db / 20)
        filt += f",alimiter=limit={limit_linear:.4f}:level=false"
    await _ffmpeg(
        "-i", str(wav),
        "-af", filt,
        "-ar", str(_STD_SAMPLE_RATE),
        "-ac", str(_STD_CHANNELS),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )


async def silent_wav(
    out_wav: Path, duration: float, sample_rate: int = 48000, channels: int = 2
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    layout = "stereo" if channels == 2 else "mono"
    await _ffmpeg(
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl={layout}",
        "-t", f"{max(0.01, duration):.3f}",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )


async def tone_wav(out_wav: Path, duration: float, freq: float = 340.0) -> None:
    """Speech-shaped placeholder: sine wave with light tremolo, attenuated to -18 dB, 48k stereo."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.05, duration)
    await _ffmpeg(
        "-f", "lavfi",
        "-i", f"sine=frequency={freq}:duration={dur:.3f}:sample_rate={_STD_SAMPLE_RATE}",
        "-af", "tremolo=f=4:d=0.5,volume=-18dB",
        "-ar", str(_STD_SAMPLE_RATE),
        "-ac", str(_STD_CHANNELS),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )


async def to_std_wav(src: Path, out_wav: Path) -> None:
    """Any audio -> 48k s16 stereo wav."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    await _ffmpeg(
        "-i", str(src),
        "-ar", str(_STD_SAMPLE_RATE),
        "-ac", str(_STD_CHANNELS),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )


# ---- internal helpers used only by app/pipeline/audio.py ---------------------------------------
# Kept here (rather than in pipeline/audio.py) because ARCHITECTURE.md restricts ffmpeg invocation
# to this module.


async def trim_start(wav: Path, out_wav: Path, start: float) -> None:
    """Drop the first `start` seconds of a wav."""
    await _ffmpeg("-ss", f"{start:.3f}", "-i", str(wav), str(out_wav))


async def pad_or_trim(wav: Path, out_wav: Path, target: float) -> None:
    """Pad the end with silence (apad) or hard-trim (-t) so output is exactly `target` seconds."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    await _ffmpeg(
        "-i", str(wav),
        "-af", "apad",
        "-t", f"{max(0.01, target):.3f}",
        "-ar", str(_STD_SAMPLE_RATE),
        "-ac", str(_STD_CHANNELS),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )


async def amix(vocals: Path, background: Path, out_wav: Path, background_gain_db: float = -2.0) -> None:
    """Mix two wavs (ffmpeg amix, normalize=0) with a volume filter applied to `background`."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    filt = (
        f"[1:a]volume={background_gain_db}dB[bg];"
        "[0:a][bg]amix=inputs=2:duration=first:normalize=0[out]"
    )
    await _ffmpeg(
        "-i", str(vocals),
        "-i", str(background),
        "-filter_complex", filt,
        "-map", "[out]",
        "-ar", str(_STD_SAMPLE_RATE),
        "-ac", str(_STD_CHANNELS),
        "-c:a", "pcm_s16le",
        str(out_wav),
    )
