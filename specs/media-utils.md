# Media utilities contract — implemented by B2, consumed by B1 (stages) and providers

All ffmpeg invocation goes through these modules. Async functions run ffmpeg/ffprobe subprocesses via
`asyncio.create_subprocess_exec` with `-y -hide_banner -loglevel error`; raise `RuntimeError` with
stderr excerpt on nonzero exit.

## `app/media/ffmpeg.py`

```python
async def probe(path: Path) -> MediaInfo                     # ffprobe json: duration, w, h, fps, has_audio
async def wav_duration(path: Path) -> float
async def extract_audio(video: Path, out_wav: Path, sample_rate: int = 48000, channels: int = 2) -> None
async def make_playback(video: Path, out_mp4: Path) -> None
    # h264 + aac + faststart, scale to ≤1080p; stream-copy video when already h264
async def slice_audio(wav: Path, out_wav: Path, start: float, end: float, pad: float = 0.0) -> None
async def concat_audio(wavs: list[Path], out_wav: Path, gap: float = 0.15) -> None
async def atempo(wav: Path, out_wav: Path, factor: float) -> None
    # factor >1 speeds up; chain atempo filters to stay within ffmpeg's per-filter 0.5–2.0 bounds
async def mux(video: Path, audio: Path, out_path: Path) -> None      # replace audio track, -shortest
async def loudness_normalize(wav: Path, out_wav: Path, i: float = -16.0) -> None    # loudnorm one-pass
async def silent_wav(out_wav: Path, duration: float, sample_rate: int = 48000, channels: int = 2) -> None
async def tone_wav(out_wav: Path, duration: float, freq: float = 340.0) -> None
    # speech-shaped placeholder: sine + light tremolo, -18 dB
async def to_std_wav(src: Path, out_wav: Path) -> None       # any audio → 48k s16 stereo wav
def ffmpeg_available() -> bool
```

## `app/media/waveform.py`

```python
async def generate_peaks(audio: Path, out_json: Path, pairs_per_second: int = 50) -> None
```
Decode to mono s16le PCM via ffmpeg pipe, compute per-bucket (min, max) normalized to -1..1, write
`{"version": 1, "sample_rate": <pairs_per_second>, "peaks": [min0, max0, min1, max1, ...]}` (floats,
3 decimals). Must handle multi-hour files without loading everything at once (stream in chunks).

## `app/pipeline/audio.py`

```python
async def fit_to_duration(take: Path, out_wav: Path, target: float,
                          min_rate: float = 0.6, max_rate: float = 1.6) -> float
    # Normalize to 48k s16 stereo. rate = take_duration / target, clamped; atempo; then pad with
    # silence or hard-trim to exactly `target`. Returns the applied (possibly clamped) rate factor.
async def assemble_track(placements: list[tuple[float, Path]], total_duration: float, out_wav: Path) -> None
    # All inputs are 48k s16 stereo (fit_to_duration output). Build the full-length track in pure
    # Python with the `wave` module: preallocated silence buffer, copy each clip's frames at its
    # start offset (clip overlaps: later placement wins / simple overwrite). No giant ffmpeg graphs.
async def mix_tracks(vocals: Path, background: Path, out_wav: Path, background_gain_db: float = -2.0) -> None
    # ffmpeg amix (normalize=0) with a volume filter on background; output 48k s16 stereo;
    # then loudness_normalize to -16 LUFS.
```
