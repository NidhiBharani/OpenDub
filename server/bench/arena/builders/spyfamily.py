"""SPY×FAMILY episode 1 (official Muse Asia / Muse India uploads) → the SCENE/ja headline pack.

Three official YouTube uploads of the same episode share one music & effects bed:

* Japanese original with burned-in English subtitles (Muse Asia) — the dubbing *input*;
* the official English dub (Muse Asia) and the official Hindi dub (Muse India) — human
  *reference performances* (and, once transcribed, human reference translations).

The builder downloads the three videos with ``yt-dlp`` (a user tool, never a server
dependency) into ``data/eval/_sources/anime/spyfamily/``, measures the time offset of each dub
against the Japanese timeline by cross-correlating 100 Hz onset envelopes at checkpoints across
the whole episode (so an edit that shifts one version shows up as drift), picks dialogue-dense
1–3 minute stretches away from the songs, cuts aligned clips and writes
``data/eval/SCENE/ja/manifest.jsonl``::

    {"id": "spyfamily-ep1-s1", "group": "spyfamily-ep1",
     "inputs": {"video": "clips/….mp4", "audio": "clips/….ja.wav"},
     "refs": {"dub_en_audio": "clips/….en.wav", "dub_hi_audio": "clips/….hi.wav"},
     "meta": {"start_s", "end_s", "duration_s", "offsets": {"en", "hi"}, "sources": {...}}}

Speech density is judged without any model: where the Japanese, English and Hindi mixes all
agree spectrally, nobody (or only a song) is audible over the shared bed; where both dubs
diverge from the original in the speech bands, someone is talking. Clip boundaries snap to
moments where no version diverges, preferring ``silencedetect`` gaps in the Japanese track.
The analysis is cached in ``_sources/anime/spyfamily/analysis.json`` (``force=True`` redoes it).

Build (downloads if missing; no models run)::

    uv run --frozen python -m bench arena pack spyfamily --lang ja --n 4
    uv run --frozen python -m bench.arena.builders.spyfamily [--segments 4] [--force]

Opt-in reference transcripts (runs the A4 worker; GPU; the user runs this, builders never do)::

    uv run --frozen python -m bench.arena.builders.spyfamily --transcribe-refs \
        [--candidate qwen3-asr-1.7b]

which calls :func:`transcribe_refs` to add ``refs.dub_en_text`` / ``refs.dub_hi_text`` (the
human reference translations for C2, and the reference performances' words for D-phase
comparisons). Private local evaluation only: YouTube ToS apply, never commit or redistribute.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from ..packs import pack_dir, write_pack
from . import builder

CAPABILITY = "SCENE"
GROUP = "spyfamily-ep1"
RATE = 100            # envelope frames per second
SR = 16000            # analysis sample rate
ANALYSIS_VERSION = 2


@dataclass(frozen=True)
class Source:
    lang: str
    video_id: str
    channel: str
    title: str

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def filename(self) -> str:
        return f"ep1-{self.lang}.mp4"


SOURCES: dict[str, Source] = {
    "ja": Source("ja", "h_tL16PZ0IE", "Muse Asia", "SPY×FAMILY - Episode 01 [English Sub]"),
    "en": Source("en", "ZgyothCzhLA", "Muse Asia", "SPY×FAMILY - Episode 01 [English Dub]"),
    "hi": Source("hi", "dXgq4u3ViFs", "Muse Hindi Dub",
                 "[Hindi Dub] SPY×FAMILY - Episode 01 | Muse IN"),
}
DUBS = ("en", "hi")

# Best mp4-compatible stream ≤1080p (H.264 + AAC, no DRC-normalised audio), merged into mp4.
YTDLP_FORMAT = ("bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a][format_id!*=drc]"
                "/b[height<=1080][ext=mp4]")

LICENSE_MD = """# SCENE / ja — SPY×FAMILY episode 1 (official uploads)

Sources (official uploads by the licensed distributor, downloaded with yt-dlp for private,
local evaluation only):

- Japanese original, English subtitles burned in — Muse Asia: {ja}
- Official English dub — Muse Asia: {en}
- Official Hindi dub — Muse India (Muse Hindi Dub): {hi}

© Tatsuya Endo/Shueisha, SPY×FAMILY Project. All rights reserved by the rights holders.
Use is subject to the YouTube Terms of Service. **Private evaluation only: do not commit,
publish, upload or otherwise redistribute these clips, the source videos or anything derived
from them (outputs of models run on them included).** `data/` is gitignored for this reason.

Clips: dialogue-dense stretches cut on the Japanese timeline; the dub audio in `refs` is cut at
the measured offset so all three play in sync. Group: the whole episode is one group
(`{group}`), so the bootstrap treats it as a single title.
Builder: `server/bench/arena/builders/spyfamily.py` (offsets and segment choice in
`data/eval/_sources/anime/spyfamily/analysis.json`).
"""


# ---------------------------------------------------------------------------------------------
# Side effects: network and ffmpeg. Small functions so tests can monkeypatch them.
# ---------------------------------------------------------------------------------------------

def src_dir() -> Path:
    return paths.eval_dir() / "_sources" / "anime" / "spyfamily"


def ytdlp_command() -> list[str]:
    exe = shutil.which("yt-dlp")
    if exe is None:
        raise RuntimeError("yt-dlp not found; install it as a user tool: "
                           "`uv tool install 'yt-dlp[default]'` (never into server/.venv)")
    cmd = [exe]
    if shutil.which("deno") is None and shutil.which("node") is not None:
        cmd += ["--js-runtimes", "node"]  # YouTube's JS challenge needs a runtime
    return cmd


def download(url: str, dest: Path) -> Path:
    """Fetch one video with yt-dlp into ``dest`` (plus ``.info.json``). Network."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ytdlp_command() + ["--no-playlist", "-f", YTDLP_FORMAT, "--merge-output-format", "mp4",
                             "--write-info-json", "--no-progress",
                             "-o", str(dest.with_suffix("")) + ".%(ext)s", url]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not dest.exists():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"yt-dlp failed for {url}: {' | '.join(tail)}")
    return dest


def decode_mono(path: Path, sr: int = SR) -> np.ndarray:
    """Decode the first audio stream to mono float32 at ``sr`` with ffmpeg."""
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
                           "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(proc.stdout, np.float32)


def detect_silences(path: Path, noise_db: float = -35.0,
                    min_s: float = 0.35) -> list[tuple[float, float]]:
    """ffmpeg ``silencedetect`` on the audio → [(start, end)] seconds."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af",
                           f"silencedetect=noise={noise_db}dB:d={min_s}", "-f", "null", "-"],
                          capture_output=True, text=True, check=True)
    return parse_silences(proc.stderr)


def parse_silences(log: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    start: float | None = None
    for m in re.finditer(r"silence_(start|end): (-?[0-9.]+)", log):
        t = float(m.group(2))
        if m.group(1) == "start":
            start = max(0.0, t)
        elif start is not None:
            out.append((start, t))
            start = None
    return out


def cut_video(src: Path, dest: Path, start: float, duration: float) -> Path:
    """Frame-accurate re-encoded cut (H.264 CRF 18 + AAC 192k) of the original video."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src),
                    "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "0:a:0",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dest)],
                   check=True)
    return dest


def cut_audio(src: Path, dest: Path, start: float, duration: float) -> Path:
    """Sample-accurate PCM cut of the audio at its native rate and channel count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:a:0",
                    "-af", (f"atrim=start={max(start, 0.0):.4f}:duration={duration:.4f},"
                            "asetpts=PTS-STARTPTS"),
                    "-c:a", "pcm_s16le", str(dest)], check=True)
    return dest


# ---------------------------------------------------------------------------------------------
# Pure numpy analysis
# ---------------------------------------------------------------------------------------------

def band_energies(x: np.ndarray, sr: int = SR, rate: int = RATE, n_fft: int = 512,
                  n_bands: int = 24, fmin: float = 180.0) -> np.ndarray:
    """Log-spaced band energies in dB, one row per 1/``rate`` s frame → (frames, bands)."""
    hop = sr // rate
    x = np.asarray(x, np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    lo = max(1, round(fmin * n_fft / sr))
    edges = np.unique(np.geomspace(lo, n_fft // 2, n_bands + 1).astype(int))
    win = np.hanning(n_fft).astype(np.float32)
    rows = []
    for i in range(0, n_frames, 16384):
        idx = np.arange(n_fft)[None, :] + hop * np.arange(i, min(n_frames, i + 16384))[:, None]
        spec = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
        rows.append(np.stack([spec[:, a:b].sum(1) for a, b in pairwise(edges)], 1))
    return 10 * np.log10(np.concatenate(rows) + 1e-10)


def onset_envelope(bands: np.ndarray) -> np.ndarray:
    """Spectral flux (half-wave rectified band-energy rise) from :func:`band_energies`."""
    d = np.diff(bands, axis=0, prepend=bands[:1])
    return np.maximum(d, 0).sum(1)


def estimate_offset(a: np.ndarray, b: np.ndarray, *, start: int = 0, win: int | None = None,
                    search: int, rate: int = RATE) -> tuple[float, float]:
    """Lag (s) such that ``b[t + lag] ≈ a[t]`` for the window ``a[start:start+win]``.

    Normalised cross-correlation of envelope frames, searching ±``search`` frames. Returns
    ``(lag_s, confidence)`` where confidence is the peak over the median |correlation| (≳10 is
    a clean match; dialogue-only windows score low because the dubs replace the dialogue).
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    win = min(win or len(a) - start, len(a) - start)
    seg = a[start:start + win]
    seg = (seg - seg.mean()) / (seg.std() + 1e-12)
    lo, hi = max(0, start - search), min(len(b), start + win + search)
    ref = b[lo:hi]
    if len(ref) < win:
        return 0.0, 0.0
    n = 1 << int(np.ceil(np.log2(len(ref) + win)))
    corr = np.fft.irfft(np.fft.rfft(ref, n) * np.conj(np.fft.rfft(seg, n)), n)
    corr = corr[:len(ref) - win + 1]
    cs = np.concatenate([[0.0], np.cumsum(ref)])
    cs2 = np.concatenate([[0.0], np.cumsum(ref * ref)])
    mean = (cs[win:] - cs[:-win]) / win
    var = np.maximum((cs2[win:] - cs2[:-win]) / win - mean ** 2, 1e-12)
    corr = corr / (win * np.sqrt(var))
    k = int(np.argmax(corr))
    frac = 0.0
    if 0 < k < len(corr) - 1:  # parabolic sub-frame refinement
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den else 0.0
    conf = float(corr[k] / (np.median(np.abs(corr)) + 1e-12))
    return round(float(lo + k + frac - start) / rate, 3), round(conf, 1)


def offset_checkpoints(ref_env: np.ndarray, dub_env: np.ndarray, *, win_s: float = 30.0,
                       step_s: float = 60.0, search_s: float = 60.0,
                       rate: int = RATE) -> list[dict[str, float]]:
    """Offsets at regular checkpoints over the whole Japanese timeline (start → end)."""
    win, step, search = int(win_s * rate), int(step_s * rate), int(search_s * rate)
    out = []
    for start in range(0, max(1, len(ref_env) - win + 1), step):
        lag, conf = estimate_offset(ref_env, dub_env, start=start, win=win, search=search,
                                    rate=rate)
        out.append({"t": round(start / rate, 2), "offset_s": lag, "confidence": conf})
    return out


def robust_offset(checkpoints: list[dict[str, float]], min_conf: float = 12.0) -> dict[str, Any]:
    """Median of the confident checkpoints, plus drift (max-min) among them."""
    good = [c["offset_s"] for c in checkpoints if c["confidence"] >= min_conf]
    if not good:
        raise RuntimeError("no confident alignment checkpoint; are these the same episode?")
    return {"offset_s": float(np.median(good)), "drift_s": round(max(good) - min(good), 3),
            "confident": len(good), "checkpoints": len(checkpoints)}


def per_second(frames: np.ndarray, rate: int = RATE) -> np.ndarray:
    m = len(frames) // rate
    return frames[:m * rate].reshape(m, rate, *frames.shape[1:]).mean(1)


def shift(bands: np.ndarray, offset_s: float, n: int, rate: int = RATE) -> np.ndarray:
    """Frames of a dub resampled onto the Japanese timeline (``dub[t + offset]``), edge-padded."""
    idx = np.clip(np.arange(n) + round(offset_s * rate), 0, len(bands) - 1)
    return bands[idx]


def dialogue_profile(ja: np.ndarray, dubs: list[np.ndarray],
                     speech_bands: slice = slice(3, 17)) -> tuple[np.ndarray, np.ndarray]:
    """Per-second divergence of each dub from the original in the speech bands (dB).

    Inputs are aligned band-energy frames. Returns ``(score, loudest)``: ``score`` = the
    smallest divergence (high only where *every* dub differs, i.e. dialogue on screen),
    ``loudest`` = the largest (low only where no version has dialogue: a safe cut point).
    """
    n = min(len(ja), *(len(d) for d in dubs))
    divs = []
    for d in dubs:
        d = d[:n] - np.median(d[:n] - ja[:n], axis=0)  # per-band gain match
        divs.append(per_second(np.abs(ja[:n] - d)[:, speech_bands]).mean(1))
    stack = np.stack(divs)
    return stack.min(0), stack.max(0)


def shared_runs(loudest: np.ndarray, *, thresh_db: float = 3.0, min_s: int = 10,
                merge_s: int = 30, pad_s: int = 5) -> list[tuple[int, int]]:
    """Stretches where all versions sound the same (songs, eyecatches, previews), merged."""
    runs: list[list[int]] = []
    start = None
    for i, v in enumerate(list(loudest < thresh_db) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_s:
                if runs and start - runs[-1][1] <= merge_s:
                    runs[-1][1] = i
                else:
                    runs.append([start, i])
            start = None
    return [(max(0, a - pad_s), min(len(loudest), b + pad_s)) for a, b in runs]


def _snap(t: int, loudest: np.ndarray, silences: list[tuple[float, float]], radius: int,
          lo: int, hi: int) -> float:
    """Best cut point near ``t``: a second where no version diverges, ideally inside a silence."""
    best, best_cost = float(t), float("inf")
    for s in range(max(lo, t - radius), min(hi, t + radius) + 1):
        if s >= len(loudest):
            break
        sil = next(((a, b) for a, b in silences if a <= s + 0.5 <= b), None)
        cost = float(loudest[s]) + (0.0 if sil else 4.0) + 0.05 * abs(s - t)
        if cost < best_cost:
            best_cost = cost
            best = round((sil[0] + sil[1]) / 2, 2) if sil and sil[1] - sil[0] < 3 else s + 0.5
    return best


def choose_segments(score: np.ndarray, loudest: np.ndarray,
                    silences: list[tuple[float, float]], *, n: int = 3, length_s: int = 120,
                    exclude: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (), min_len_s: float = 60.0,
                    max_len_s: float = 180.0, snap_s: int = 10, min_gap_s: int = 30,
                    min_score: float = 0.0) -> list[tuple[float, float]]:
    """Greedy: the ``n`` non-overlapping ``length_s`` windows with the highest mean dialogue
    score that avoid ``exclude``, at least ``min_gap_s`` apart (different scenes rather than one
    long one), each boundary snapped to a nearby quiet moment."""
    total = len(score)
    L = min(length_s, total)
    blocked = np.zeros(total, bool)
    for a, b in exclude:
        blocked[max(0, a):min(total, b)] = True
    cs = np.concatenate([[0.0], np.cumsum(score)])
    means = (cs[L:] - cs[:-L]) / L
    chosen: list[tuple[float, float]] = []
    for s in np.argsort(-means, kind="stable"):
        if len(chosen) >= n or means[s] < min_score:
            break
        s = int(s)
        if blocked[s:s + L].any():
            continue
        lo = int(max([e + min_gap_s for _, e in chosen if e <= s] + [0]))
        hi = int(min([b - min_gap_s for b, _ in chosen if b >= s + L] + [total]))
        start = _snap(s, loudest, silences, snap_s, lo, s + L // 2)
        end = _snap(s + L, loudest, silences, snap_s, s + L // 2, hi)
        for a, b in exclude:  # snapping must not walk into a song
            if a < end <= b:
                end = float(a)
            if a <= start < b:
                start = float(b)
        if not (min_len_s <= end - start <= max_len_s):
            continue
        if any(start < b + min_gap_s and a - min_gap_s < end for a, b in chosen):
            continue
        chosen.append((start, end))
        blocked[max(0, int(start) - min_gap_s):int(np.ceil(end)) + min_gap_s] = True
    return sorted(chosen)


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------

def ensure_sources(*, download_missing: bool = True, log: Callable[[str], None] = print,
                   ) -> dict[str, Path]:
    """Local paths of the three videos, downloading the missing ones. A failed download is
    reported and skipped (a missing dub just drops out of the refs); the original is required."""
    out: dict[str, Path] = {}
    for lang, src in SOURCES.items():
        dest = src_dir() / src.filename
        if not dest.exists() and download_missing:
            log(f"spyfamily: downloading {lang} {src.url}")
            try:
                download(src.url, dest)
            except RuntimeError as exc:
                log(f"spyfamily: {lang} unavailable ({exc}); continuing without it")
        if dest.exists():
            out[lang] = dest
    if "ja" not in out:
        raise RuntimeError(f"the Japanese original is missing ({SOURCES['ja'].url}); "
                           f"download it into {src_dir()}")
    return out


def analyse(files: dict[str, Path], *, n: int = 4, length_s: int = 120, force: bool = False,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    """Offsets + segment choice, cached in ``analysis.json`` next to the sources."""
    cache = src_dir() / "analysis.json"
    key = {"version": ANALYSIS_VERSION, "n": n, "length_s": length_s,
           "files": {k: [p.name, p.stat().st_size] for k, p in sorted(files.items())}}
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        if data.get("key") == key:
            return data
    log("spyfamily: decoding audio and measuring offsets")
    bands = {lang: band_energies(decode_mono(p)) for lang, p in files.items()}
    envs = {lang: onset_envelope(b) for lang, b in bands.items()}
    ja_len = len(bands["ja"])
    offsets: dict[str, Any] = {}
    for lang in DUBS:
        if lang not in envs:
            continue
        cps = offset_checkpoints(envs["ja"], envs[lang])
        offsets[lang] = {**robust_offset(cps), "checkpoints_detail": cps}
        log(f"  {lang}: offset {offsets[lang]['offset_s']:+.3f}s, drift "
            f"{offsets[lang]['drift_s']:.3f}s over {offsets[lang]['confident']} checkpoints")
    dub_langs = [lang for lang in DUBS if lang in offsets]
    if not dub_langs:
        raise RuntimeError("no dub available to align against")
    aligned = [shift(bands[lang], offsets[lang]["offset_s"], ja_len) for lang in dub_langs]
    score, loudest = dialogue_profile(bands["ja"], aligned)
    songs = shared_runs(loudest)
    silences = detect_silences(files["ja"])
    segs = choose_segments(score, loudest, silences, n=n, length_s=length_s, exclude=songs)
    segments = []
    for a, b in segs:
        local = {}
        for lang in dub_langs:  # local re-estimate over the clip itself, in case of drift
            lag, conf = estimate_offset(envs["ja"], envs[lang], start=int(a * RATE),
                                        win=int((b - a) * RATE), search=int(60 * RATE))
            glob = offsets[lang]["offset_s"]
            local[lang] = float(lag if conf >= 12 and abs(lag - glob) < 30 else glob)
        segments.append({"start_s": a, "end_s": b, "offsets": local,
                         "dialogue_score": round(float(score[int(a):int(b)].mean()), 2)})
        log(f"  segment {_fmt(a)}–{_fmt(b)} ({b - a:.0f}s) score "
            f"{segments[-1]['dialogue_score']:.2f} offsets {local}")
    data = {"key": key, "durations_s": {k: round(len(v) / RATE, 2) for k, v in bands.items()},
            "offsets": offsets, "songs_s": songs, "silences": len(silences),
            "profile_per_s": {"score": [round(float(v), 2) for v in score],
                              "loudest": [round(float(v), 2) for v in loudest]},
            "segments": segments}
    cache.write_text(json.dumps(data, indent=1))
    return data


# Run as ``python -m`` the module executes twice (package import + __main__): register once.
_register = (builder("spyfamily", capabilities=[CAPABILITY], langs=["ja"],
                     license="official YouTube uploads (Muse Asia / Muse India); "
                             "private evaluation only")
             if __name__ != "__main__" else (lambda fn: fn))


@_register
def spyfamily(lang: str = "ja", n: int | None = 4, *, length_s: int = 120, force: bool = False,
              download_missing: bool = True, log: Callable[[str], None] = print) -> Path:
    """SCENE/ja pack: aligned SPY×FAMILY ep1 clips (ja video+audio, official en/hi dub audio)."""
    if lang != "ja":
        raise ValueError("spyfamily builds the ja SCENE pack only (the dubs are references)")
    n = max(1, min(n or 4, 4))  # `arena pack` passes --n 300 by default: cap at 4 clips
    files = ensure_sources(download_missing=download_missing, log=log)
    data = analyse(files, n=n, length_s=length_s, force=force, log=log)
    root = pack_dir(CAPABILITY, lang)
    clips = root / "clips"
    old = _existing_text_refs(root)
    rows = []
    for i, seg in enumerate(data["segments"], 1):
        a, b = seg["start_s"], seg["end_s"]
        dur = round(b - a, 3)
        item_id = f"{GROUP}-s{i}"
        stem = f"{item_id}-{int(a)}-{int(b)}"
        video, audio = clips / f"{stem}.ja.mp4", clips / f"{stem}.ja.wav"
        if force or not video.exists():
            cut_video(files["ja"], video, a, dur)
        if force or not audio.exists():
            cut_audio(files["ja"], audio, a, dur)
        refs: dict[str, Any] = {}
        for dub in DUBS:
            if dub not in files or dub not in seg["offsets"]:
                continue
            path = clips / f"{stem}.{dub}.wav"
            if force or not path.exists():
                cut_audio(files[dub], path, a + seg["offsets"][dub], dur)
            refs[f"dub_{dub}_audio"] = str(path.relative_to(root))
        refs.update({k: v for k, v in old.get(item_id, {}).items()
                     if k.endswith("_text") and old[item_id].get("_stem") == stem})
        rows.append({
            "id": item_id, "group": GROUP, "split": "test",
            "inputs": {"video": str(video.relative_to(root)),
                       "audio": str(audio.relative_to(root))},
            "refs": refs,
            "meta": {"start_s": a, "end_s": b, "duration_s": dur, "offsets": seg["offsets"],
                     "dialogue_score": seg["dialogue_score"], "source_lang": "ja",
                     "target_langs": [d for d in DUBS if f"dub_{d}_audio" in refs],
                     "sources": {k: SOURCES[k].url for k in files}, "source": "spyfamily"},
        })
    keep = {str(root / p) for r in rows for p in [*r["inputs"].values(), *r["refs"].values()]}
    for stale in clips.glob("*"):  # clips from an earlier segment choice
        if stale.suffix in (".mp4", ".wav") and str(stale) not in keep:
            stale.unlink()
    urls = {k: v.url for k, v in SOURCES.items()}
    return write_pack(CAPABILITY, lang, rows, LICENSE_MD.format(group=GROUP, **urls))


def _existing_text_refs(root: Path) -> dict[str, dict[str, Any]]:
    """Keep transcripts from an earlier ``transcribe_refs`` when the clip itself is unchanged."""
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        return {}
    out = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            stem = Path(row.get("inputs", {}).get("video", "")).name.removesuffix(".ja.mp4")
            out[row["id"]] = {**row.get("refs", {}), "_stem": stem}
    return out


# ---------------------------------------------------------------------------------------------
# Opt-in: transcribe the official dubs with an A4 candidate (runs a model; the user runs it)
# ---------------------------------------------------------------------------------------------

def transcribe_refs(pack: Path | None = None, candidate: str = "qwen3-asr-1.7b", *,
                    langs: tuple[str, ...] = DUBS, params: dict[str, Any] | None = None,
                    log: Callable[[str], None] = print) -> Path:
    """Transcribe each clip's official en/hi dub audio with an A4 arena candidate and write the
    text into the pack as ``refs.dub_en_text`` / ``refs.dub_hi_text`` (plus the timed segments
    under ``refs.dub_<lang>_asr``). One isolated worker process per language via
    :func:`bench.arena.runner.run_worker_job`; ``params`` overrides the candidate's params
    (the default raises ``max_new_tokens`` so a 2-minute clip is not truncated).

    These are machine transcripts of human performances: good enough as C2 references and for
    D-phase comparisons, but audit them before treating them as gold.
    """
    from ..registry import load_candidates, select
    from ..runner import ensure_gpu_room, run_worker_job

    root = Path(pack) if pack else pack_dir(CAPABILITY, "ja")
    manifest = root / "manifest.jsonl"
    rows = [json.loads(ln) for ln in manifest.read_text().splitlines() if ln.strip()]
    cand = select(load_candidates("A4"), [candidate])[0]
    run_params = {**cand.params, "max_new_tokens": 2048, **(params or {})}
    out_dir = root / "refs_asr" / cand.id
    for lang in langs:
        if not cand.supports(lang):
            log(f"transcribe_refs: {cand.id} does not declare {lang}; skipped")
            continue
        items = [{"id": r["id"], "inputs": {"audio": str(root / r["refs"][f"dub_{lang}_audio"])},
                  "meta": {"duration_s": r["meta"].get("duration_s")},
                  "out": str(out_dir / f"{r['id']}_{lang}")}
                 for r in rows if f"dub_{lang}_audio" in r.get("refs", {})]
        if not items:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        ensure_gpu_room(cand.id, cand.requires.vram_gb)
        res = run_worker_job(worker=cand.worker, env=cand.env, params=run_params, lang=lang,
                             items=items, label=f"spyfamily-refs-{lang}-{cand.id}", log=log)
        for r in rows:
            rec = res.records.get(r["id"])
            payload_path = (out_dir / f"{r['id']}_{lang}").with_suffix(".json")
            if not rec or rec.get("status") != "ok" or not payload_path.exists():
                continue
            payload = json.loads(payload_path.read_text())
            r["refs"][f"dub_{lang}_text"] = payload.get("text", "").strip()
            r["refs"][f"dub_{lang}_asr"] = str(payload_path.relative_to(root))
            r["meta"].setdefault("ref_transcripts", {})[lang] = cand.key
    with manifest.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.arena.builders.spyfamily",
                                description=__doc__.split("\n")[0])
    p.add_argument("--segments", type=int, default=4, help="number of clips (1–4)")
    p.add_argument("--length", type=int, default=120, help="nominal clip length in seconds")
    p.add_argument("--force", action="store_true", help="redo the analysis and re-cut clips")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--transcribe-refs", action="store_true",
                   help="only transcribe the dub references of the existing pack (runs a model)")
    p.add_argument("--candidate", default="qwen3-asr-1.7b", help="A4 candidate id")
    a = p.parse_args(argv)
    if a.transcribe_refs:
        print(transcribe_refs(candidate=a.candidate))
        return 0
    print(spyfamily("ja", n=a.segments, length_s=a.length, force=a.force,
                    download_missing=not a.no_download))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
