"""Helpers shared by the G-phase (judge meta-evaluation) builders. Not a builder module.

Everything here is data plumbing: downloads, user-supplied paths, FLEURS access shared with the
A4 builder's cache, and small numpy signal edits used to inject defects. Nothing runs a model.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from .. import paths
from .fleurs import FLEURS_CODES

SR = 16000


def sources(name: str) -> Path:
    d = paths.eval_dir() / "_sources" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, dest: Path) -> Path:
    """Fetch ``url`` to ``dest`` once (atomic rename; a partial file never looks complete)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "opendub-arena/1"})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)
    return dest


def user_path(value: str | Path | None, env: str, what: str, how: str) -> Path:
    """A dataset the user must obtain themselves (registration, licence form)."""
    raw = value or os.environ.get(env)
    if not raw:
        raise FileNotFoundError(f"{what} needs a local copy: pass the path as an option or set "
                                f"{env}. {how}")
    p = Path(raw).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"{what}: {p} does not exist. {how}")
    return p


# ------------------------------------------------------------------ audio

def read_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Mono float32 at ``sr`` (soundfile when the rate matches, else ffmpeg)."""
    import soundfile as sf

    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        if rate == sr:
            return data.mean(axis=1)
    except RuntimeError:
        pass
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(sr), "-f", "f32le", "-"], capture_output=True,
                         check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def write_audio(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0).astype(np.float32), sr, subtype="PCM_16")
    return path


def shift_audio(x: np.ndarray, offset_ms: float, sr: int = SR) -> np.ndarray:
    """Same length; positive = audio delayed (lags the picture), negative = audio early."""
    n = round(abs(offset_ms) * sr / 1000.0)
    if n == 0:
        return x.copy()
    if n >= len(x):
        return np.zeros_like(x)
    if offset_ms > 0:
        return np.concatenate([np.zeros(n, dtype=x.dtype), x[:-n]])
    return np.concatenate([x[n:], np.zeros(n, dtype=x.dtype)])


def hard_clip(x: np.ndarray, gain_db: float = 18.0, ceiling: float = 0.5) -> np.ndarray:
    """Audible clipping: boost then flatten peaks at ``ceiling`` and renormalise to full scale."""
    y = np.clip(x * 10 ** (gain_db / 20.0), -ceiling, ceiling)
    return y / ceiling * 0.99


def lowpass(x: np.ndarray, cutoff_hz: float, sr: int = SR) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    spec[freqs > cutoff_hz] = 0
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = float(np.mean(x ** 2)) or 1e-8
    return (x + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10.0)), len(x))).astype(np.float32)


def clipped_share(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return float(np.mean(np.abs(x) >= peak * 0.999)) if peak else 0.0


# ------------------------------------------------------------------ FLEURS (shared cache)

def fleurs_rows(lang: str) -> list[dict]:
    """FLEURS test rows (downloads into the same cache as the A4 ``fleurs`` builder)."""
    code = FLEURS_CODES[lang]
    root = paths.eval_dir() / "_sources" / "fleurs"
    tsv = root / "data" / code / "test.tsv"
    tar = root / "data" / code / "audio" / "test.tar.gz"
    if not tsv.exists() or not tar.exists():
        from huggingface_hub import hf_hub_download

        for f in (f"data/{code}/test.tsv", f"data/{code}/audio/test.tar.gz"):
            hf_hub_download("google/fleurs", f, repo_type="dataset", local_dir=str(root))
    with tsv.open(newline="") as f:
        rows = [r for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 7]
    return [{"sid": r[0], "file": r[1], "text": r[2], "samples": int(r[5]), "gender": r[6],
             "lang": lang} for r in rows]


def fleurs_extract(lang: str, names: set[str]) -> Path:
    """Extract the named clips of one language once; returns the directory holding them."""
    code = FLEURS_CODES[lang]
    root = paths.eval_dir() / "_sources" / "fleurs"
    out = root / "extracted" / code
    out.mkdir(parents=True, exist_ok=True)
    missing = {n for n in names if not (out / n).exists()}
    if missing:
        with tarfile.open(root / "data" / code / "audio" / "test.tar.gz") as tf:
            for member in tf:
                name = Path(member.name).name
                if member.isfile() and name in missing:
                    with tf.extractfile(member) as src:  # type: ignore[union-attr]
                        (out / name).write_bytes(src.read())
    return out


def class_weights(labels: list[str]) -> dict[str, float]:
    """Weights making every class count equally in a weighted mean (macro average)."""
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    n, k = len(labels), len(counts)
    return {lab: n / (k * c) for lab, c in counts.items()}


def merge_pack(cap: str, lang: str, rows: list[dict], source: str, license_md: str) -> Path:
    """Write ``rows`` (tagged ``meta.source = source``) into a pack several G builders share:
    items and LICENSE sections from other sources are kept, this source's are replaced."""
    import json

    from ..packs import pack_dir, write_pack

    for r in rows:
        r.setdefault("meta", {})["source"] = source
    root = pack_dir(cap, lang)
    kept = []
    if (root / "manifest.jsonl").exists():
        for line in (root / "manifest.jsonl").read_text().splitlines():
            if line.strip():
                raw = json.loads(line)
                if raw.get("meta", {}).get("source") != source:
                    kept.append(raw)
    sections: dict[str, str] = {}
    if (root / "LICENSE.md").exists():
        for chunk in (root / "LICENSE.md").read_text().split("<!-- source: ")[1:]:
            name, _, body = chunk.partition(" -->\n")
            sections[name] = body.strip()
    sections[source] = license_md.strip()
    lic = "\n\n".join(f"<!-- source: {k} -->\n{v}" for k, v in sorted(sections.items())) + "\n"
    return write_pack(cap, lang, kept + rows, lic)


# ------------------------------------------------------------------ video (ffmpeg)

def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def cut_clip(src: Path, video_out: Path, wav_out: Path, start: float, dur: float) -> None:
    """A silent 25 fps H.264 picture track and its 16 kHz mono audio, both [start, start+dur)."""
    video_out.parent.mkdir(parents=True, exist_ok=True)
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    _ff("-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-an", "-vf", "fps=25",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(video_out))
    _ff("-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-vn", "-ac", "1", "-ar",
        str(SR), str(wav_out))


def mux(video: Path, wav: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _ff("-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "160k", str(out))
    return out


def media_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "default=nw=1:nk=1", str(path)], capture_output=True, text=True,
                         check=True).stdout.strip()
    return float(out)
