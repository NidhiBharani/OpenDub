"""Shared helpers for the phase-D (voice) pack builders. Not a builder (no ``@builder``).

Builders never run models; they download public data (or read local sources), cut clips with
ffmpeg/soundfile, and write ``data/eval/<ID>/<lang>/manifest.jsonl`` via ``packs.write_pack``.
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from .. import paths
from ..packs import write_pack

FLEURS_CODES = {"en": "en_us", "hi": "hi_in", "ja": "ja_jp", "zh": "cmn_hans_cn", "ko": "ko_kr",
                "de": "de_de", "fr": "fr_fr", "es": "es_419", "pt": "pt_br", "ta": "ta_in",
                "te": "te_in", "bn": "bn_in"}


def source_dir(name: str) -> Path:
    return paths.eval_dir() / "_sources" / name


def hf_file(repo: str, filename: str, local: Path, *, repo_type: str = "dataset",
            revision: str | None = None, gated: bool = False) -> Path:
    """Download one file of a HF repo into ``local`` once (the repo layout is kept)."""
    dest = local / filename
    if dest.exists():
        return dest
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if gated and not token:
        raise RuntimeError(f"{repo} is gated: accept its terms on huggingface.co and set HF_TOKEN")
    hf_hub_download(repo, filename, repo_type=repo_type, revision=revision, local_dir=str(local),
                    token=token)
    return dest


# ------------------------------------------------------------------ FLEURS

def fleurs_rows(lang: str, split: str = "test") -> tuple[list[dict[str, Any]], Path]:
    """FLEURS rows as dicts {id, file, text, norm, samples, gender} and the audio tar, reusing
    the ``data/eval/_sources/fleurs/data/<code>/`` layout of the A4 builder (downloads on first
    use)."""
    code = FLEURS_CODES[lang]
    root = source_dir("fleurs")
    tsv = hf_file("google/fleurs", f"data/{code}/{split}.tsv", root)
    tar = hf_file("google/fleurs", f"data/{code}/audio/{split}.tar.gz", root)
    with tsv.open(newline="") as f:
        rows = [{"id": r[0], "file": r[1], "text": r[2], "norm": r[3], "samples": int(r[5]),
                 "gender": r[6].lower()}
                for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 7]
    return rows, tar


def extract(tar: Path, names: set[str], dest: Path) -> set[str]:
    """Extract the members whose basename is in ``names`` into ``dest``; returns what exists."""
    dest.mkdir(parents=True, exist_ok=True)
    todo = {n for n in names if not (dest / n).exists()}
    if todo:
        with tarfile.open(tar) as tf:
            for m in tf:
                name = Path(m.name).name
                if m.isfile() and name in todo:
                    with tf.extractfile(m) as src:  # type: ignore[union-attr]
                        (dest / name).write_bytes(src.read())
    return {n for n in names if (dest / n).exists()}


# ------------------------------------------------------------------ audio

def cut(src: str | Path, dest: Path, start: float | None = None, end: float | None = None,
        sr: int | None = None) -> Path:
    """Cut [start, end] of an audio/video file to a mono wav (ffmpeg; the user's tool)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None and start is not None:
        cmd += ["-t", f"{max(0.05, end - start):.3f}"]
    cmd += ["-i", str(src), "-vn", "-ac", "1"]
    if sr:
        cmd += ["-ar", str(sr)]
    subprocess.run(cmd + [str(dest)], check=True)
    return dest


def write_audio_bytes(data: bytes, dest: Path, sr: int | None = None) -> Path:
    """Write encoded audio bytes (wav/flac/mp3 from a parquet ``audio`` struct) as wav."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        x, fs = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        sf.write(str(dest), x.mean(axis=1), fs)
    except Exception:  # noqa: BLE001 - mp3/opus: let ffmpeg decode
        tmp = dest.with_suffix(".src")
        tmp.write_bytes(data)
        try:
            cut(tmp, dest, sr=sr)
        finally:
            tmp.unlink(missing_ok=True)
    return dest


def duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def rel(path: Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(root.resolve()))


def pack_root(capability: str, lang: str) -> Path:
    return paths.eval_dir() / capability / lang


def finish(capability: str, lang: str, rows: list[dict[str, Any]], license_md: str) -> Path:
    if not rows:
        raise RuntimeError(f"{capability}/{lang}: the builder produced no items")
    return write_pack(capability, lang, rows, license_md)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
