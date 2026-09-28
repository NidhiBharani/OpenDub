"""Shared helpers for the phase-F pack builders (video normalisation, pack merging).

Several builders feed the same ``data/eval/F1/<lang>`` pack (self-reenactment, cross-lingual,
the SPY×FAMILY control split), so rows are merged per builder instead of overwriting: each row
carries ``meta.pack_source`` and a builder replaces only its own rows. Licences are kept per
source under ``LICENSE.d/`` and concatenated into ``LICENSE.md``.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..packs import pack_dir, write_pack

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}


def ffmpeg(*args: str, timeout: float = 1800) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to build phase-F packs")
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True,
                          text=True, timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-600:]}")


def probe(path: Path) -> dict[str, Any]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,width,height,r_frame_rate", "-of",
                          "json", str(path)], capture_output=True, text=True, check=False,
                         timeout=120).stdout
    doc = json.loads(out or "{}")
    info: dict[str, Any] = {"duration": float(doc.get("format", {}).get("duration") or 0.0),
                            "width": 0, "height": 0, "fps": 0.0, "has_audio": False}
    for s in doc.get("streams", []):
        if s.get("codec_type") == "video" and not info["width"]:
            num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
            info.update(width=int(s.get("width") or 0), height=int(s.get("height") or 0),
                        fps=float(num) / float(den or 1) if float(den or 1) else 0.0)
        elif s.get("codec_type") == "audio":
            info["has_audio"] = True
    return info


def normalize_clip(src: Path, video_out: Path, audio_out: Path | None, *, start: float = 0.0,
                   duration: float | None = None, fps: float = 25.0,
                   max_height: int = 720) -> None:
    """25 fps CFR H.264 (no audio) + 16 kHz mono WAV of the same window."""
    video_out.parent.mkdir(parents=True, exist_ok=True)
    win = ["-ss", f"{start:.3f}"] + (["-t", f"{duration:.3f}"] if duration else [])
    ffmpeg(*win, "-i", str(src), "-vf", f"fps={fps:g},scale=-2:'min({max_height},ih)'",
           "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-an", str(video_out))
    if audio_out is not None:
        extract_audio(src, audio_out, start=start, duration=duration)


def extract_audio(src: Path, dst: Path, *, start: float = 0.0, duration: float | None = None,
                  sr: int = 16000) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    win = ["-ss", f"{start:.3f}"] + (["-t", f"{duration:.3f}"] if duration else [])
    ffmpeg(*win, "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
           str(dst))


def sample(rows: list[Any], n: int | None, seed: int = 0) -> list[Any]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def one_per_group(rows: list[dict], key: str = "group") -> list[dict]:
    """Keep the first row of every group (independent units for the bootstrap)."""
    seen, out = set(), []
    for r in rows:
        if r[key] not in seen:
            seen.add(r[key])
            out.append(r)
    return out


def merge_pack(capability: str, lang: str, source: str, rows: list[dict[str, Any]],
               license_md: str) -> Path:
    """Replace this builder's rows in the pack, keeping rows other builders wrote."""
    root = pack_dir(capability, lang)
    manifest = root / "manifest.jsonl"
    kept: list[dict[str, Any]] = []
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if (row.get("meta") or {}).get("pack_source") != source:
                    kept.append(row)
    for r in rows:
        r.setdefault("meta", {})["pack_source"] = source
    lic_dir = root / "LICENSE.d"
    lic_dir.mkdir(parents=True, exist_ok=True)
    (lic_dir / f"{source}.md").write_text(license_md)
    combined = "\n\n".join(p.read_text().strip() for p in sorted(lic_dir.glob("*.md")))
    return write_pack(capability, lang, kept + rows, combined + "\n")


def find_media(root: Path, exts: set[str] = VIDEO_EXT) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file())
