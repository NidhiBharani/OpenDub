"""F1 self-reenactment pack: re-drive each original talking-head clip with its own audio.

The original clip is both the input picture and the ground truth (``refs.video``), the standard
protocol of the LatentSync / MuseTalk / KeySync papers. Sources, all read from a media dir
(``media_dir`` opt, ``OPENDUB_F1_MEDIA`` env, default ``data/eval/_sources/f1_selfreenact``):

- ``hdtf/``: HDTF (CC BY 4.0 annotations; the videos are YouTube uploads). With
  ``download=True`` (or ``OPENDUB_F1_DOWNLOAD=1``) the builder fetches the official
  ``MRzzm/HDTF`` url/time/crop lists and downloads with ``yt-dlp``, cutting the first annotated
  interval and the square face crop; otherwise it uses whatever ``hdtf/*.mp4`` exist.
- ``voxceleb2/``: VoxCeleb2 **test** mp4s in the official layout ``…/idXXXXX/<ytid>/NNNNN.mp4``
  (needs the VGG access form; never downloaded here).
- ``lrs3/``: LRS3 **test** mp4s ``…/test/<speaker>/NNNNN.mp4`` (TED/TEDx, research licence).
- ``user/``: any other talking-head clips the user adds (one group per file stem prefix).

Clips are normalised to 25 fps CFR H.264 (≤720p) + 16 kHz mono WAV and trimmed to ``max_s``.
One clip per speaker (the bootstrap unit). Research sets: eval-only.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .. import paths
from . import builder
from ._fvideo import find_media, merge_pack, normalize_clip, one_per_group, probe, sample

SOURCE = "f1_selfreenact"
HDTF_RAW = "https://raw.githubusercontent.com/MRzzm/HDTF/main/HDTF_dataset/{split}_{kind}.txt"
HDTF_SPLITS = ("RD", "WDA", "WRA")

LICENSE = """# F1 self-reenactment ({lang})

Built by `bench/arena/builders/f1_selfreenact.py` from local media under `{media}`.
- HDTF: annotations CC BY 4.0 (github.com/MRzzm/HDTF); videos are third-party YouTube uploads,
  downloaded for evaluation only.
- VoxCeleb2 test (VGG, research use under the dataset terms) and LRS3 test (TED/TEDx, CC BY-NC-ND
  via the LRS3 agreement): eval-only, never redistributed.
- user/: clips supplied by the user; eval-only, private.
Protocol: the original clip is re-driven with its own audio; the original is ground truth.
Groups: speaker (VoxCeleb2 id, LRS3 speaker, HDTF video, user file prefix).
"""


def media_root(media_dir: str | Path | None) -> Path:
    return Path(media_dir or os.environ.get("OPENDUB_F1_MEDIA")
                or paths.eval_dir() / "_sources" / SOURCE).expanduser()


def _hdtf_lists(split: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for kind in ("video_url", "annotion_time", "crop_wh", "resolution"):
        with urllib.request.urlopen(HDTF_RAW.format(split=split, kind=kind), timeout=60) as r:
            text = r.read().decode("utf-8-sig")
        out[kind] = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return out


def _mmss(s: str) -> float:
    parts = [float(x) for x in s.split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def download_hdtf(root: Path, n: int, max_s: float) -> None:
    """Fetch up to ``n`` HDTF clips with yt-dlp (first annotated interval, square face crop)."""
    if not shutil.which("yt-dlp"):
        raise RuntimeError("download=True needs yt-dlp on PATH")
    out_dir = root / "hdtf"
    out_dir.mkdir(parents=True, exist_ok=True)
    got = len(list(out_dir.glob("*.mp4")))
    for split in HDTF_SPLITS:
        lists = _hdtf_lists(split)
        times = {ln.split()[0].removesuffix(".mp4"): ln.split()[1:] for ln in lists["annotion_time"]}
        crops = {ln.split()[0].removesuffix(".mp4"): [int(x) for x in ln.split()[1:5]]
                 for ln in lists["crop_wh"]}
        res = {ln.split()[0].removesuffix(".mp4"): int(ln.split()[1]) for ln in lists["resolution"]}
        for ln in lists["video_url"]:
            if got >= n:
                return
            name, url = ln.split()[:2]
            final = out_dir / f"{split}_{name}.mp4"
            if final.exists() or not times.get(name):
                continue
            start, _, end = times[name][0].partition("-")
            t0, t1 = _mmss(start), _mmss(end)
            raw = out_dir / f".{split}_{name}.raw.mp4"
            height = res.get(name, 1080)
            proc = subprocess.run(["yt-dlp", "-q", "-f", f"bv*[height<={height}]+ba/b", "--merge-output-format",
                                   "mp4", "--download-sections", f"*{t0}-{min(t1, t0 + max_s + 1)}",
                                   "-o", str(raw), url], capture_output=True, text=True,
                                  check=False, timeout=1800)
            if proc.returncode or not raw.exists():
                continue  # unavailable upload: skip, keep going
            crop = crops.get(f"{name}_0")
            vf = "null"
            if crop:
                x, w, y, h = crop
                scale = probe(raw)["height"] / float(height)
                vf = (f"crop={int(w * scale)}:{int(h * scale)}:{int(x * scale)}:"
                      f"{int(y * scale)}")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-vf", vf,
                            "-c:v", "libx264", "-crf", "14", "-c:a", "aac", str(final)],
                           check=False, timeout=1800)
            raw.unlink(missing_ok=True)
            got += final.exists()


def sidecar_text(video: Path) -> str | None:
    """Transcript next to a clip: LRS3's ``NNNNN.txt`` ("Text:  WORDS …") or a plain .txt."""
    txt = video.with_suffix(".txt")
    if not txt.exists():
        return None
    lines = txt.read_text(errors="replace").splitlines()
    for ln in lines:
        if ln.startswith("Text:"):
            return ln.split(":", 1)[1].strip().lower() or None
    return (lines[0].strip() if lines else "") or None


def discover(root: Path) -> list[dict]:
    """Candidate clips with their source and speaker group."""
    found: list[dict] = []
    for p in find_media(root / "hdtf"):
        found.append({"path": p, "source": "hdtf", "group": f"hdtf:{p.stem}"})
    for p in find_media(root / "voxceleb2"):
        spk = next((part for part in p.parts if re.fullmatch(r"id\d{5}", part)), p.parent.name)
        found.append({"path": p, "source": "voxceleb2", "group": f"vox:{spk}"})
    for p in find_media(root / "lrs3"):
        found.append({"path": p, "source": "lrs3", "group": f"lrs3:{p.parent.name}"})
    for p in find_media(root / "user"):
        found.append({"path": p, "source": "user", "group": f"user:{p.stem.split('_')[0]}"})
    return found


@builder(SOURCE, capabilities=["F1"], langs=["en"], license="CC BY 4.0 (HDTF) / research")
def f1_selfreenact(lang: str = "en", n: int | None = 40, seed: int = 0,
                   media_dir: str | None = None, download: bool | None = None,
                   min_s: float = 3.0, max_s: float = 10.0,
                   sources: tuple[str, ...] = ("hdtf", "voxceleb2", "lrs3", "user")) -> Path:
    """F1 self-reenactment pack (original clip re-driven with its own audio; original = GT)."""
    root = media_root(media_dir)
    if download is None:
        download = os.environ.get("OPENDUB_F1_DOWNLOAD") == "1"
    if download:
        download_hdtf(root, n or 40, max_s)
    found = [f for f in discover(root) if f["source"] in sources]
    if not found:
        raise FileNotFoundError(f"no self-reenactment media under {root} "
                                f"(add hdtf/, voxceleb2/, lrs3/ or user/ clips, or download=True)")
    out = paths.eval_dir() / "F1" / lang
    rows = []
    for f in one_per_group(sample(found, None, seed)):
        if n and len(rows) >= n:
            break
        info = probe(f["path"])
        if not info["has_audio"] or info["duration"] < min_s:
            continue
        dur = min(info["duration"], max_s)
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{f['source']}_{f['group'].split(':', 1)[1]}_"
                      f"{f['path'].stem}")
        v, a = out / SOURCE / f"{stem}.mp4", out / SOURCE / f"{stem}.wav"
        if not v.exists() or not a.exists():
            normalize_clip(f["path"], v, a, duration=dur)
        meta = {"kind": "self", "source": f["source"], "duration_s": round(dur, 3),
                "audio_duration_s": round(dur, 3), "fps": 25, "orig": str(f["path"])}
        text = sidecar_text(f["path"])
        if text and dur >= info["duration"] - 0.05:  # a trimmed clip no longer says all of it
            meta["text"] = text  # text-driven generators (LTX DubIt) need the spoken line
        rows.append({"id": f"self-{stem}", "group": f["group"], "split": "test",
                     "inputs": {"video": str(v.relative_to(out)), "audio": str(a.relative_to(out))},
                     "refs": {"video": str(v.relative_to(out))}, "meta": meta})
    return merge_pack("F1", lang, SOURCE, rows, LICENSE.format(lang=lang, media=root))
