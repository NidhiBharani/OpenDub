"""Shared helpers for the A-phase pack builders (not a builder: no ``@builder`` here).

FLEURS clip access (same source and licence as ``fleurs.py``), BS.1770-style loudness, simple
signal generators for fallback beds, and readers for user-annotated label files (Audacity
labels, Praat TextGrids, RTTM). numpy / scipy / soundfile only — builders never run models.
"""
from __future__ import annotations

import csv
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

from .. import paths

SOURCES = "_sources"


def source_dir(name: str) -> Path:
    return paths.eval_dir() / SOURCES / name


def fetch(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` once (``.part`` then rename)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def need(path: Path, what: str, how: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}. {how}")
    return path


# ------------------------------------------------------------------ FLEURS

def fleurs_rows(lang: str, split: str = "test") -> tuple[list[list[str]], Path]:
    """FLEURS tsv rows (id, file_name, raw_transcription, transcription, phonemes, num_samples,
    gender) and the audio tar path; downloads both on first use."""
    from .fleurs import FLEURS_CODES

    code = FLEURS_CODES[lang]
    root = paths.eval_dir() / SOURCES / "fleurs"
    tsv, tar = root / "data" / code / f"{split}.tsv", root / "data" / code / "audio" / \
        f"{split}.tar.gz"
    if not tsv.exists() or not tar.exists():
        from huggingface_hub import hf_hub_download

        for f in (f"data/{code}/{split}.tsv", f"data/{code}/audio/{split}.tar.gz"):
            hf_hub_download("google/fleurs", f, repo_type="dataset", local_dir=str(root))
    with tsv.open(newline="") as f:
        rows = [r for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 7]
    return rows, tar


def fleurs_clips(lang: str, n: int | None, dest: Path, seed: int = 0,
                 split: str = "test") -> list[dict[str, Any]]:
    """``n`` FLEURS utterances extracted into ``dest``: [{id, path, text, duration_s, group,
    gender}] (fixed seed)."""
    rows, tar = fleurs_rows(lang, split)
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if n else rows
    wanted = {r[1] for r in rows}
    dest.mkdir(parents=True, exist_ok=True)
    missing = [w for w in wanted if not (dest / w).exists()]
    if missing:
        with tarfile.open(tar) as tf:
            for member in tf:
                name = Path(member.name).name
                if member.isfile() and name in wanted and not (dest / name).exists():
                    with tf.extractfile(member) as fsrc:  # type: ignore[union-attr]
                        (dest / name).write_bytes(fsrc.read())
    return [{"id": Path(r[1]).stem, "path": dest / r[1], "text": r[2],
             "duration_s": int(r[5]) / 16000, "group": r[0], "gender": r[6]}
            for r in rows if (dest / r[1]).exists()]


FLEURS_NOTE = ("FLEURS (google/fleurs, CC BY 4.0): read speech, 16 kHz. Likely present in public "
               "training data; rank within a language only.")


# ------------------------------------------------------------------ audio

def read_mono(path: str | Path, sr: int):
    import numpy as np
    import soundfile as sf

    x, rate = sf.read(str(path), always_2d=True, dtype="float64")
    x = x.mean(axis=1)
    if rate != sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, rate)
        x = resample_poly(x, sr // g, rate // g)
    return np.asarray(x, dtype=np.float64)


def write(path: Path, x, sr: int) -> str:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="FLOAT")
    return str(path)


def _k_filter(x, sr: int):
    """ITU-R BS.1770 K-weighting (high shelf + high pass), coefficients re-derived for ``sr``
    (the pyloudnorm formulation)."""
    import numpy as np
    from scipy.signal import lfilter

    def biquad(g, q, fc, kind):
        a = 10 ** (g / 40)
        w0 = 2 * np.pi * fc / sr
        alpha = np.sin(w0) / (2 * q)
        cos = np.cos(w0)
        if kind == "shelf":
            b0 = a * ((a + 1) + (a - 1) * cos + 2 * np.sqrt(a) * alpha)
            b1 = -2 * a * ((a - 1) + (a + 1) * cos)
            b2 = a * ((a + 1) + (a - 1) * cos - 2 * np.sqrt(a) * alpha)
            a0 = (a + 1) - (a - 1) * cos + 2 * np.sqrt(a) * alpha
            a1 = 2 * ((a - 1) - (a + 1) * cos)
            a2 = (a + 1) - (a - 1) * cos - 2 * np.sqrt(a) * alpha
        else:  # high pass
            b0, b1, b2 = (1 + cos) / 2, -(1 + cos), (1 + cos) / 2
            a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
        return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0

    b, a = biquad(4.0, 1 / np.sqrt(2), 1500.0, "shelf")
    y = lfilter(b, a, x)
    b, a = biquad(0.0, 0.5, 38.0, "hp")
    return lfilter(b, a, y)


def loudness(x, sr: int) -> float:
    """Gated integrated loudness in LUFS (400 ms blocks, 75 % overlap, −70 LUFS absolute and
    −10 LU relative gates). Mono."""
    import numpy as np

    y = _k_filter(np.asarray(x, dtype=np.float64), sr)
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    if len(y) < blk:
        ms = np.array([np.mean(y ** 2)])
    else:
        ms = np.array([np.mean(y[i:i + blk] ** 2) for i in range(0, len(y) - blk + 1, hop)])
    lk = -0.691 + 10 * np.log10(ms + 1e-12)
    keep = ms[lk > -70]
    if not len(keep):
        return -70.0
    rel = -0.691 + 10 * np.log10(keep.mean()) - 10
    keep = ms[(lk > -70) & (lk > rel)]
    return float(-0.691 + 10 * np.log10(keep.mean() + 1e-12)) if len(keep) else -70.0


def set_loudness(x, sr: int, target: float):
    cur = loudness(x, sr)
    return x * 10 ** ((target - cur) / 20) if cur > -70 else x


def pink_noise(n: int, rng):
    import numpy as np

    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec))
    spec /= np.sqrt(np.maximum(f, 1))
    y = np.fft.irfft(spec, n)
    return y / (np.abs(y).max() + 1e-9)


def synth_music(n: int, sr: int, rng):
    """A crude tonal bed (random triads with decaying partials + a pulse): a no-download stand-in
    for music when no DnR / music source is present."""
    import numpy as np

    t = np.arange(n) / sr
    y = np.zeros(n)
    beat = int(sr * rng.uniform(0.4, 0.7))
    root = rng.uniform(110, 220)
    for start in range(0, n, beat * 4):
        chord = root * np.array([1, 2 ** (rng.choice([3, 4]) / 12), 2 ** (7 / 12)])
        seg = slice(start, min(n, start + beat * 4))
        env = np.exp(-3 * (t[seg] - t[start]))
        for f0 in chord:
            for h in range(1, 5):
                y[seg] += env * np.sin(2 * np.pi * f0 * h * t[seg]) / h
        root *= 2 ** (rng.choice([-5, -2, 0, 2, 5, 7]) / 12)
        root = min(max(root, 90), 260)
    for k in range(0, n, beat):
        y[k:k + int(0.02 * sr)] += 0.5 * rng.standard_normal(min(int(0.02 * sr), n - k))
    return y / (np.abs(y).max() + 1e-9)


def audio_files(root: Path, patterns=("*.wav", "*.flac", "*.mp3", "*.ogg")) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out += root.rglob(pat)
    return sorted(out)


# ------------------------------------------------------------------ label files

def read_audacity(path: Path) -> list[dict[str, Any]]:
    """Audacity label track export: ``start<TAB>end<TAB>label`` per line."""
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and not line.startswith("\\"):
            try:
                segs.append({"start": float(parts[0]), "end": float(parts[1]),
                             "label": parts[2].strip()})
            except ValueError:
                continue
    return segs


def read_textgrid(path: Path, tier: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Long-format Praat TextGrid → {tier name: [{start, end, label}]} (interval tiers; point
    tiers give start == end)."""
    import re

    txt = path.read_text(encoding="utf-8", errors="replace")
    tiers: dict[str, list[dict[str, Any]]] = {}
    for block in re.split(r"\n\s*item \[\d+\]:", txt)[1:]:
        m = re.search(r'name = "(.*?)"', block)
        if not m or (tier and m.group(1) != tier):
            continue
        rows = []
        for a, b, lab in re.findall(
                r'xmin = ([-\d.eE]+)\s*\n\s*xmax = ([-\d.eE]+)\s*\n\s*text = "((?:[^"]|"")*)"',
                block):
            rows.append({"start": float(a), "end": float(b), "label": lab.replace('""', '"')})
        for t, lab in re.findall(r'(?:number|time) = ([-\d.eE]+)\s*\n\s*mark = "((?:[^"]|"")*)"',
                                 block):
            rows.append({"start": float(t), "end": float(t), "label": lab.replace('""', '"')})
        tiers[m.group(1)] = rows
    return tiers


def write_textgrid(path: Path, xmax: float, tiers: dict[str, list[dict[str, Any]]]) -> None:
    """Long-format TextGrid with one IntervalTier per entry (gaps filled with empty intervals)."""
    def esc(s: str) -> str:
        return str(s).replace('"', '""')

    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
           "xmin = 0", f"xmax = {xmax:.4f}", "tiers? <exists>", f"size = {len(tiers)}",
           "item []:"]
    for k, (name, rows) in enumerate(tiers.items(), 1):
        ivs, t = [], 0.0
        for r in sorted(rows, key=lambda r: r["start"]):
            s, e = max(t, float(r["start"])), min(xmax, float(r["end"]))
            if s > t + 1e-6:
                ivs.append((t, s, ""))
            if e > s:
                ivs.append((s, e, r["label"]))
                t = e
        if t < xmax - 1e-6 or not ivs:
            ivs.append((t, xmax, ""))
        out += [f"    item [{k}]:", '        class = "IntervalTier"', f'        name = "{name}"',
                "        xmin = 0", f"        xmax = {xmax:.4f}",
                f"        intervals: size = {len(ivs)}"]
        for j, (s, e, lab) in enumerate(ivs, 1):
            out += [f"        intervals [{j}]:", f"            xmin = {s:.4f}",
                    f"            xmax = {e:.4f}", f'            text = "{esc(lab)}"']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def read_labels(path: Path) -> list[dict[str, Any]]:
    """Audacity .txt or TextGrid (first tier) → segments."""
    if path.suffix.lower() == ".textgrid":
        tiers = read_textgrid(path)
        return [r for r in next(iter(tiers.values()), []) if r["label"].strip()]
    return read_audacity(path)


def duration(path: str | Path) -> float:
    import soundfile as sf

    return float(sf.info(str(path)).duration)


# ------------------------------------------------------------------ pack writing

def write_merged(capability: str, lang: str, rows: list[dict[str, Any]], license_md: str,
                 source: str, merge: bool = True) -> Path:
    """Write ``rows`` into the capability/lang pack, keeping rows other sources wrote there
    (``meta.source`` differs), so e.g. VoxConverse and AMI, or DnR v3 and the remix, share one
    pack. ``LICENSE.md`` keeps one marked section per source."""
    import json

    from ..packs import pack_dir, write_pack

    root = pack_dir(capability, lang)
    kept: list[dict[str, Any]] = []
    sections: dict[str, str] = {}
    if merge and (root / "manifest.jsonl").exists():
        for line in (root / "manifest.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if (row.get("meta") or {}).get("source") != source:
                    kept.append(row)
        lic = (root / "LICENSE.md").read_text() if (root / "LICENSE.md").exists() else ""
        for part in lic.split("<!-- source:")[1:]:
            name, _, body = part.partition(" -->\n")
            sections[name] = body
    for r in rows:
        r.setdefault("meta", {})["source"] = source
    sections[source] = license_md.rstrip() + "\n\n"
    text = "".join(f"<!-- source:{k} -->\n{v}" for k, v in sections.items())
    return write_pack(capability, lang, kept + rows, text)
