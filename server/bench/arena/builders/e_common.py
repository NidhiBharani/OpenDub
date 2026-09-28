"""Shared plumbing for the phase-E pack builders (no builder registered here).

- :func:`fleurs_utterances` — FLEURS test utterances (downloaded like ``fleurs.py``) as local wavs.
- :func:`arena_outputs` — ok outputs of another capability (e.g. D1 takes) read from the arena DB,
  joined with that capability's pack items: E1/E2/E4 run on what the voice stage produced.
- :func:`merge_pack` — one pack per (capability, lang) exists, so builders that feed the same
  capability from different sources replace only their own rows (``meta.source``).
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from .. import paths
from ..packs import pack_dir, write_pack
from .fleurs import FLEURS_CODES


def fleurs_utterances(lang: str, n: int | None, seed: int = 0) -> list[dict[str, Any]]:
    """[{id, path, text, group, duration_s, gender}] from the FLEURS test split."""
    code = FLEURS_CODES[lang]
    root = paths.eval_dir() / "_sources" / "fleurs"
    src = root / "data" / code
    tsv, tar = src / "test.tsv", src / "audio" / "test.tar.gz"
    if not tsv.exists() or not tar.exists():
        from huggingface_hub import hf_hub_download

        for f in (f"data/{code}/test.tsv", f"data/{code}/audio/test.tar.gz"):
            hf_hub_download("google/fleurs", f, repo_type="dataset", local_dir=str(root))
    with tsv.open(newline="") as f:
        rows = [r for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 7]
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if n else rows
    out_dir = root / "extracted" / code
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {r[1] for r in rows if not (out_dir / r[1]).exists()}
    if wanted:
        with tarfile.open(tar) as tf:
            for member in tf:
                name = Path(member.name).name
                if member.isfile() and name in wanted:
                    with tf.extractfile(member) as fsrc:  # type: ignore[union-attr]
                        (out_dir / name).write_bytes(fsrc.read())
    return [{"id": f"fleurs-{Path(r[1]).stem}", "path": out_dir / r[1], "text": r[2],
             "group": r[0], "duration_s": int(r[5]) / 16000, "gender": r[6]}
            for r in rows if (out_dir / r[1]).exists()]


def arena_outputs(capability: str, lang: str, candidates: list[str] | None = None,
                  audio_key: str = "audio") -> list[dict[str, Any]]:
    """Ok outputs of ``capability`` in ``lang`` that produced an audio file, with their pack item.

    Returns [{candidate, cand_key, item (packs.Item), audio (abs path), payload}]. Read-only.
    ``candidates`` filters by candidate id.
    """
    from .. import db
    from ..packs import load_pack

    items = {it.id: it for it in load_pack(capability, lang, split=None)}
    con = db.connect()
    rows = con.execute("SELECT * FROM outputs WHERE capability=? AND lang=? AND status='ok'",
                       (capability, lang)).fetchall()
    con.close()
    out = []
    for r in rows:
        if candidates and r["candidate"] not in candidates:
            continue
        it = items.get(r["item"])
        js = db.output_path(r).with_suffix(".json")
        if it is None or not js.exists():
            continue
        payload = json.loads(js.read_text())
        audio = (payload.get("files") or {}).get(audio_key)
        if audio and Path(audio).exists():
            out.append({"candidate": r["candidate"], "cand_key": r["cand_key"], "item": it,
                        "audio": audio, "payload": payload})
    return sorted(out, key=lambda o: (o["item"].id, o["candidate"]))


def merge_pack(capability: str, lang: str, rows: list[dict[str, Any]], license_md: str,
               source: str) -> Path:
    """Write ``rows`` (all tagged ``meta.source = source``) keeping other sources' rows."""
    root = pack_dir(capability, lang)
    keep: list[dict[str, Any]] = []
    manifest = root / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                raw = json.loads(line)
                if (raw.get("meta") or {}).get("source") != source:
                    keep.append(raw)
    for r in rows:
        r.setdefault("meta", {})["source"] = source
    lic_dir = root / "licenses"
    lic_dir.mkdir(parents=True, exist_ok=True)
    (lic_dir / f"{source}.md").write_text(license_md)
    combined = "\n\n".join(p.read_text() for p in sorted(lic_dir.glob("*.md")))
    return write_pack(capability, lang, keep + rows, combined)


def ffmpeg(*args: str) -> None:
    """Run ffmpeg quietly (builders may use it; they never run models)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is required to build this pack")
    subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def atempo_chain(tempo: float) -> str:
    """ffmpeg atempo chain for any tempo (each stage within 0.5–2.0)."""
    stages, f = [], tempo
    while f > 2.0:
        stages.append(2.0)
        f /= 2.0
    while f < 0.5:
        stages.append(0.5)
        f /= 0.5
    stages.append(f)
    return ",".join(f"atempo={s:.6f}" for s in stages)


def read_mono(path: str | Path, sr: int | None = None):
    """(float64 mono, sr), optionally resampled (scipy polyphase)."""
    import math

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr and fs != sr:
        g = math.gcd(int(fs), int(sr))
        x = resample_poly(x, int(sr) // g, int(fs) // g)
        fs = sr
    return np.asarray(x), int(fs)


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))
