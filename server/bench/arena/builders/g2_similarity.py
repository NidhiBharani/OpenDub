"""G2 packs: speaker-similarity judges vs human similarity ratings (VoxSim, VCC2018).

Items: inputs {audio, ref_audio}, refs {sim: human similarity (higher = more similar)}, group =
the reference speaker (VoxSim) or the converting system (VCC2018).
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

from . import builder
from ._gcommon import download, merge_pack, sources, user_path

VOXSIM_URL = "https://mm.kaist.ac.kr/projects/voxsim/voxsim_test_list.txt"
VOXSIM_LICENSE = """# VoxSim test list (Ahn et al., Interspeech 2024)

Ratings: https://mm.kaist.ac.kr/projects/voxsim/ (voxsim_test_list.txt), CC BY 4.0, research.
Audio: VoxCeleb1 (user-supplied copy; VoxCeleb terms apply). {n} utterance pairs; refs.sim = mean
listener similarity rating per pair. Groups: speaker of the reference clip. Mostly English
celebrity speech (some other languages); labelled "en" for ranking.
"""


def parse_voxsim(lines: list[str]) -> dict[tuple[str, str], float]:
    """Lines ``clip1,clip2,…,rating`` (individual or averaged) -> mean rating per pair."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for line in lines:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 3:
            continue
        try:
            rating = float(parts[-1])
        except ValueError:
            continue  # header
        acc[(parts[0], parts[1])].append(rating)
    return {k: sum(v) / len(v) for k, v in acc.items()}


@builder("voxsim", capabilities=["G2"], langs=["en"], license="CC BY 4.0 (ratings); VoxCeleb1")
def voxsim(lang: str = "en", n: int | None = 400, seed: int = 0,
           voxceleb1: str | None = None) -> Path:
    """G2 pack from the VoxSim test ratings; needs a local VoxCeleb1 wav tree."""
    root = user_path(voxceleb1, "OPENDUB_VOXCELEB1", "VoxSim (VoxCeleb1 audio)",
                     "Download VoxCeleb1 (dev+test wav) per https://www.robots.ox.ac.uk/~vgg/data/"
                     "voxceleb/ and pass voxceleb1=/path/to/wav.")
    lst = download(VOXSIM_URL, sources("voxsim") / "voxsim_test_list.txt")
    pairs = parse_voxsim(lst.read_text().splitlines())
    rows = []
    for (c1, c2), sim in sorted(pairs.items()):
        a, b = root / c1, root / c2
        if a.exists() and b.exists():
            key = hashlib.sha1(f"{c1}|{c2}".encode()).hexdigest()[:12]
            rows.append({"id": f"voxsim-{key}",
                         "group": c1.split("/")[0], "split": "test",
                         "inputs": {"audio": str(b), "ref_audio": str(a)},
                         "refs": {"sim": round(sim, 4)}, "meta": {"pair": [c1, c2]}})
    if not rows:
        raise FileNotFoundError(f"no VoxSim clip found under {root} (expected idXXXXX/<yt>/NNNNN.wav)")
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if n else rows
    return merge_pack("G2", lang, rows, "voxsim", VOXSIM_LICENSE.format(n=len(rows)))


# ------------------------------------------------------------------ VCC2018

VCC_FILES = {  # Edinburgh DataShare handle 10283/3061 (CC BY 4.0)
    "results": ("vcc2018_evaluation_listening_test_raw_results.txt",
                "https://datashare.ed.ac.uk/bitstreams/28bbed68-ddb3-4339-b28d-fc9ed6ec1522/download"),
    "converted": ("vcc2018_submitted_systems_converted_speech.tar.gz",
                  "https://datashare.ed.ac.uk/bitstreams/0096c045-bde7-468d-bc7f-b3247d13492e/download"),
    "reference": ("vcc2018_database_reference.zip",
                  "https://datashare.ed.ac.uk/bitstreams/cdd78f50-fe0a-4708-b576-4b361948c885/download"),
}
VCC_LICENSE = """# VCC2018 similarity listening test (Lorenzo-Trueba et al., Odyssey 2018)

Source: https://datashare.ed.ac.uk/handle/10283/3061 (CC BY 4.0): raw listening-test results,
submitted converted speech, target reference speech. {n} items over {cells} (system, target
speaker) cells. refs.sim = share of "same speaker" judgements (1-4 scale mapped (4-s)/3, averaged)
for that system/target cell — a *system-level* target repeated on each utterance (the raw results
do not identify utterances). Groups: converting system. English.
"""
SYSTEM_RE = re.compile(r"^[NDB]\d\d$")


def parse_vcc_similarity(lines: list[str]) -> dict[tuple[str, str], float]:
    """Raw results -> mean similarity per (system, target speaker) for system-vs-target trials.

    Similarity rows name two stimuli (col 5/6: a system id like N06/D05 and T00 = natural target)
    and carry the 1..4 same/different answer in col 12 (1 = same, sure … 4 = different, sure)."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in csv.reader(lines):
        if len(row) < 12 or not row[11].strip():
            continue
        a, b = row[4].strip(), row[5].strip()
        if "T00" not in (a, b):
            continue
        system = b if a == "T00" else a
        if not SYSTEM_RE.match(system):
            continue
        try:
            score = float(row[11])
        except ValueError:
            continue
        acc[(system, row[7].strip())].append((4.0 - score) / 3.0)
    return {k: sum(v) / len(v) for k, v in acc.items()}


@builder("vcc2018_sim", capabilities=["G2"], langs=["en"], license="CC BY 4.0")
def vcc2018_sim(lang: str = "en", n: int | None = 400, seed: int = 0, per_cell: int = 3) -> Path:
    """G2 pack from the VCC2018 similarity test (system-level targets; ~2 GB download)."""
    src = sources("vcc2018")
    got = {k: download(url, src / name) for k, (name, url) in VCC_FILES.items()}
    cells = parse_vcc_similarity(got["results"].read_text(errors="replace").splitlines())
    conv_dir, ref_dir = src / "converted", src / "reference"
    if not conv_dir.exists():
        with tarfile.open(got["converted"]) as tf:
            tf.extractall(conv_dir, filter="data")
    if not ref_dir.exists():
        with zipfile.ZipFile(got["reference"]) as zf:
            zf.extractall(ref_dir)
    refs_by_spk: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(ref_dir.rglob("*.wav")):
        spk = next((part for part in p.parts if part.startswith("VCC2T")), None)
        if spk:
            refs_by_spk[spk].append(p)
    rng = random.Random(seed)
    rows = []
    wavs = sorted(conv_dir.rglob("*.wav"))
    for (system, tgt), sim in sorted(cells.items()):
        mine = [p for p in wavs if system in p.parts and tgt in str(p)]
        if not mine or not refs_by_spk.get(tgt):
            continue
        for p in rng.sample(mine, min(per_cell, len(mine))):
            same_num = [r for r in refs_by_spk[tgt] if r.stem[-5:] == p.stem[-5:]]
            ref = same_num[0] if same_num else rng.choice(refs_by_spk[tgt])
            rows.append({"id": f"vcc18-{system}-{p.stem}", "group": system, "split": "test",
                         "inputs": {"audio": str(p), "ref_audio": str(ref)},
                         "refs": {"sim": round(sim, 4)},
                         "meta": {"system": system, "target": tgt}})
    if not rows:
        raise RuntimeError(f"VCC2018: could not match converted files under {conv_dir} to "
                           "(system, target) cells; the archive layout differs from the expected "
                           "<system>/…<VCC2Txx>…wav — check and adjust vcc2018_sim")
    rng.shuffle(rows)
    rows = rows[:n] if n else rows
    return merge_pack("G2", lang, rows, "vcc2018",
                      VCC_LICENSE.format(n=len(rows), cells=len({r['group'] + r['meta']['target']
                                                                 for r in rows})))
