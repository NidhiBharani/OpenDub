"""G3 packs: MOS predictors vs human naturalness ratings (BVCC, SOMOS, TTSDS2 listening test).

Items: inputs {audio}, refs {mos}, group = the synthesis system (system-level structure is what
the bootstrap must respect). All three sets are English; rank within a language only.
"""
from __future__ import annotations

import random
import zipfile
from collections import defaultdict
from pathlib import Path

from . import builder
from ._gcommon import download, merge_pack, sources, user_path

BVCC_LICENSE = """# BVCC (VoiceMOS Challenge 2022 main track), {split} list

Source: user-prepared copy of the BVCC main track (Zenodo record 6572573 + the challenge's
gather scripts, which fetch parts of the audio from Blizzard/VCC sources under their own terms).
{n} utterances; refs.mos = mean listener MOS. Groups: system id. English.
"""
SOMOS_URL = "https://zenodo.org/records/7378801/files/somos.zip?download=1"
SOMOS_LICENSE = """# SOMOS (Samsung Open MOS, Maniati et al., Interspeech 2022), {split} list

Source: https://zenodo.org/records/7378801 (somos.zip), CC BY-NC-SA 4.0 (non-commercial).
{n} utterances (Tacotron-like + LPCNet voices of LJ Speech); refs.mos = mean naturalness.
Groups: system id parsed from the file name (last "_" token; unverified). English.
"""
TTSDS_REPO = "ttsds/listening_test"
TTSDS_LICENSE = """# TTSDS2 listening test (Minixhofer et al., 2025)

Source: https://huggingface.co/datasets/ttsds/listening_test (MIT). {n} synthetic utterances
from 20 systems over four English domains (clean, noisy, wild, kids); refs.mos = mean of the
"mos" ratings per file. Groups: system.
"""


def read_mos_list(path: Path) -> dict[str, float]:
    """``<file>,<mos>`` lines (BVCC/SOMOS style); header lines are skipped."""
    out = {}
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            out[parts[0]] = float(parts[-1])
        except ValueError:
            continue
    return out


def _sample(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


@builder("bvcc", capabilities=["G3"], langs=["en"], license="mixed (Blizzard/VCC sources)")
def bvcc(lang: str = "en", n: int | None = 400, seed: int = 0, root: str | None = None,
         split: str = "test") -> Path:
    """G3 pack from a prepared BVCC main-track copy (DATA/sets/<split>_mos_list.txt)."""
    base = user_path(root, "OPENDUB_BVCC", "BVCC",
                     "Get main.tar.gz from https://zenodo.org/records/6572573, run its gather "
                     "scripts, and pass root=/path/to/main (containing DATA/).")
    data = base / "DATA" if (base / "DATA").exists() else base
    mos = read_mos_list(data / "sets" / f"{split}_mos_list.txt")
    rows = [{"id": f"bvcc-{Path(f).stem}", "group": f.split("-")[0], "split": "test",
             "inputs": {"audio": str(data / "wav" / f)}, "refs": {"mos": round(m, 4)},
             "meta": {"system": f.split("-")[0]}}
            for f, m in sorted(mos.items()) if (data / "wav" / f).exists()]
    if not rows:
        raise FileNotFoundError(f"no BVCC wavs found under {data / 'wav'}")
    rows = _sample(rows, n, seed)
    return merge_pack("G3", lang, rows, "bvcc", BVCC_LICENSE.format(split=split, n=len(rows)))


@builder("somos", capabilities=["G3"], langs=["en"], license="CC BY-NC-SA 4.0")
def somos(lang: str = "en", n: int | None = 400, seed: int = 0, split: str = "test") -> Path:
    """G3 pack from SOMOS (4 GB download from Zenodo)."""
    src = sources("somos")
    zpath = download(SOMOS_URL, src / "somos.zip")
    root = src / "somos"
    if not root.exists():
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(root)
    for inner in root.rglob("*.zip"):  # audios.zip inside the archive
        target = inner.with_suffix("")
        if not target.exists():
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(target)
    lists = sorted(root.rglob(f"{split}_mos_list.txt"))
    lists = [p for p in lists if "clean" in p.parts] or lists
    if not lists:
        raise FileNotFoundError(f"SOMOS: no {split}_mos_list.txt under {root}")
    mos = read_mos_list(lists[0])
    wavs = {p.name: p for p in root.rglob("*.wav")}
    rows = []
    for f, m in sorted(mos.items()):
        name = f if f.endswith(".wav") else f"{f}.wav"
        if name in wavs:
            system = Path(name).stem.split("_")[-1]
            rows.append({"id": f"somos-{Path(name).stem}", "group": system, "split": "test",
                         "inputs": {"audio": str(wavs[name])}, "refs": {"mos": round(m, 4)},
                         "meta": {"system": system}})
    rows = _sample(rows, n, seed)
    return merge_pack("G3", lang, rows, "somos", SOMOS_LICENSE.format(split=split, n=len(rows)))


def ttsds_mos(records: list[dict]) -> dict[str, dict]:
    """Parquet records -> {audio path: {mos, system, domain}} over rating_type == "mos"."""
    acc: dict[str, list[float]] = defaultdict(list)
    info: dict[str, dict] = {}
    for r in records:
        if r.get("rating_type") != "mos":
            continue
        acc[r["audio"]].append(float(r["value"]))
        info[r["audio"]] = {"system": r.get("system"), "domain": r.get("dataset")}
    return {a: {"mos": sum(v) / len(v), **info[a]} for a, v in acc.items()}


@builder("ttsds2_ratings", capabilities=["G3"], langs=["en"], license="MIT")
def ttsds2_ratings(lang: str = "en", n: int | None = 400, seed: int = 0) -> Path:
    """G3 pack from the TTSDS2 listening-test MOS ratings (audio in the same HF dataset)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    src = sources("ttsds2")
    table = pq.read_table(hf_hub_download(TTSDS_REPO, "parquet/data.parquet",
                                          repo_type="dataset", local_dir=str(src)))
    per = ttsds_mos(table.to_pylist())
    chosen = _sample(sorted(per), n, seed)
    rows = []
    for a in chosen:
        wav = Path(hf_hub_download(TTSDS_REPO, a, repo_type="dataset", local_dir=str(src)))
        rows.append({"id": f"ttsds2-{a.replace('/', '-').removesuffix('.wav')}",
                     "group": str(per[a]["system"]), "split": "test",
                     "inputs": {"audio": str(wav)}, "refs": {"mos": round(per[a]["mos"], 4)},
                     "meta": {"system": per[a]["system"], "domain": per[a]["domain"]}})
    return merge_pack("G3", lang, rows, "ttsds2", TTSDS_LICENSE.format(n=len(rows)))
