"""FLEURS test split → A4 packs (read speech, CC BY 4.0, 102 languages incl. en/hi/ja)."""
from __future__ import annotations

import csv
import random
import tarfile
from pathlib import Path

from .. import paths
from ..packs import write_pack
from . import builder

FLEURS_CODES = {"en": "en_us", "hi": "hi_in", "ja": "ja_jp", "es": "es_419", "fr": "fr_fr",
                "de": "de_de", "pt": "pt_br", "zh": "cmn_hans_cn", "ko": "ko_kr", "ta": "ta_in",
                "te": "te_in", "bn": "bn_in"}

FLEURS_LICENSE = """# FLEURS test split ({code})

Source: https://huggingface.co/datasets/google/fleurs (`data/{code}/test.tsv`, `audio/test.tar.gz`)
License: CC BY 4.0. Read speech (FLoRes-101 sentences), 16 kHz. Downloaded for evaluation.
Groups: FLoRes sentence id (several speakers read the same sentence).
Caveat: likely present in public ASR training data; rank within a language only.
"""


@builder("fleurs", capabilities=["A4"], langs=sorted(FLEURS_CODES), license="CC BY 4.0")
def fleurs(lang: str, n: int | None = 300, seed: int = 0, capability: str = "A4") -> Path:
    """A4 pack from FLEURS test: ``n`` utterances sampled with a fixed seed."""
    code = FLEURS_CODES[lang]
    src = paths.eval_dir() / "_sources" / "fleurs" / "data" / code
    tsv, tar = src / "test.tsv", src / "audio" / "test.tar.gz"
    if not tsv.exists() or not tar.exists():
        from huggingface_hub import hf_hub_download

        for f in (f"data/{code}/test.tsv", f"data/{code}/audio/test.tar.gz"):
            hf_hub_download("google/fleurs", f, repo_type="dataset",
                            local_dir=str(paths.eval_dir() / "_sources" / "fleurs"))
    # columns: id, file_name, raw_transcription, transcription, phonemes, num_samples, gender
    with tsv.open(newline="") as f:
        rows = [r for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 7]
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if n else rows
    wanted = {r[1] for r in rows}

    out_dir = paths.eval_dir() / capability / lang
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf:
        for member in tf:
            name = Path(member.name).name
            if member.isfile() and name in wanted and not (audio_dir / name).exists():
                with tf.extractfile(member) as fsrc:  # type: ignore[union-attr]
                    (audio_dir / name).write_bytes(fsrc.read())
    manifest = [{"id": f"fleurs-{Path(r[1]).stem}", "group": r[0], "split": "test",
                 "inputs": {"audio": f"audio/{r[1]}"}, "refs": {"text": r[2]},
                 "meta": {"duration_s": round(int(r[5]) / 16000, 3), "gender": r[6],
                          "source": "fleurs"}}
                for r in rows if (audio_dir / r[1]).exists()]
    return write_pack(capability, lang, manifest, FLEURS_LICENSE.format(code=code))

