"""Human ratings the user has to obtain themselves (registration / licence forms) -> G packs.

For sets this repo cannot download: VoiceMOS Challenge 2026 (T1 naturalness, T2 emotional TTS
EMOS, T3 speaker similarity), LIMMITS (Indic TTS MOS), Blizzard, or in-house ratings. Convert
the answer file to a CSV with a header and point this builder at it::

    audio,ref_audio,score,group,split
    wav/sys01_utt3.wav,ref/spk1.wav,3.75,sys01,test

``audio`` (and ``ref_audio`` for similarity/emotion) are relative to the CSV's folder or
absolute; ``group`` defaults to the audio file's parent folder; ``split`` defaults to test.

Target per capability: G1 ``defective`` (0/1), G2 ``sim``, G3 ``mos``, G4 ``emo``.
"""
from __future__ import annotations

import csv
from pathlib import Path

from . import builder
from ._gcommon import merge_pack, user_path

TARGETS = {"G1": "defective", "G2": "sim", "G3": "mos", "G4": "emo"}
NEEDS_REF = {"G2", "G4"}

LICENSE = """# User-supplied ratings: {name}

Source: {csv} (converted by the user from a set they obtained themselves; its own licence and
terms apply — record them here). {n} items; refs.{target} = the human score.
"""


def read_ratings(csv_path: Path, capability: str) -> list[dict]:
    target = TARGETS[capability]
    base = csv_path.parent
    rows = []
    with csv_path.open(newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            audio = Path(r["audio"]) if Path(r["audio"]).is_absolute() else base / r["audio"]
            inputs = {"audio": str(audio)}
            if capability in NEEDS_REF:
                ref = r.get("ref_audio") or ""
                if not ref:
                    raise ValueError(f"{csv_path}:{i + 2}: {capability} needs ref_audio")
                inputs["ref_audio"] = str(Path(ref) if Path(ref).is_absolute() else base / ref)
            if capability == "G1":
                inputs["text"] = r.get("text", "")
            rows.append({"id": f"user-{csv_path.stem}-{i:05d}",
                         "group": r.get("group") or audio.parent.name,
                         "split": r.get("split") or "test", "inputs": inputs,
                         "refs": {target: float(r["score"]),
                                  **({"text": r["text"]} if r.get("text") else {})},
                         "meta": {}})
    return rows


@builder("user_ratings", capabilities=["G1", "G2", "G3", "G4"],
         langs=["en", "hi", "ja"], license="user-supplied (VMC2026, LIMMITS, in-house …)")
def user_ratings(lang: str, n: int | None = None, csv_path: str | None = None,
                 capability: str = "G3", name: str | None = None) -> Path:
    """Human ratings from a user-converted CSV (VoiceMOS 2026, LIMMITS, in-house) -> G pack."""
    if capability not in TARGETS:
        raise ValueError(f"capability must be one of {', '.join(TARGETS)}")
    path = user_path(csv_path, "OPENDUB_USER_RATINGS", "user_ratings",
                     "Convert the rating set to audio,ref_audio,score,group CSV (see module doc).")
    rows = read_ratings(path, capability)
    rows = rows[:n] if n else rows
    tag = name or path.stem
    return merge_pack(capability, lang, rows, f"user:{tag}",
                      LICENSE.format(name=tag, csv=path, n=len(rows),
                                     target=TARGETS[capability]))
