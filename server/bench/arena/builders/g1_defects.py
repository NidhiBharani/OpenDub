"""G1 pack: intelligibility-judge meta-evaluation on real read speech with injected defects.

Every item is a take plus the line it was meant to say; about half are the untouched FLEURS
recording (clean), the rest carry one defect a dubbing reviewer would reject: truncated tail,
dropped middle, the wrong line, mumbled (low-passed + noise), a looped chunk, a reversed
(gibberish) span. ``refs.defective`` is the label the judges' defect scores are ranked against
(detection AUC). Clean items also give each judge's floor CER on real speech.

This is a synthetic stand-in for the plan's in-house 200 labelled takes per language; the
``arena_audit`` builder adds the user's own ok/bad marks on real TTS takes as they accumulate.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from ..packs import pack_dir
from . import builder
from ._gcommon import (
    SR,
    add_noise,
    fleurs_extract,
    fleurs_rows,
    lowpass,
    merge_pack,
    read_audio,
    write_audio,
)

DEFECTS = ("truncated", "dropped_middle", "wrong_line", "mumbled", "looped", "reversed_span")

LICENSE = """# G1 defect pack ({lang}) — injected defects on FLEURS test

Source: https://huggingface.co/datasets/google/fleurs (test split), CC BY 4.0.
Derived: {n} takes; {n_def} carry one synthetic defect ({defects}), the rest are untouched.
Labels are by construction (refs.defective). Groups: FLoRes sentence id.
Caveat: FLEURS is likely in public ASR training data; synthetic defects are not the full range
of TTS failures. Rank within a language; complement with `arena_audit` (the user's marks).
"""


def inject(x: np.ndarray, defect: str, rng: np.random.Generator,
           other: np.ndarray | None = None) -> np.ndarray:
    """One defect applied to a 16 kHz take (pure numpy; ``other`` = a different sentence)."""
    n = len(x)
    if defect == "truncated":
        return x[: int(n * rng.uniform(0.4, 0.7))].copy()
    if defect == "dropped_middle":
        a = int(n * rng.uniform(0.25, 0.4))
        b = a + int(n * rng.uniform(0.25, 0.4))
        return np.concatenate([x[:a], x[b:]])
    if defect == "wrong_line":
        if other is None:
            raise ValueError("wrong_line needs another sentence's audio")
        return other.copy()
    if defect == "mumbled":
        return add_noise(lowpass(x, 700.0), 5.0, rng)
    if defect == "looped":
        a = int(n * rng.uniform(0.2, 0.5))
        seg = x[a: a + int(SR * rng.uniform(0.8, 1.5))]
        return np.concatenate([x[:a], seg, seg, seg, x[a:]])
    if defect == "reversed_span":
        a = int(n * rng.uniform(0.2, 0.4))
        b = a + int(n * rng.uniform(0.3, 0.45))
        return np.concatenate([x[:a], x[a:b][::-1], x[b:]])
    raise ValueError(f"unknown defect {defect!r}")


def build_items(rows: list[dict], audio_dir: Path, out_dir: Path, *, seed: int = 0,
                clean_share: float = 0.5) -> list[dict]:
    """rows: [{sid, file, text}] with the audio in ``audio_dir``; writes takes to out_dir/audio."""
    rng = np.random.default_rng(seed)
    pick = random.Random(seed)
    manifest = []
    for i, r in enumerate(rows):
        x = read_audio(audio_dir / r["file"])
        clean = pick.random() < clean_share
        defect = None if clean else DEFECTS[i % len(DEFECTS)]
        other = None
        if defect == "wrong_line":
            o = rows[(i + 1 + pick.randrange(len(rows) - 1)) % len(rows)]
            if o["sid"] == r["sid"]:
                defect = "truncated"
            else:
                other = read_audio(audio_dir / o["file"])
        y = x if clean else inject(x, defect, rng, other)
        name = f"{Path(r['file']).stem}-{defect or 'clean'}.wav"
        write_audio(out_dir / "audio" / name, y)
        manifest.append({"id": f"g1-{Path(r['file']).stem}", "group": r["sid"], "split": "test",
                         "inputs": {"audio": f"audio/{name}", "text": r["text"]},
                         "refs": {"text": r["text"], "defective": 0 if clean else 1},
                         "meta": {"duration_s": round(len(y) / SR, 3), "defect": defect}})
    return manifest


@builder("g1_defects", capabilities=["G1"], langs=["en", "hi", "ja"], license="CC BY 4.0")
def g1_defects(lang: str, n: int | None = 240, seed: int = 0) -> Path:
    """G1 pack: FLEURS takes, about half with one injected intelligibility defect."""
    rows = fleurs_rows(lang)
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if n else rows
    audio_dir = fleurs_extract(lang, {r["file"] for r in rows})
    out = pack_dir("G1", lang)
    manifest = build_items([r for r in rows if (audio_dir / r["file"]).exists()], audio_dir, out,
                           seed=seed)
    n_def = sum(m["refs"]["defective"] for m in manifest)
    return merge_pack("G1", lang, manifest, "fleurs_defects",
                      LICENSE.format(lang=lang, n=len(manifest), n_def=n_def,
                                     defects=", ".join(DEFECTS)))
