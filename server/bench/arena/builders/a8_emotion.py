"""A8 packs: speech-emotion subsets (EmoBox protocol), JVNV (ja) and in-house labelled clips.

- ``a8-emobox``: an EmoBox dataset's test fold (github.com/emo-box/EmoBox; jsonl rows ``{key,
  dataset, wav, emo, …}``) with the raw corpus downloaded by the user under
  ``data/eval/_sources/emobox/`` (the EmoBox ``data/`` tree + ``downloads/``). EmoBox has 13
  English sets (CREMA-D ODbL is the default: 91 actors, 6 emotions) and no Japanese or Hindi one.
- ``a8-jvnv``: JVNV (Japanese emotional speech with nonverbal vocalisations, CC BY-SA 4.0), from an
  extracted copy in ``data/eval/_sources/jvnv/``; emotion and speaker are read from the path
  (``F1``/``M2`` … and ``anger``/``happy`` … folders).
- ``a8-labelled-ih``: user clips listed in ``data/eval/_sources/a8_ih/<lang>/labels.csv``
  (``path, emotion?, arousal?, valence?, dominance?, speaker?``; dimensions in [0, 1]), with
  optional same-stem Audacity/TextGrid label files of nonverbal events (laughter, crying …) —
  the only route to hi/ta/te emotion data (IITKGP-SEHSC is by request).

Groups: speakers (the bootstrap unit for the corpus metrics).
"""
from __future__ import annotations

import csv
import json
import random
import re
import shutil
from pathlib import Path

from .. import paths
from . import builder
from ._aphase import audio_files, duration, need, read_labels, source_dir, write_merged

EMOBOX_DEFAULT = {"en": "crema-d"}
JVNV_EMOS = {"anger": "angry", "angry": "angry", "disgust": "disgust", "fear": "fear",
             "happy": "happy", "happiness": "happy", "sad": "sad", "sadness": "sad",
             "surprise": "surprise", "neutral": "neutral"}


def _copy(src: Path, out: Path, rel: str) -> str:
    dst = out / "audio" / rel
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    return f"audio/{rel}"


def _balanced(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    """Up to ``n`` rows, round-robin over emotion classes (keeps macro-F1 informative)."""
    if not n or n >= len(rows):
        return rows
    pyr = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["refs"].get("emotion", ""), []).append(r)
    for v in by.values():
        pyr.shuffle(v)
    out: list[dict] = []
    while len(out) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


@builder("a8-emobox", capabilities=["A8"], langs=["en"], license="per dataset (see LICENSE.md)")
def a8_emobox(lang: str = "en", n: int | None = 600, seed: int = 0, dataset: str | None = None,
              fold: int = 1, source: str | None = None) -> Path:
    """A8 pack from an EmoBox dataset's test fold (user-downloaded corpus)."""
    ds = dataset or EMOBOX_DEFAULT.get(lang)
    if not ds:
        raise ValueError(f"EmoBox has no {lang} dataset; use a8-labelled-ih or a8-jvnv")
    root = need(Path(source) if source else source_dir("emobox"), "EmoBox",
                "Clone github.com/emo-box/EmoBox there and download the raw corpus into its "
                "downloads/ folder per the EmoBox README.")
    split = next(iter(sorted(root.rglob(f"{ds}*test*fold*{fold}*.json*"))), None)
    if split is None:
        raise FileNotFoundError(f"no {ds} test fold {fold} split file under {root}")
    out = paths.eval_dir() / "A8" / lang
    rows = []
    for line in split.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        wav = next((b / r["wav"] for b in (root, root / "downloads", split.parent)
                    if (b / r["wav"]).exists()), None)
        if wav is None:
            continue
        spk = re.split(r"[_\-]", Path(r["key"]).name)[0]
        rows.append({"id": f"emobox-{ds}-{r['key']}".replace("/", "_"), "group": spk,
                     "split": "test", "inputs": {"audio": _copy(wav, out, f"{ds}/{wav.name}")},
                     "refs": {"emotion": r["emo"]},
                     "meta": {"duration_s": r.get("length") or round(duration(wav), 3),
                              "dataset": ds}})
    rows = _balanced(rows, n, seed)
    classes = sorted({r["refs"]["emotion"] for r in rows})
    for r in rows:
        r["meta"]["classes"] = classes
    lic = (f"# EmoBox {ds} test fold {fold} ({lang})\n\nSplit: {split.name} (EmoBox, "
           "github.com/emo-box/EmoBox). Audio: the original corpus under its own licence "
           "(CREMA-D: ODbL; RAVDESS: CC BY-NC-SA 4.0; IEMOCAP / MSP-Podcast: by agreement). "
           "Groups: speaker.\n")
    return write_merged("A8", lang, rows, lic, f"emobox-{ds}")


@builder("a8-jvnv", capabilities=["A8"], langs=["ja"], license="CC BY-SA 4.0")
def a8_jvnv(lang: str = "ja", n: int | None = 600, seed: int = 0,
            source: str | None = None) -> Path:
    """A8 pack: JVNV Japanese emotional speech (user-extracted copy)."""
    root = need(Path(source) if source else source_dir("jvnv"), "JVNV",
                "Download JVNV (CC BY-SA 4.0) and extract it there.")
    out = paths.eval_dir() / "A8" / lang
    rows = []
    for wav in audio_files(root, ("*.wav",)):
        parts = [p.lower() for p in wav.relative_to(root).parts]
        emo = next((JVNV_EMOS[p] for p in parts if p in JVNV_EMOS), None)
        spk = next((p.upper() for p in parts if re.fullmatch(r"[fm]\d+", p)), "unknown")
        if emo is None:
            continue
        rows.append({"id": f"jvnv-{'-'.join(parts)}".replace(".wav", ""), "group": spk,
                     "split": "test",
                     "inputs": {"audio": _copy(wav, out, "jvnv/" + "_".join(parts))},
                     "refs": {"emotion": emo},
                     "meta": {"duration_s": round(duration(wav), 3), "dataset": "jvnv"}})
    rows = _balanced(rows, n, seed)
    for r in rows:
        r["meta"]["classes"] = sorted(set(JVNV_EMOS.values()) - {"neutral"})
    return write_merged("A8", lang, rows, "# JVNV (ja)\n\nJVNV corpus (Xin et al.), CC BY-SA "
                        "4.0, user-downloaded. Groups: speaker.\n", "jvnv")


@builder("a8-labelled-ih", capabilities=["A8"], langs=["en", "hi", "ja", "ta", "te", "ko"],
         license="in-house (user media)")
def a8_labelled_ih(lang: str, n: int | None = None, source: str | None = None) -> Path:
    """A8 pack from user clips with emotion / dimension labels and optional event labels."""
    root = need(Path(source) if source else source_dir("a8_ih") / lang, f"A8 in-house ({lang})",
                "Put clips there with labels.csv (path, emotion, arousal, valence, dominance, "
                "speaker) and optional same-stem event label files.")
    out = paths.eval_dir() / "A8" / lang
    rows = []
    with (root / "labels.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            wav = root / r["path"]
            if not wav.exists():
                continue
            refs: dict = {}
            if r.get("emotion"):
                refs["emotion"] = r["emotion"].strip().lower()
            for d in ("arousal", "valence", "dominance"):
                if r.get(d):
                    refs[d] = float(r[d])
            lab = next((wav.with_suffix(s) for s in (".txt", ".TextGrid")
                        if wav.with_suffix(s).exists()), None)
            if lab is not None:
                refs["events"] = read_labels(lab)
            rows.append({"id": f"ih-{wav.stem}", "group": r.get("speaker") or wav.stem,
                         "split": "test", "inputs": {"audio": _copy(wav, out, f"ih/{wav.name}")},
                         "refs": refs, "meta": {"duration_s": round(duration(wav), 3)}})
            if n and len(rows) >= n:
                break
    classes = sorted({r["refs"]["emotion"] for r in rows if "emotion" in r["refs"]})
    for r in rows:
        r["meta"]["classes"] = classes
    return write_merged("A8", lang, rows, f"# A8 in-house ({lang})\n\nUser media from {root}; "
                        "private, evaluation only. Groups: speaker.\n", "in-house")
