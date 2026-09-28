"""FLEURS parallel sentences → D-phase voice packs (cross-lingual cloning, duration, stock voices,
VC and S2ST), CC BY 4.0.

FLEURS reads the same FLoRes-101 sentence ids in every language, so a sentence id links:

- a **target sentence** in the pack language (``inputs.text``) and its **human recording**
  (``refs.human_audio``: the ASR floor for ``rt_cer_ratio``; its duration is D3's ``target_s``);
- a **reference voice**: a source-language (e.g. Japanese) FLEURS reading of the SAME sentence
  id (``inputs.ref_audio`` + ``ref_text``): the model hears a Japanese speaker and must say the
  English/Hindi version of that sentence in the same voice (group = the sentence id);
- for D8, the **source recording of the same sentence** (``inputs.audio``) and the target text as
  the reference translation.

Directions (pack language = target language): en ← ja; hi ← ja, en; ja ← en. Split ``test`` holds
the cross-lingual items, split ``control`` the same-language items (reference voice: a
target-language reading of a *different* sentence, shared by ``per_ref`` items, group = that clip)
— run with ``--split control`` for the monolingual baseline.
Capabilities: D1/D4 (clone), D3 (+ target_s, ``compress`` scales it: e.g. [0.8, 1.0, 1.2]),
D5 (text only + gender), D6 (source = target-language recording, target voice = source-language
clip), D8 (source-language recording → target text). Audio lives under the pack's ``audio/<lang>/``.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from . import builder
from ._dvoice import extract, finish, fleurs_rows, pack_root

SOURCES = {"en": ["ja"], "hi": ["ja", "en"], "ja": ["en"]}

LICENSE = """# FLEURS parallel voice pack ({capability}, target {lang})

Source: https://huggingface.co/datasets/google/fleurs (test split: data/<code>/test.tsv and
audio/test.tar.gz of {langs}). License: CC BY 4.0. Read speech of FLoRes-101 sentences, 16 kHz.
Built by bench/arena/builders/d_fleurs_voice.py: target sentence + its human recording
(refs.human_audio) in {lang}; the reference voice is the source-language reading of the same
FLoRes sentence (directions: {pairs}); split `control` uses {lang} readings of other sentences.
Groups: FLoRes sentence id (test) / reference clip (control); FLEURS has no speaker ids. Caveat: FLEURS is likely in public TTS/ASR
training data; compare systems within a language only.
"""


def _by_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["id"], []).append(r)
    return out


def _dur(r: dict[str, Any]) -> float:
    return round(r["samples"] / 16000, 3)


@builder("fleurs_voice", capabilities=["D1", "D3", "D4", "D5", "D6", "D8"],
         langs=["en", "hi", "ja"], license="CC BY 4.0")
def fleurs_voice(lang: str, n: int | None = 200, seed: int = 0, capability: str = "D1",
                 sources: list[str] | None = None, n_control: int = 60, per_ref: int = 2,
                 min_ref_s: float = 3.0, max_ref_s: float = 12.0,
                 compress: list[float] | None = None) -> Path:
    """D-phase pack from FLEURS parallel sentences: ``n`` cross-lingual items per direction plus
    ``n_control`` same-language items (split ``control``)."""
    rng = random.Random(seed)
    cap = capability.upper()
    tgt_rows, tgt_tar = fleurs_rows(lang)
    tgt = _by_id(tgt_rows)
    root = pack_root(cap, lang)
    srcs = [] if cap == "D5" else (sources or SOURCES.get(lang, ["en"]))
    plans: list[tuple[str, str, int]] = [(s, "test", n or 10**9) for s in srcs]
    if cap not in ("D5", "D8") and n_control:
        plans.append((lang, "control", n_control))
    if cap == "D5":
        plans = [(lang, "test", n or 10**9)]

    src_cache: dict[str, tuple[dict[str, list[dict[str, Any]]], Path]] = {lang: (tgt, tgt_tar)}
    rows_out: list[dict[str, Any]] = []
    want: dict[str, set[str]] = {}

    for src, split, count in plans:
        if src not in src_cache:
            r, t = fleurs_rows(src)
            src_cache[src] = (_by_id(r), t)
        src_by_id, _ = src_cache[src]
        ids = sorted(set(tgt) & set(src_by_id)) if src != lang else sorted(tgt)
        rng.shuffle(ids)
        ids = ids[:count]
        # control: reference voices are target-language clips of sentences NOT chosen
        chosen = set(ids)
        pool = [r for i, rs in src_by_id.items() if i not in chosen for r in rs
                if min_ref_s <= _dur(r) <= max_ref_s]
        rng.shuffle(pool)
        for k, sid in enumerate(ids):
            human = rng.choice(tgt[sid])
            if src != lang:  # cross-lingual: the source-language reading of the same sentence
                fits = [r for r in src_by_id[sid] if min_ref_s <= _dur(r) <= max_ref_s]
                ref = rng.choice(fits or src_by_id[sid])
            else:
                ref = pool[(k // max(1, per_ref)) % len(pool)] if pool else None
            pair = f"{src}-{lang}"
            meta = {"src_lang": src, "pair": pair, "duration_s": _dur(human),
                    "gender": human["gender"], "flores_id": sid, "source": "fleurs"}
            want.setdefault(lang, set()).add(human["file"])
            base = {"id": f"fleurs-{pair}-{sid}-{Path(human['file']).stem[-6:]}", "split": split}
            h_rel = f"audio/{lang}/{human['file']}"
            if cap == "D5":
                rows_out.append({**base, "group": sid, "inputs": {"text": human["text"]},
                                 "refs": {"text": human["text"], "human_audio": h_rel},
                                 "meta": meta})
                continue
            if cap == "D8":
                srec = rng.choice(src_by_id[sid])
                want.setdefault(src, set()).add(srec["file"])
                rows_out.append({**base, "group": sid,
                                 "inputs": {"audio": f"audio/{src}/{srec['file']}"},
                                 "refs": {"text": human["text"], "human_audio": h_rel,
                                          "source_text": srec["text"]},
                                 "meta": {**meta, "duration_s": _dur(srec),
                                          "tgt_lang": lang}})
                continue
            if ref is None:
                continue
            want.setdefault(src, set()).add(ref["file"])
            r_rel = f"audio/{src}/{ref['file']}"
            meta.update(ref_gender=ref["gender"], ref_duration_s=_dur(ref))
            group = sid if src != lang else f"{src}:{Path(ref['file']).stem}"
            if cap == "D6":
                rows_out.append({**base, "group": group,
                                 "inputs": {"audio": h_rel, "ref_audio": r_rel,
                                            "text": human["text"]},
                                 "refs": {"text": human["text"]}, "meta": meta})
                continue
            item = {**base, "group": group,
                    "inputs": {"text": human["text"], "ref_audio": r_rel, "ref_text": ref["text"]},
                    "refs": {"text": human["text"], "human_audio": h_rel}, "meta": meta}
            if cap == "D3":
                for f in compress or [1.0]:
                    rows_out.append({**item, "id": item["id"] + ("" if f == 1.0 else f"-x{round(f * 100)}"),
                                     "inputs": {**item["inputs"],
                                                "target_s": round(_dur(human) * f, 3)},
                                     "meta": {**meta, "compress": f}})
            else:
                rows_out.append(item)

    for code, files in want.items():
        _, tar = src_cache[code] if code in src_cache else (None, fleurs_rows(code)[1])
        extract(tar, files, root / "audio" / code)
    have = {p.relative_to(root).as_posix() for p in (root / "audio").rglob("*.wav")} \
        if (root / "audio").exists() else set()
    rows_out = [r for r in rows_out if all(v in have for k, v in {**r["inputs"], **r["refs"]}
                                           .items() if k in ("audio", "ref_audio",
                                                             "human_audio"))]
    pairs = ", ".join(sorted({f"{s}→{lang}" for s, sp, _ in plans if sp == "test"})) or lang
    return finish(cap, lang, rows_out,
                  LICENSE.format(capability=cap, lang=lang, pairs=pairs,
                                 langs=", ".join(sorted({lang, *srcs}))))
