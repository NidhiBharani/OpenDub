"""Local subtitle files → C1 and C6 packs (no downloads: the data needs registration or is
private, so the user places it under ``data/eval/_sources/``).

- ``c1_srt_words`` (C1, pack lang = source language): ``_sources/c1_srt/<lang>/<name>.words.json``
  (timed words: an A4/A5 payload or a plain [{start, end, word}] list) plus an optional
  ``<name>.srt`` whose cue boundaries are the gold dub-line/subtitle breaks (e.g. MuST-Cinema
  talks aligned with an A5 aligner, Anim-400K research clips, in-house Indic). Words are cut into
  ≤ 60 s windows at pauses; gold ends are the last word inside each cue (by word midpoint).
- ``c6_srt_pairs`` (C6, pack lang = direction): ``_sources/c6_srt/<src>-<tgt>/<name>.full.srt``
  (uncondensed target-language cues, e.g. a verbatim translation or a C2 output) and
  ``<name>.ref.srt`` (reference subtitles), optional ``<name>.src.srt`` (source-language cues,
  matched by time, shown to condensers as in the HW-TSC recipe). Items are chunks of ~20 cues
  (AppTek's APE chunking); ``refs.cues`` are the reference cues overlapping the chunk's span.
- ``iwslt_subtitling``: the same layout under ``_sources/iwslt26_subtitling/<src>-<tgt>/`` for
  the IWSLT 2026 subtitling data (en→ja is the in-scope pair; registration required,
  non-commercial). IWSLT ships reference SRTs only, so ``<name>.full.srt`` must be produced
  first (e.g. from a C2 run) — the builder says so when it is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..packs import write_pack
from . import builder
from ._ctext import license_md, read_srt, sources_dir, split_direction, windows, words_from_payload


def _gold_ends(words: list[dict], cues: list[dict]) -> list[int]:
    ends = []
    for c in cues:
        inside = [k for k, w in enumerate(words)
                  if c["start"] - 0.05 <= (w["start"] + w["end"]) / 2 <= c["end"] + 0.05]
        if inside:
            ends.append(inside[-1])
    return sorted(set(ends))


@builder("c1_srt_words", capabilities=["C1"], langs=["en", "hi", "ja"],
         license="as placed by the user")
def c1_srt_words(lang: str, max_window_s: float = 60.0, capability: str = "C1") -> Path:
    """C1 items from local timed words (+ gold cue breaks from an SRT)."""
    root = sources_dir("c1_srt") / lang
    files = sorted(root.glob("*.words.json"))
    if not files:
        raise FileNotFoundError(f"put <name>.words.json (+ <name>.srt) under {root}")
    items = []
    for f in files:
        name = f.name.removesuffix(".words.json")
        words = words_from_payload(json.loads(f.read_text()))
        srt = f.with_name(f"{name}.srt")
        gold = _gold_ends(words, read_srt(srt)) if srt.exists() else None
        for k, (a, b) in enumerate(windows(words, max_s=max_window_s)):
            ws = words[a:b]
            refs = {}
            if gold is not None:
                refs["ends"] = [e - a for e in gold if a <= e < b]
            joiner = "" if lang in ("ja", "zh") else " "
            items.append({"id": f"{name}-W{k:03d}", "group": name, "split": "test",
                          "inputs": {"words": ws, "src_lang": lang,
                                     "text": joiner.join(w["word"] for w in ws)},
                          "refs": refs,
                          "meta": {"src_lang": lang, "source": f"c1_srt:{name}",
                                   "duration_s": round(ws[-1]["end"] - ws[0]["start"], 3)}})
    return write_pack(capability, lang, items, license_md(
        f"C1 local timed words ({lang})", str(root), "as placed by the user (check the "
        "source's terms; MuST-Cinema CC BY-NC-ND 4.0, Anim-400K research only)",
        "Groups = source file."))


def _srt_pairs(root: Path, lang: str, chunk: int, source: str) -> list[dict]:
    from ..specs.c6 import LIMITS

    src, tgt = split_direction(lang)
    fulls = sorted(root.glob("*.full.srt"))
    items = []
    for full in fulls:
        name = full.name.removesuffix(".full.srt")
        ref_path = full.with_name(f"{name}.ref.srt")
        if not ref_path.exists():
            continue
        cues, ref = read_srt(full), read_srt(ref_path)
        src_path = full.with_name(f"{name}.src.srt")
        src_cues = read_srt(src_path) if src_path.exists() else []
        for c in cues:
            mid = (c["start"] + c["end"]) / 2
            hit = [s["text"] for s in src_cues if s["start"] <= mid <= s["end"]]
            if hit:
                c["source"] = " ".join(hit).replace("\n", " ")
        for k in range(0, len(cues), chunk):
            part = cues[k:k + chunk]
            lo, hi = part[0]["start"], part[-1]["end"]
            ref_part = [r for r in ref if r["end"] > lo and r["start"] < hi]
            items.append({"id": f"{name}-C{k // chunk:03d}", "group": name, "split": "test",
                          "inputs": {"cues": part, "limits": LIMITS.get(tgt, LIMITS["en"]),
                                     "src_lang": src, "tgt_lang": tgt},
                          "refs": {"cues": ref_part},
                          "meta": {"src_lang": src, "tgt_lang": tgt, "source": source,
                                   "duration_s": round(hi - lo, 3)}})
    return items


@builder("c6_srt_pairs", capabilities=["C6"], langs=["en-ja", "ja-en", "ja-hi", "en-hi"],
         license="as placed by the user")
def c6_srt_pairs(lang: str, chunk: int = 20, capability: str = "C6") -> Path:
    """C6 items from local (uncondensed, reference) SRT pairs."""
    root = sources_dir("c6_srt") / lang
    items = _srt_pairs(root, lang, chunk, "c6_srt")
    if not items:
        raise FileNotFoundError(f"put <name>.full.srt + <name>.ref.srt (+ .src.srt) under {root}")
    return write_pack(capability, lang, items, license_md(
        f"C6 local subtitle pairs ({lang})", str(root), "as placed by the user",
        "Chunks of ~20 cues; groups = file."))


@builder("iwslt_subtitling", capabilities=["C6"], langs=["en-ja"],
         license="IWSLT 2026 subtitling data (registration, non-commercial)")
def iwslt_subtitling(lang: str = "en-ja", chunk: int = 20, capability: str = "C6") -> Path:
    """IWSLT 2026 subtitling (en→ja): local reference SRTs + an uncondensed translation."""
    root = sources_dir("iwslt26_subtitling") / lang
    if not list(root.glob("*.ref.srt")):
        raise FileNotFoundError(f"download the IWSLT 2026 subtitling data (iwslt.org/2026/"
                                f"subtitling; registration) and put <talk>.ref.srt under {root}")
    items = _srt_pairs(root, lang, chunk, "iwslt26")
    if not items:
        raise FileNotFoundError(f"{root}: IWSLT ships reference SRTs only — add "
                                "<talk>.full.srt (an uncondensed translation, e.g. from a C2 "
                                "run) next to each <talk>.ref.srt")
    return write_pack(capability, lang, items, license_md(
        f"IWSLT 2026 subtitling ({lang})", "https://iwslt.org/2026/subtitling",
        "registration required; non-commercial research use — eval-only",
        "ja limits 4 CPS nominal (gated at 6, see specs/c6.py), 13 CPL, 2 lines."))
