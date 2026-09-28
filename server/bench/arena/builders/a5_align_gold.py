"""A5 pack: FLEURS clips + verbatim text, with Praat TextGrid annotation stubs for gold word
boundaries (there is no public word-boundary gold for hi/ja; Buckeye for en is NC and needs a
licence, so it is not fetched here).

Workflow:

1. ``arena pack a5-align-gold --lang hi`` writes the pack and, for each item, a stub
   ``data/eval/_sources/a5_gold/<lang>/<id>.TextGrid`` (never overwritten once it exists) with a
   ``words`` tier pre-filled by proportional interpolation over the voiced span, and a
   ``status`` tier whose single interval reads ``stub``.
2. The user corrects the ``words`` boundaries in Praat and changes ``status`` to ``done``.
3. Re-running the builder puts every ``done`` TextGrid into ``refs.words`` (gold). Target: at
   least ~1,000 boundaries per language (plan §4.10).

Pseudo references (clearly secondary): ``pseudo_dir=<dir of <id>.json with {"words": [...]}>``,
e.g. copied outputs of another aligner, become ``refs.words_pseudo`` for items without gold;
the A5 judge scores them under ``pseudo_*`` metrics only. ``text_source="asr"`` with
``asr_dir=<dir of <id>.json {"text"}>`` swaps the input text for an ASR hypothesis (the
plan's "with ASR text" condition) while keeping the gold words.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .. import paths
from . import builder
from ._aphase import (
    FLEURS_NOTE,
    fleurs_clips,
    read_mono,
    read_textgrid,
    source_dir,
    write,
    write_merged,
    write_textgrid,
)

SR = 16000


def _tokenizer():
    spec = importlib.util.spec_from_file_location("a5_common",
                                                  paths.WORKERS_DIR / "a5_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.tokenize, mod._norm


def stub_words(x, text: str, lang: str) -> list[dict]:
    """Proportional interpolation of the text's tokens over the voiced span (the A5 floor)."""
    import numpy as np

    tokenize, norm = _tokenizer()
    hop = SR // 100
    fr = x[: len(x) // hop * hop].reshape(-1, hop)
    db = 10 * np.log10((fr ** 2).mean(axis=1) + 1e-10)
    act = np.flatnonzero(db > np.percentile(db, 95) - 30)
    t0 = act[0] / 100 if len(act) else 0.0
    t1 = (act[-1] + 1) / 100 if len(act) else len(x) / SR
    toks = tokenize(text, lang)
    w = [max(1, len(norm(t))) for t in toks]
    out, t = [], t0
    for tok, k in zip(toks, w):
        d = (t1 - t0) * k / sum(w)
        out.append({"start": round(t, 3), "end": round(t + d, 3), "label": tok})
        t += d
    return out


@builder("a5-align-gold", capabilities=["A5"], langs=["en", "hi", "ja"], license="CC BY 4.0")
def a5_align_gold(lang: str, n: int | None = 150, seed: int = 0, pseudo_dir: str | None = None,
                  text_source: str = "gold", asr_dir: str | None = None) -> Path:
    """A5 pack + TextGrid stubs; completed (status=done) TextGrids become gold references."""
    clips = fleurs_clips(lang, n or 150, source_dir("fleurs_clips") / lang, seed=seed)
    gold_dir = source_dir("a5_gold") / lang
    out = paths.eval_dir() / "A5" / lang
    rows, n_gold, n_bounds = [], 0, 0
    for c in clips:
        x = read_mono(c["path"], SR)
        rel = Path(write(out / "audio" / f"{c['id']}.wav", x, SR)).relative_to(out)
        tg = gold_dir / f"{c['id']}.TextGrid"
        if not tg.exists():
            write_textgrid(tg, len(x) / SR, {"words": stub_words(x, c["text"], lang),
                                              "status": [{"start": 0.0, "end": len(x) / SR,
                                                          "label": "stub"}]})
        tiers = read_textgrid(tg)
        refs: dict = {}
        status = " ".join(r["label"] for r in tiers.get("status", [])).strip().lower()
        if status == "done":
            words = [{"start": r["start"], "end": r["end"], "word": r["label"]}
                     for r in tiers.get("words", []) if r["label"].strip()]
            refs["words"] = words
            n_gold += 1
            n_bounds += 2 * len(words)
        elif pseudo_dir and (Path(pseudo_dir) / f"{c['id']}.json").exists():
            refs["words_pseudo"] = json.loads((Path(pseudo_dir) / f"{c['id']}.json")
                                              .read_text())["words"]
            refs["pseudo_source"] = str(pseudo_dir)
        text = c["text"]
        if text_source == "asr" and asr_dir and (Path(asr_dir) / f"{c['id']}.json").exists():
            text = json.loads((Path(asr_dir) / f"{c['id']}.json").read_text())["text"]
        suffix = "" if text_source == "gold" else f"-{text_source}"
        rows.append({"id": f"align-{c['id']}{suffix}", "group": c["group"], "split": "test",
                     "inputs": {"audio": str(rel), "text": text}, "refs": refs,
                     "meta": {"duration_s": round(len(x) / SR, 3), "text_source": text_source,
                              "ref_kind": "gold" if "words" in refs else
                              ("pseudo" if refs else "none"),
                              "textgrid": str(tg)}})
    lic = (f"# A5 alignment ({lang})\n\nAudio + text: {FLEURS_NOTE}\nGold word boundaries: "
           f"user-corrected TextGrids in {gold_dir} ({n_gold} items, {n_bounds} boundaries at "
           "build time). Pseudo references, if any, are secondary only.\n")
    return write_merged("A5", lang, rows, lic, f"fleurs-{text_source}")
