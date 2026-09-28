"""Public MT test sets → C2 packs (one builder per source, all pure downloads, no models).

- ``wmt24pp``     google/wmt24pp (Apache-2.0): en→hi (en-hi_IN), en→ja (en-ja_JP); post-edited
                  references, documents from four domains incl. speech; 998 segments per pair.
- ``in22_conv``   ai4bharat/IN22-Conv (CC-BY-4.0, gated): conversational en↔hi, 1,503 turns.
- ``flores_plus`` openlanguagedata/flores_plus (CC-BY-SA-4.0, gated): devtest, parallel across
                  languages, so ja→en, ja→hi, en→hi, hi→en, en→ja are all possible.
- ``bsd``         ryo0634/bsd_ja_en — Business Scene Dialogue (CC-BY-NC-SA-4.0, **eval-only**):
                  ja↔en dialogue with speakers; scenarios originally written in the source
                  language only (no translationese sources) by default.

Every item carries the source line, the direction, up to two previous and one next line of the
same document/dialogue as context, and the reference translation. ``group`` = document.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..packs import write_pack
from . import builder
from ._ctext import license_md, load_hf_dataset, mt_row, sample_groups, split_direction

WMT24PP = {"en-hi": "en-hi_IN", "en-ja": "en-ja_JP"}
FLORES = {"en": "eng_Latn", "hi": "hin_Deva", "ja": "jpn_Jpan"}
DIRECTIONS = ["ja-en", "ja-hi", "en-hi", "hi-en", "en-ja"]


def _rows(repo: str, config: str | None, split: str, gated: bool = False) -> list[dict]:
    """Rows of an HF dataset split (monkeypatched in tests)."""
    return [dict(r) for r in load_hf_dataset(repo, config, split, gated=gated)]


def _doc_items(docs: dict[str, list[tuple[str, str | None, dict]]], src: str, tgt: str,
               source: str, prefix: str) -> list[dict[str, Any]]:
    """docs: doc id → ordered [(source text, reference, extra meta)] → manifest rows."""
    out = []
    for doc, segs in docs.items():
        texts = [s for s, _, _ in segs]
        for k, (text, ref, meta) in enumerate(segs):
            ctx = {"context_before": texts[max(0, k - 2):k], "context_after": texts[k + 1:k + 2]}
            extra = {kk: meta.pop(kk) for kk in ("speaker",) if kk in meta}
            out.append(mt_row(f"{prefix}-{doc}-{k}", str(doc), src, tgt, text, ref, source,
                              context=ctx, meta=meta, **extra))
    return out


@builder("wmt24pp", capabilities=["C2"], langs=list(WMT24PP), license="Apache-2.0")
def wmt24pp(lang: str, n: int | None = 500, seed: int = 0, capability: str = "C2") -> Path:
    """WMT24++ en→hi / en→ja: post-edited references, bad sources dropped."""
    src, tgt = split_direction(lang)
    rows = [r for r in _rows("google/wmt24pp", WMT24PP[lang], "train")
            if not r.get("is_bad_source")]
    rows = sample_groups(rows, n, seed, key=lambda r: str(r["document_id"]))
    docs: dict[str, list] = {}
    for r in sorted(rows, key=lambda r: (str(r["document_id"]), int(r["segment_id"]))):
        docs.setdefault(str(r["document_id"]), []).append(
            (r["source"], r["target"], {"domain": r.get("domain"),
                                        "segment_id": r.get("segment_id")}))
    items = _doc_items(docs, src, tgt, "wmt24pp", "wmt24pp")
    return write_pack(capability, lang, items, license_md(
        f"WMT24++ {lang}", "https://huggingface.co/datasets/google/wmt24pp",
        "Apache-2.0", "Post-edited references (`target`); `is_bad_source` rows removed. "
        "Groups = documents. Likely seen by recent LLMs: rank within a direction only."))


@builder("in22_conv", capabilities=["C2"], langs=["en-hi", "hi-en"], license="CC-BY-4.0")
def in22_conv(lang: str, n: int | None = 500, seed: int = 0, capability: str = "C2") -> Path:
    """IN22-Conv conversational turns, en↔hi (gated: needs HF_TOKEN)."""
    src, tgt = split_direction(lang)
    col = {"en": "sentence_eng_Latn", "hi": "sentence_hin_Deva"}
    rows = _rows("ai4bharat/IN22-Conv", "eng_Latn-hin_Deva", "gen", gated=True)
    rows = sample_groups(rows, n, seed, key=lambda r: str(r["doc_id"]))
    docs: dict[str, list] = {}
    for r in sorted(rows, key=lambda r: (str(r["doc_id"]), int(r["sent_id"]))):
        docs.setdefault(str(r["doc_id"]), []).append(
            (r[col[src]], r[col[tgt]], {"domain": r.get("domain"), "topic": r.get("topic"),
                                        "scenario": r.get("scenario"),
                                        "speaker": r.get("speaker")}))
    items = _doc_items(docs, src, tgt, "in22_conv", "in22conv")
    return write_pack(capability, lang, items, license_md(
        f"IN22-Conv {lang}", "https://huggingface.co/datasets/ai4bharat/IN22-Conv",
        "CC-BY-4.0 (gated: contact form)", "Conversational turns; groups = conversations."))


@builder("flores_plus", capabilities=["C2"], langs=DIRECTIONS, license="CC-BY-SA-4.0")
def flores_plus(lang: str, n: int | None = 500, seed: int = 0, split: str = "devtest",
                capability: str = "C2") -> Path:
    """FLORES+ devtest via parallel sentences (gated: needs HF_TOKEN)."""
    src, tgt = split_direction(lang)
    s_rows = _rows("openlanguagedata/flores_plus", FLORES[src], split, gated=True)
    t_rows = _rows("openlanguagedata/flores_plus", FLORES[tgt], split, gated=True)
    t_by_id = {r.get("id", k): r for k, r in enumerate(t_rows)}
    pairs = []
    for k, r in enumerate(s_rows):
        rid = r.get("id", k)
        if rid in t_by_id:
            pairs.append({"id": rid, "src": r["text"], "tgt": t_by_id[rid]["text"],
                          "topic": r.get("topic"), "url": r.get("url"), "domain": r.get("domain")})
    pairs = sample_groups(pairs, n, seed, key=lambda r: str(r.get("url") or r["id"]))
    docs: dict[str, list] = {}
    for p in sorted(pairs, key=lambda p: int(p["id"]) if str(p["id"]).isdigit() else 0):
        docs.setdefault(str(p.get("url") or p["id"]), []).append(
            (p["src"], p["tgt"], {"topic": p.get("topic"), "domain": p.get("domain"),
                                  "flores_id": p["id"]}))
    items = _doc_items(docs, src, tgt, "flores_plus", "flores")
    for i, it in enumerate(items):
        it["id"] = f"flores-{split}-{it['meta'].get('flores_id', i)}"
    return write_pack(capability, lang, items, license_md(
        f"FLORES+ {split} {lang}", "https://huggingface.co/datasets/openlanguagedata/flores_plus",
        "CC-BY-SA-4.0 (gated)", "Wikipedia-style sentences, translated from English: for ja→* "
        "the Japanese side is itself a translation. Groups = source article (url)."))


@builder("bsd", capabilities=["C2"], langs=["ja-en", "en-ja"], license="CC-BY-NC-SA-4.0")
def bsd(lang: str, n: int | None = 500, seed: int = 0, split: str = "test",
        originals_only: bool = True, capability: str = "C2") -> Path:
    """Business Scene Dialogue ja↔en (eval-only, NC); speakers + dialogue context."""
    src, tgt = split_direction(lang)
    rows = _rows("ryo0634/bsd_ja_en", None, split)
    orig = {"ja": ("ja", "japanese"), "en": ("en", "english")}[src]
    if originals_only:
        rows = [r for r in rows if str(r.get("original_language", "")).lower() in orig]
    rows = sample_groups(rows, n, seed, key=lambda r: str(r["id"]))
    docs: dict[str, list] = {}
    for r in sorted(rows, key=lambda r: (str(r["id"]), int(r["no"]))):
        docs.setdefault(str(r["id"]), []).append(
            (r[f"{src}_sentence"], r[f"{tgt}_sentence"],
             {"speaker": r.get(f"{src}_speaker"), "tag": r.get("tag"),
              "title": r.get("title")}))
    items = _doc_items(docs, src, tgt, "bsd", "bsd")
    return write_pack(capability, lang, items, license_md(
        f"Business Scene Dialogue {lang} ({split})", "https://huggingface.co/datasets/"
        "ryo0634/bsd_ja_en (github.com/tsuruoka-lab/BSD)", "CC-BY-NC-SA-4.0 — eval-only",
        "Dialogue scenarios; groups = scenarios. Only scenarios originally written in the "
        "source language (originals_only=True)."))
