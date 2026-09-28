"""C5 — text normalization, G2P and pronunciation lexicon.

One capability, three task families; each gets its own pack so candidates that only do one
task (nemo-text-processing: TN; espeak-ng: G2P; pyopenjtalk: readings) are compared on the
items they can do. Pack language = ``<lang>.<task>``:

- ``en.tn`` / ``hi.tn`` / ``ja.tn`` — written → spoken form. Primary **tn_exact** (sentence
  exact match after case/space/punctuation folding), also per semiotic class
  (``tn_exact.<class>``) and CER.
- ``en.g2p`` / ``hi.g2p`` / ``ja.g2p`` — word → IPA (WikiPron). Primary **per** (phone error
  rate over IPA phones: stress and syllable marks removed, diacritics and length kept on their
  phone), secondary word exact match.
- ``ja.kana`` — Japanese text → katakana reading (homographs, rendaku, counters). Primary
  **kana_cer** (after hiragana→katakana folding), secondary exact match. Accent nucleus
  accuracy needs accent-annotated references (not in the public sets yet).

hi schwa deletion is measured by ``hi.g2p`` (WikiPron broad transcriptions delete it);
homograph/polyphone accuracy by the in-house ``hi.tn``/``ja.kana`` templates (``c5_inhouse``).
"""
from __future__ import annotations

import unicodedata
from typing import Any

from app.pipeline.quality import _distance

from ..judges import Row, Spec, cost, speed
from ..packs import Item

LANGS = ("en", "hi", "ja")
TASKS = ("tn", "g2p", "kana")
PACK_LANGS = [f"{lang}.{task}" for lang in LANGS for task in ("tn", "g2p")] + ["ja.kana"]
SEMIOTIC = ("cardinal", "ordinal", "decimal", "fraction", "date", "time", "money", "measure",
            "telephone", "electronic", "address", "verbatim", "letters", "roman", "whitelist",
            "range", "serial", "math", "other")

_STRIP = set("ˈˌ.‿͜͡|‖ ")
_ATTACH_CATS = ("Mn", "Lm", "Sk")                # combining marks, modifier letters
_ATTACH = set("ːˑʰʷʲˠˤ̃ʼ")


def task_of(item: Item) -> str:
    return item.meta.get("task") or (item.lang.split(".", 1)[1] if "." in item.lang else "tn")


def ipa_phones(text: str) -> list[str]:
    """Split an IPA string into phones (space-separated or not): stress/syllable marks dropped,
    combining diacritics, modifier letters and length marks stay attached to their base."""
    text = unicodedata.normalize("NFD", text or "").replace("g", "ɡ")
    phones: list[str] = []
    for ch in text:
        if ch in _STRIP or ch.isspace():
            continue
        if phones and (unicodedata.category(ch) in _ATTACH_CATS or ch in _ATTACH):
            phones[-1] += ch
        else:
            phones.append(ch)
    # affricates written with a tie bar were split; re-join common pairs
    out: list[str] = []
    for ph in phones:
        if out and out[-1] in ("t", "d") and ph[:1] in ("ʃ", "ʒ", "s", "z", "ɕ", "ʑ"):
            out[-1] += ph
        else:
            out.append(ph)
    return [unicodedata.normalize("NFC", p) for p in out]


def fold_tn(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def to_katakana(text: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFKC", text or ""):
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            ch = chr(code + 0x60)                 # hiragana → katakana
        if 0x30A1 <= ord(ch) <= 0x30FA or ch == "ー":
            out.append(ch)
    return "".join(out)


def c5_scores(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    ref, hyp = str(item.refs.get("text", "")), str(out.get("text", ""))
    if not ref:
        return []
    task = task_of(item)
    if task == "tn":
        r, h = fold_tn(ref), fold_tn(hyp)
        exact = float(r == h)
        rows: list[Row] = [("tn_exact", exact, 1.0)]
        cls = str(item.meta.get("semiotic_class", "")).lower()
        if cls in SEMIOTIC:
            rows.append((f"tn_exact.{cls}", exact, 1.0))
        rc, hc = r.replace(" ", ""), h.replace(" ", "")
        if rc:
            rows.append(("tn_cer", _distance(list(rc), list(hc)) / len(rc), float(len(rc))))
        return rows
    if task == "g2p":
        rp, hp = ipa_phones(ref), ipa_phones(hyp)
        if not rp:
            return []
        return [("per", _distance(rp, hp) / len(rp), float(len(rp))),
                ("g2p_exact", float(rp == hp), 1.0)]
    rk, hk = to_katakana(ref), to_katakana(hyp)
    if not rk:
        return []
    return [("kana_cer", _distance(list(rk), list(hk)) / len(rk), float(len(rk))),
            ("kana_exact", float(rk == hk), 1.0)]


_PRIMARY = {"tn": "tn_exact", "g2p": "per", "kana": "kana_cer"}

SPEC = Spec(
    id="C5",
    title="Text normalization, G2P and lexicon",
    judges={"c5_scores@1": c5_scores, "speed@1": speed, "cost@1": cost},
    primary={"*": "tn_exact", **{pl: _PRIMARY[pl.split(".")[1]] for pl in PACK_LANGS}},
    higher_is_better={"tn_exact": True, "tn_cer": False, "per": False, "g2p_exact": True,
                      "kana_cer": False, "kana_exact": True, "rtfx": True, "cost_usd": False,
                      **{f"tn_exact.{c}": True for c in SEMIOTIC}},
    threshold={"tn_exact": 0.01, "per": 0.005, "kana_cer": 0.005},
    secondary=["tn_cer", "g2p_exact", "kana_exact", "cost_usd",
               *[f"tn_exact.{c}" for c in SEMIOTIC[:10]]],
    packs=["nemo_tn_tests", "polynorm", "wikipron", "c5_inhouse"],
    io="""pack lang = <lang>.<task> (en.tn, hi.tn, ja.tn, en.g2p, hi.g2p, ja.g2p, ja.kana).
item.inputs: {text}; item.refs: {text (spoken form | IPA | katakana)};
item.meta: {task, semiotic_class?, source}. payload: {text}.
LLM workers run with params.task = tn | g2p | kana.""",
)
