"""Shared helpers for A5 alignment workers (not a worker; stdlib only).

``tokenize(text, lang)``: whitespace words, or for unspaced scripts (ja/zh) script runs of
[kanji/katakana/latin…][trailing hiragana] — a crude bunsetsu-like unit, the same one the
TextGrid stub builder uses. ``project(tokens, units)``: give every token the time span of the
timed units (ASR words / characters) its characters align to, interpolating tokens that match
nothing between their neighbours, so the output always covers the given text.
"""
from __future__ import annotations

import difflib
import unicodedata

UNSPACED = {"ja", "zh", "th", "lo", "km", "my"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith(("P", "Z", "S")))


def _kind(ch: str) -> str:
    name = unicodedata.name(ch, "")
    if "HIRAGANA" in name:
        return "hira"
    if unicodedata.category(ch).startswith("P") or ch.isspace():
        return "punct"
    return "other"


def tokenize(text: str, lang: str) -> list[str]:
    if lang not in UNSPACED:
        return [w for w in text.split() if _norm(w)]
    toks: list[str] = []
    cur, prev = "", ""
    for ch in text:
        k = _kind(ch)
        if k == "punct":
            if cur:
                toks.append(cur)
            cur, prev = "", ""
            continue
        if cur and k == "other" and prev == "hira":
            toks.append(cur)
            cur = ""
        cur += ch
        prev = k
    if cur:
        toks.append(cur)
    return [t for t in toks if _norm(t)]


def project(tokens: list[str], units: list[dict]) -> list[dict]:
    """tokens → [{start, end, word}] from timed ``units`` ({start, end, word|text})."""
    rs, ro = [], []
    for k, t in enumerate(tokens):
        for ch in _norm(t):
            rs.append(ch)
            ro.append(k)
    hs, ho = [], []
    for k, u in enumerate(units):
        for ch in _norm(str(u.get("word", u.get("text", "")))):
            hs.append(ch)
            ho.append(k)
    spans: dict[int, list[float]] = {}
    sm = difflib.SequenceMatcher(None, "".join(rs), "".join(hs), autojunk=False)
    for blk in sm.get_matching_blocks():
        for i in range(blk.size):
            tk, u = ro[blk.a + i], units[ho[blk.b + i]]
            s, e = float(u["start"]), float(u["end"])
            if tk in spans:
                spans[tk][0] = min(spans[tk][0], s)
                spans[tk][1] = max(spans[tk][1], e)
            else:
                spans[tk] = [s, e]
    out = []
    known = sorted(spans)
    t_end = max((float(u["end"]) for u in units), default=0.0)
    for k, tok in enumerate(tokens):
        if k in spans:
            s, e = spans[k]
        else:  # interpolate between neighbours
            prev = max((j for j in known if j < k), default=None)
            nxt = min((j for j in known if j > k), default=None)
            s = spans[prev][1] if prev is not None else 0.0
            e = spans[nxt][0] if nxt is not None else max(s, t_end)
            gap = [j for j in range(prev + 1 if prev is not None else 0,
                                    nxt if nxt is not None else len(tokens))]
            pos = gap.index(k)
            step = (e - s) / len(gap) if gap else 0.0
            s, e = s + pos * step, s + (pos + 1) * step
        out.append({"start": round(s, 4), "end": round(max(s, e), 4), "word": tok})
    return out
