"""Per-language speaking-rate model for C-phase duration budgets (stdlib only).

Characters are not comparable across scripts (one kanji ≈ two morae, one Devanagari akshara ≈ one
syllable, English spelling is ≈ 3 letters per syllable), so every C-phase budget and duration
check counts *spoken units* and divides by a per-language rate:

- ``en`` (and other Latin-script languages): syllables, from vowel groups with a silent-e rule;
- ``ja``: morae — each kana is one mora (small ゃゅょ… are not, っ/ー/ん are), each kanji is
  :data:`KANJI_MORAE` morae on average (no reading dictionary at this level);
- ``hi``: syllables — one per independent vowel or consonant that carries a vowel (not followed
  by a virama), minus the word-final inherent schwa (schwa deletion).

Digits are expanded by a per-language estimate; Latin words inside hi/ja text use the English
counter (×1.5 in Japanese, where loanwords gain epenthetic vowels).

Rates are units per second of dubbed dialogue. Defaults come from Pellegrino, Coupé & Marsico
(2011, *Language* 87:3) for en/ja (6.19 syllables/s, 7.84 morae/s); hi is not in that study and
is an estimate (5.6 syllables/s). **They are starting points:** calibrate them per language
with :func:`calibrate` on any pack with text + measured durations (e.g. ``data/eval/A4/<lang>``,
FLEURS read speech, or the official dubs in SCENE), then pass the result as ``rates=``.

Used by the C-phase specs (duration compliance) and by the C-phase workers (budget hints), so
this module must stay importable from any worker env: stdlib only, no package-relative imports.
"""
from __future__ import annotations

import math
import re
import statistics
import unicodedata

UNIT_NAMES = {"en": "syllables", "hi": "syllables", "ja": "morae"}
DEFAULT_RATES = {"en": 6.19, "ja": 7.84, "hi": 5.6}   # units per second (see module docstring)
GENERIC_RATE = 6.0
KANJI_MORAE = 1.9          # mean morae per kanji in running Japanese text (on/kun average)
_DIGIT_UNITS = {"en": 1.4, "hi": 1.6, "ja": 2.0}     # spoken units per digit, rough

_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮヵヶ")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*")
_VOWEL_GROUP = re.compile(r"[aeiouy]+")

# Devanagari code points
_DEV_INDEP_VOWELS = set(range(0x0904, 0x0915)) | {0x0960, 0x0961}
_DEV_CONSONANTS = set(range(0x0915, 0x093A)) | set(range(0x0958, 0x0960)) | {0x0978, 0x0979,
                                                                            0x097A, 0x097B}
_DEV_MATRAS = set(range(0x093E, 0x094D)) | {0x0955, 0x0956, 0x0957, 0x0962, 0x0963}
_DEV_VIRAMA = 0x094D
_DEV_NUKTA = 0x093C


def base_lang(lang: str | None) -> str:
    """``"ja-en"`` → target ``"en"`` is NOT done here; this strips region/task suffixes only:
    ``"hi_IN"``/``"hi.g2p"``/``"ja-JP"`` → ``"hi"``/``"hi"``/``"ja"``."""
    if not lang:
        return ""
    return re.split(r"[._\-]", lang.strip().lower())[0]


def english_syllables(word: str) -> int:
    w = word.lower().strip("'")
    if not w:
        return 0
    if w.isdigit():
        return 0
    n = len(_VOWEL_GROUP.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1                                   # silent e: "make", "time"
    if w.endswith(("ed",)) and n > 1 and len(w) > 3 and w[-3] not in "td":
        n -= 1                                   # "walked" = 1, "wanted" = 2
    if re.search(r"[aeiouy]ing$", w):
        n += 1                                   # hiatus: "going", "seeing"
    return max(1, n)


def _digits_units(token: str, lang: str) -> float:
    digits = sum(ch.isdigit() for ch in token)
    return digits * _DIGIT_UNITS.get(lang, 1.5) if digits else 0.0


def _latin_units(text: str, lang: str) -> float:
    total = 0.0
    for tok in _WORD_RE.findall(text):
        total += _digits_units(tok, lang) if tok[0].isdigit() else english_syllables(tok)
    return total


def _hindi_word_units(word: str) -> float:
    cps = [ord(c) for c in word]
    n = 0
    for i, cp in enumerate(cps):
        if cp in _DEV_INDEP_VOWELS:
            n += 1
        elif cp in _DEV_CONSONANTS:
            j = i + 1
            while j < len(cps) and cps[j] == _DEV_NUKTA:
                j += 1
            nxt = cps[j] if j < len(cps) else None
            if nxt == _DEV_VIRAMA:
                continue                          # half consonant: part of a cluster
            n += 1
            if nxt is None and n > 1:
                n -= 1                            # word-final schwa deletion: "kamal" = ka-mal
    return float(n)


def spoken_units(text: str, lang: str) -> float:
    """Spoken units (syllables, or morae for ja) in ``text`` for language ``lang``."""
    lang = base_lang(lang)
    text = unicodedata.normalize("NFKC", text or "")
    if lang == "ja":
        units = 0.0
        latin: list[str] = []
        for ch in text:
            code = ord(ch)
            if 0x3040 <= code <= 0x30FF:          # hiragana + katakana (incl. ー)
                if ch in _SMALL_KANA or ch in "・゠":
                    continue
                units += 1
            elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or ch in "々〆":
                units += KANJI_MORAE
            elif ch.isascii() and (ch.isalnum() or ch == "'"):
                latin.append(ch)
                continue
            if latin and not (ch.isascii() and ch.isalnum()):
                units += _latin_units("".join(latin), "ja") * (
                    1.0 if "".join(latin).isdigit() else 1.5)
                latin = []
        if latin:
            chunk = "".join(latin)
            units += _latin_units(chunk, "ja") * (1.0 if chunk.isdigit() else 1.5)
        return units
    if lang == "hi":
        units = 0.0
        for tok in re.findall(r"[ऀ-ॿ]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*",
                              text):
            if "ऀ" <= tok[0] <= "ॿ":
                units += _hindi_word_units(tok)
            else:
                units += _latin_units(tok, "hi")
        return units
    if lang == "en" or all(ord(c) < 0x250 for c in text):
        return _latin_units(text, lang)
    # Unknown script: letters / 3 is a rough syllable proxy.
    return sum(ch.isalpha() for ch in text) / 3.0


def rate_for(lang: str, rates: dict[str, float] | None = None) -> float:
    lang = base_lang(lang)
    return (rates or {}).get(lang) or DEFAULT_RATES.get(lang, GENERIC_RATE)


def predicted_seconds(text: str, lang: str, rates: dict[str, float] | None = None) -> float:
    """Predicted spoken duration of ``text`` at the language's dubbing rate."""
    return spoken_units(text, lang) / rate_for(lang, rates)


def budget_units(seconds: float, lang: str, rates: dict[str, float] | None = None) -> int:
    """How many spoken units fit a slot of ``seconds`` (for prompt budget hints)."""
    return max(1, round(max(0.0, seconds) * rate_for(lang, rates)))


def duration_ratio(text: str, lang: str, slot_s: float,
                   rates: dict[str, float] | None = None) -> float | None:
    if not slot_s or slot_s <= 0 or not math.isfinite(slot_s):
        return None
    return predicted_seconds(text, lang, rates) / slot_s


def calibrate(pairs: list[tuple[str, float]], lang: str) -> float:
    """Median units/second over (text, measured speech seconds) pairs — pass speech-only
    durations (VAD-trimmed) where possible, else the result is biased low by pauses."""
    vals = [spoken_units(t, lang) / s for t, s in pairs if s and s > 0.3 and t.strip()]
    return round(statistics.median(vals), 3) if vals else rate_for(lang)
