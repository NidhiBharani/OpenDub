"""G4 pack: emotion-consistency judges on same-vs-different-emotion pairs (ESD, JVNV).

No public cross-lingual "same feeling" ratings exist outside VoiceMOS 2026 T2 (registration;
use ``user_ratings`` with capability=G4). This builder makes pairs from acted emotional corpora
with categorical labels: refs.emo = 1 when both clips carry the same emotion, else 0. Clips in a
pair always differ in speaker and text — the dubbing condition — and, for ESD, the reference
can be a Mandarin speaker while the take is English (``cross_lingual``), the cross-language
setting emotion2vec cosine is reported to fail in.

- ESD (Zhou et al., HLT-NUS): user copy (licence agreement), ``root/<spk>/<Emotion>/*.wav``;
  speakers 0001–0010 Mandarin, 0011–0020 English; emotions Angry, Happy, Neutral, Sad, Surprise.
- JVNV (ja, CC BY-SA 4.0): user copy, ``root/<spk>/<emotion>/…/*.wav`` (emotion = the folder
  directly under the speaker; layout unverified).
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from . import builder
from ._gcommon import merge_pack, user_path

LICENSE = """# Emotion pairs from {corpus}

Source: user-supplied {corpus} copy ({terms}). {n} pairs, {pos} same-emotion / {neg} different;
speakers and texts always differ within a pair{xl}. refs.emo = 1 same emotion, 0 different.
Groups: reference speaker.
"""


def index_corpus(root: Path, corpus: str) -> list[dict]:
    """[{path, spk, emo, lang, text_id}] for every wav under ``root``."""
    out = []
    for wav in sorted(root.rglob("*.wav")):
        rel = wav.relative_to(root).parts
        if len(rel) < 3:
            continue
        spk, emo = rel[0], rel[1]
        if corpus == "esd":
            if not spk.isdigit():
                continue
            lang = "zh" if int(spk) <= 10 else "en"
            num = wav.stem.split("_")[-1]
            # ESD numbers utterances continuously across emotions, 350 sentences per emotion
            text_id = f"{lang}{(int(num) - 1) % 350}" if num.isdigit() else wav.stem
        else:
            lang, text_id = "ja", wav.stem
        out.append({"path": str(wav), "spk": spk, "emo": emo.lower(), "lang": lang,
                    "text_id": text_id})
    return out


def make_pairs(clips: list[dict], lang: str, n: int, seed: int = 0,
               ref_lang: str | None = None) -> list[dict]:
    """Balanced same/different-emotion pairs: take in ``lang``, reference in ``ref_lang``
    (default the same language), different speaker and text."""
    rng = random.Random(seed)
    takes = [c for c in clips if c["lang"] == lang]
    refs = [c for c in clips if c["lang"] == (ref_lang or lang)]
    by_emo: dict[str, list[dict]] = defaultdict(list)
    for c in refs:
        by_emo[c["emo"]].append(c)
    emos = sorted(by_emo)
    if len(emos) < 2 or not takes:
        return []
    pairs, tries = [], 0
    while len(pairs) < n and tries < n * 50:
        tries += 1
        t = rng.choice(takes)
        same = len(pairs) % 2 == 0
        pool = by_emo[t["emo"]] if same else by_emo[rng.choice([e for e in emos if e != t["emo"]])]
        r = rng.choice(pool) if pool else None
        if r is None or r["spk"] == t["spk"] or r["text_id"] == t["text_id"]:
            continue
        pairs.append({"take": t, "ref": r, "same": int(same)})
    return pairs


@builder("emotion_pairs", capabilities=["G4"], langs=["en", "ja"],
         license="ESD: research licence; JVNV: CC BY-SA 4.0")
def emotion_pairs(lang: str, n: int | None = 300, seed: int = 0, corpus: str | None = None,
                  root: str | None = None, cross_lingual: bool = True) -> Path:
    """G4 pack of same/different-emotion pairs from a local ESD (en) or JVNV (ja) copy."""
    corpus = corpus or ("jvnv" if lang == "ja" else "esd")
    how = ("ESD: request it at https://hltsingapore.github.io/ESD/ ; " if corpus == "esd" else
           "JVNV: download from the JVNV corpus page; ") + "pass root=/path/to/corpus."
    base = user_path(root, f"OPENDUB_{corpus.upper()}", corpus.upper(), how)
    clips = index_corpus(base, corpus)
    n = n or 300
    xl = corpus == "esd" and cross_lingual and lang == "en"
    pairs = make_pairs(clips, lang, n // 2 if xl else n, seed)
    if xl:
        pairs += make_pairs(clips, lang, n - len(pairs), seed + 1, ref_lang="zh")
    rows = [{"id": f"{corpus}-{i:05d}", "group": p["ref"]["spk"], "split": "test",
             "inputs": {"audio": p["take"]["path"], "ref_audio": p["ref"]["path"]},
             "refs": {"emo": p["same"]},
             "meta": {"take_emotion": p["take"]["emo"], "ref_emotion": p["ref"]["emo"],
                      "cross_lingual": p["ref"]["lang"] != p["take"]["lang"]}}
            for i, p in enumerate(pairs)]
    if not rows:
        raise RuntimeError(f"no {corpus} pairs for {lang} under {base}")
    pos = sum(r["refs"]["emo"] for r in rows)
    return merge_pack("G4", lang, rows, corpus, LICENSE.format(
        corpus=corpus.upper(), terms="research licence" if corpus == "esd" else "CC BY-SA 4.0",
        n=len(rows), pos=pos, neg=len(rows) - pos,
        xl="; half the references are Mandarin speakers" if xl else ""))
