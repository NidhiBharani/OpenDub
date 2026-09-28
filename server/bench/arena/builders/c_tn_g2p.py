"""C5 packs: text normalization and G2P references (downloads only), plus in-house templates.

Pack languages are ``<lang>.<task>`` (``en.tn``, ``hi.g2p``, ``ja.kana`` …), see specs/c5.py.

- ``nemo_tn_tests``: the NeMo-text-processing test cases (``tests/nemo_text_processing/<lang>/
  data_text_normalization/test_cases_<class>.txt``, ``written~spoken`` per line; Apache-2.0),
  en/hi/ja, grouped by semiotic class. Caveat: nemo-tn is itself a candidate and was developed
  against these cases — its score here is an upper bound; prefer PolyNorm / in-house for ranking.
- ``polynorm``: Apple PolyNorm-Bench (github.com/apple/ml-speech-polynorm-bench, CC-BY-NC-ND-4.0,
  eval-only), en and ja (no hi). The file layout is not documented in this repo's research, so
  the builder accepts any TSV/CSV/JSONL whose columns look like (written, spoken) and fails with
  the list of files it saw otherwise.
- ``wikipron``: WikiPron scraped pronunciations (CUNY-CL/wikipron, Wiktionary data), single
  words: en (US broad), hi (broad), ja (hiragana narrow).
- ``c5_inhouse``: annotation templates for hi/ja semiotic classes and ja readings. The first run
  writes ``data/eval/_sources/c5_inhouse/<pack lang>.tsv`` seeded with written forms; once the
  ``spoken`` column is filled in, the next run builds the pack from the filled rows.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from ..packs import write_pack
from . import builder
from ._ctext import (
    download,
    github_tarball,
    license_md,
    read_text_bytes,
    sample,
    sources_dir,
    tar_members,
)

NEMO_REFS = ["v1.2.0", "r1.2.0", "main"]
WIKIPRON = {"en": "eng_latn_us_broad.tsv", "hi": "hin_deva_broad.tsv",
            "ja": "jpn_hira_narrow.tsv"}
WIKIPRON_URL = "https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/{}"


def _base(lang: str, task: str) -> str:
    base, _, t = lang.partition(".")
    if t and t != task:
        raise ValueError(f"{lang!r} is not a .{task} pack")
    return base


@builder("nemo_tn_tests", capabilities=["C5"], langs=["en.tn", "hi.tn", "ja.tn"],
         license="Apache-2.0")
def nemo_tn_tests(lang: str, n_per_class: int | None = 50, seed: int = 0,
                  capability: str = "C5") -> Path:
    """NeMo-text-processing TN test cases (written~spoken) per semiotic class."""
    base = _base(lang, "tn")
    tgz = github_tarball("NVIDIA/NeMo-text-processing", NEMO_REFS, sources_dir("nemo_tn"))
    marker = f"tests/nemo_text_processing/{base}/data_text_normalization/"
    files = tar_members(tgz, lambda m: marker in m and m.rsplit("/", 1)[-1].startswith(
        "test_cases_") and m.endswith(".txt"))
    items = []
    for name, data in sorted(files.items()):
        cls = name.rsplit("/", 1)[-1].removeprefix("test_cases_").removesuffix(".txt")
        pairs = [ln.split("~", 1) for ln in read_text_bytes(data).splitlines() if "~" in ln]
        for k, (written, spoken) in enumerate(sample(pairs, n_per_class, seed)):
            items.append({"id": f"nemo-{base}-{cls}-{k}", "group": cls, "split": "test",
                          "inputs": {"text": written.strip()}, "refs": {"text": spoken.strip()},
                          "meta": {"task": "tn", "semiotic_class": cls, "lang": base,
                                   "source": "nemo_tn_tests"}})
    if not items:
        raise FileNotFoundError(f"no NeMo TN test cases for {base!r} in {tgz}")
    return write_pack(capability, lang, items, license_md(
        f"NeMo-text-processing TN test cases ({base})",
        "https://github.com/NVIDIA/NeMo-text-processing (tests/nemo_text_processing)",
        "Apache-2.0", "Grouped by semiotic class. nemo-tn was developed against these: its "
        "score is an upper bound."))


def _table_rows(name: str, data: bytes) -> list[dict]:
    text = read_text_bytes(data)
    if name.endswith(".jsonl"):
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    if name.endswith(".json"):
        obj = json.loads(text)
        return obj if isinstance(obj, list) else obj.get("data", [])
    dialect = "excel-tab" if name.endswith(".tsv") else "excel"
    return list(csv.DictReader(io.StringIO(text), dialect=dialect))


_WRITTEN = ("written", "input", "source", "text", "unnormalized", "before")
_SPOKEN = ("spoken", "output", "target", "normalized", "verbalized", "after", "reference")


def _pick(row: dict, keys: tuple[str, ...]) -> str | None:
    low = {str(k).lower(): v for k, v in row.items()}
    for k in keys:
        if k in low and isinstance(low[k], str) and low[k].strip():
            return low[k]
    return None


@builder("polynorm", capabilities=["C5"], langs=["en.tn", "ja.tn"],
         license="CC-BY-NC-ND-4.0 (eval-only)")
def polynorm(lang: str, n: int | None = 500, seed: int = 0, capability: str = "C5") -> Path:
    """Apple PolyNorm-Bench (written → spoken), en/ja; eval-only."""
    base = _base(lang, "tn")
    tgz = github_tarball("apple/ml-speech-polynorm-bench", ["main", "master"],
                         sources_dir("polynorm"))
    locale_tags = {"en": ("en_us", "en-us", "en_", "/en/", "english"),
                   "ja": ("ja_jp", "ja-jp", "ja_", "/ja/", "japanese")}[base]
    files = tar_members(tgz, lambda m: m.lower().endswith((".tsv", ".csv", ".jsonl", ".json"))
                        and any(t in m.lower() for t in locale_tags))
    rows = []
    for name, data in sorted(files.items()):
        for r in _table_rows(name, data):
            w, s = _pick(r, _WRITTEN), _pick(r, _SPOKEN)
            if w and s:
                cls = _pick(r, ("class", "category", "semiotic_class", "type")) or "other"
                rows.append((w, s, cls.lower(), name.rsplit("/", 1)[-1]))
    if not rows:
        raise RuntimeError(f"PolyNorm layout not recognised for {base}; files seen: "
                           f"{sorted(files)[:20]} — adapt builders/c_tn_g2p.py:polynorm")
    items = [{"id": f"polynorm-{base}-{k}", "group": cls, "split": "test",
              "inputs": {"text": w}, "refs": {"text": s},
              "meta": {"task": "tn", "semiotic_class": cls, "lang": base, "file": f,
                       "source": "polynorm"}}
             for k, (w, s, cls, f) in enumerate(sample(rows, n, seed))]
    return write_pack(capability, lang, items, license_md(
        f"PolyNorm-Bench ({base})", "https://github.com/apple/ml-speech-polynorm-bench",
        "CC-BY-NC-ND-4.0 — eval-only, no derivatives redistributed", "Groups = class."))


@builder("wikipron", capabilities=["C5"], langs=["en.g2p", "hi.g2p", "ja.g2p"],
         license="Wiktionary (CC-BY-SA) via WikiPron (Apache-2.0 code)")
def wikipron(lang: str, n: int | None = 1000, seed: int = 0, capability: str = "C5") -> Path:
    """WikiPron single-word pronunciations → IPA references."""
    base = _base(lang, "g2p")
    fname = WIKIPRON[base]
    path = download(WIKIPRON_URL.format(fname), sources_dir("wikipron") / fname)
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2 and parts[0].strip() and " " not in parts[0].strip():
            rows.append((parts[0].strip(), parts[1].strip()))
    seen: dict[str, str] = {}
    for word, ipa in rows:                        # first pronunciation per word
        seen.setdefault(word, ipa)
    chosen = sample(sorted(seen.items()), n, seed)
    items = [{"id": f"wikipron-{base}-{k}", "group": word, "split": "test",
              "inputs": {"text": word}, "refs": {"text": ipa},
              "meta": {"task": "g2p", "lang": base, "source": f"wikipron:{fname}"}}
             for k, (word, ipa) in enumerate(chosen)]
    return write_pack(capability, lang, items, license_md(
        f"WikiPron {fname}", "https://github.com/CUNY-CL/wikipron (data/scrape/tsv)",
        "Wiktionary content (CC-BY-SA); WikiPron code Apache-2.0",
        "Single words, first listed pronunciation; phones space-separated."))


# ------------------------------------------------------------------ in-house templates

SEEDS: dict[str, list[tuple[str, str]]] = {
    "hi.tn": [
        ("cardinal", "मेरे पास 25 किताबें हैं।"), ("cardinal", "इस गाँव में 1,250 लोग रहते हैं।"),
        ("cardinal", "कुल 3 लाख रुपये लगे।"),
        ("ordinal", "वह कक्षा में 1st आया।"), ("ordinal", "यह उसका 5वाँ प्रयास था।"),
        ("decimal", "तापमान 36.6 डिग्री था।"), ("decimal", "दर 7.25% है।"),
        ("fraction", "आधा कप यानी 1/2 कप दूध डालें।"), ("fraction", "3/4 काम पूरा हो गया।"),
        ("date", "मेरा जन्म 15/08/1995 को हुआ।"), ("date", "बैठक 2 अक्टूबर 2026 को है।"),
        ("time", "ट्रेन 7:45 बजे आएगी।"), ("time", "दुकान सुबह 9 बजे से रात 10:30 तक खुली है।"),
        ("money", "इसकी कीमत ₹499 है।"), ("money", "उसने $20 दिए।"),
        ("measure", "दूरी 12 km है।"), ("measure", "वज़न 2.5 kg है।"),
        ("telephone", "मुझे 98765 43210 पर कॉल करें।"),
        ("electronic", "ईमेल info@example.com पर भेजें।"),
        ("letters", "वह IIT में पढ़ता है।"), ("letters", "RBI ने दरें बदलीं।"),
    ],
    "ja.tn": [
        ("cardinal", "りんごが3個あります。"), ("cardinal", "参加者は1,234人でした。"),
        ("cardinal", "人口は約120万人です。"),
        ("ordinal", "彼は第2位でした。"), ("ordinal", "3番目の角を右です。"),
        ("decimal", "体温は36.8度です。"), ("decimal", "金利は0.25%です。"),
        ("fraction", "全体の3/4が終わりました。"),
        ("date", "2026年9月28日に出発します。"), ("date", "締め切りは10/5です。"),
        ("time", "会議は14:30からです。"), ("time", "午前7時15分に起きた。"),
        ("money", "チケットは¥3,500です。"), ("money", "$20払いました。"),
        ("measure", "距離は5kmです。"), ("measure", "重さは2.5kgです。"),
        ("telephone", "03-1234-5678に電話してください。"),
        ("electronic", "info@example.comまで連絡してください。"),
        ("letters", "NHKのニュースを見た。"), ("letters", "AIの研究をしています。"),
    ],
    "ja.kana": [
        ("homograph", "今日は雨ですね。"), ("homograph", "今日は、皆さん。"),
        ("homograph", "一日中寝ていた。"), ("homograph", "四月一日に入学した。"),
        ("homograph", "彼は料理が上手だ。"), ("homograph", "相手の方が一枚上手だった。"),
        ("homograph", "市場で魚を買った。"), ("homograph", "株式市場が荒れている。"),
        ("homograph", "大人気のアニメだ。"), ("homograph", "大人気ない態度だ。"),
        ("homograph", "何人来ましたか。"), ("homograph", "彼は何人ですか。"),
        ("rendaku", "山登りが好きだ。"), ("rendaku", "本棚に置いた。"),
        ("counter", "犬が三匹いる。"), ("counter", "鉛筆を六本買った。"),
        ("counter", "八百屋で野菜を買う。"), ("name", "黄昏は西国の諜報員だ。"),
        ("name", "ロイド・フォージャーは精神科医だ。"), ("name", "アーニャは超能力者だ。"),
    ],
}
TEMPLATE_HEADER = ["id", "semiotic_class", "written", "spoken", "notes"]


@builder("c5_inhouse", capabilities=["C5"], langs=sorted(SEEDS),
         license="in-house (OpenDub annotations)")
def c5_inhouse(lang: str, capability: str = "C5") -> Path:
    """In-house hi/ja semiotic-class TN and ja reading items from an annotation template."""
    if lang not in SEEDS:
        raise ValueError(f"no in-house template for {lang!r} (have {sorted(SEEDS)})")
    tsv = sources_dir("c5_inhouse") / f"{lang}.tsv"
    if not tsv.exists():
        with tsv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(TEMPLATE_HEADER)
            for k, (cls, written) in enumerate(SEEDS[lang]):
                w.writerow([f"{lang}-{k:03d}", cls, written, "", ""])
        print(f"c5_inhouse: wrote the annotation template {tsv} — fill the `spoken` column "
              f"(katakana for ja.kana; target: 50 items × 10 classes), then run again.")
        return tsv
    with tsv.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if (r.get("spoken") or "").strip()]
    if not rows:
        print(f"c5_inhouse: {tsv} has no filled `spoken` values yet")
        return tsv
    task = lang.split(".", 1)[1]
    items = [{"id": r["id"], "group": r["semiotic_class"], "split": "test",
              "inputs": {"text": r["written"]}, "refs": {"text": r["spoken"]},
              "meta": {"task": task, "semiotic_class": r["semiotic_class"],
                       "lang": lang.split(".")[0], "source": "c5_inhouse",
                       "notes": r.get("notes", "")}} for r in rows]
    return write_pack(capability, lang, items, license_md(
        f"OpenDub in-house C5 items ({lang})", str(tsv), "in-house annotations",
        "Groups = semiotic class."))
