"""A7 packs: speaker-verification trials and voice-bank matching.

- ``a7-voxceleb1``: VoxCeleb1-O (cleaned, ``veri_test2.txt``: 37,611 trials; ``-E``/``-H`` with
  ``trials="E"|"H"``). The trial lists download from mm.kaist.ac.kr; the audio needs the official
  request form (cn01.mmai.io/keyreq/voxceleb), so ``vox1_test_wav.zip`` (or an extracted
  ``wav/`` tree, or the dev set for E/H) must be put in ``data/eval/_sources/voxceleb1/`` by the
  user. ``n`` samples trials (stratified, so the target share is kept); ``max_s`` crops every
  clip to its first ``max_s`` seconds (the plan's 2–4 s "separated segment" condition).
- ``a7-cv-speakers``: same-language trials + voice-bank items for any Common Voice locale (CC0;
  ``client_id`` = speaker) from an extracted CV release in ``data/eval/_sources/commonvoice/
  <lang>/`` (``validated.tsv`` + ``clips/``; Mozilla Data Collective download, login needed).
  Non-target trials are same-gender where gender is known. Bank items: ``bank_size`` enrolment
  speakers, the test clip from one of them (``refs.match`` = its index) or, for
  ``impostor_rate`` of items, from nobody in the bank (``refs.match = -1``).

Enrolment clips go in ``inputs.bank_00 …`` (string keys, so the pack loader resolves them).
Groups: the test speaker, so the bootstrap resamples speakers, not trials.
"""
from __future__ import annotations

import csv
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from .. import paths
from . import builder
from ._aphase import fetch, need, read_mono, source_dir, write, write_merged

TRIALS = {"O": "veri_test2.txt", "E": "list_test_all2.txt", "H": "list_test_hard2.txt"}
TRIAL_URL = "https://mm.kaist.ac.kr/datasets/voxceleb/meta/{}"


def _materialise(src: Path, out: Path, rel: str, max_s: float | None) -> str:
    """Copy (or crop) a clip into the pack; returns the pack-relative path."""
    dst = out / "audio" / rel
    if max_s:
        dst = dst.with_suffix(".wav")
    if not dst.exists():
        if max_s:
            write(dst, read_mono(src, 16000)[: int(max_s * 16000)], 16000)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    return str(dst.relative_to(out))


@builder("a7-voxceleb1", capabilities=["A7"], langs=["en"], license="CC BY 4.0")
def a7_voxceleb1(lang: str = "en", n: int | None = 6000, seed: int = 0, trials: str = "O",
                 max_s: float | None = None) -> Path:
    """A7 pack: VoxCeleb1 verification trials (user-provided audio)."""
    src = source_dir("voxceleb1")
    tl = fetch(TRIAL_URL.format(TRIALS[trials]), src / TRIALS[trials])
    wav = src / "wav"
    if not wav.exists():
        z = need(src / "vox1_test_wav.zip", "VoxCeleb1 audio",
                 "Request access at https://cn01.mmai.io/keyreq/voxceleb and put "
                 "vox1_test_wav.zip (or an extracted wav/ tree) there.")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(src)
    rows_t = [ln.split() for ln in tl.read_text().splitlines() if ln.strip()]
    pyr = random.Random(seed)
    if n and n < len(rows_t):
        tgt = [r for r in rows_t if r[0] == "1"]
        non = [r for r in rows_t if r[0] != "1"]
        k = round(n * len(tgt) / len(rows_t))
        rows_t = pyr.sample(tgt, k) + pyr.sample(non, n - k)
    out = paths.eval_dir() / "A7" / lang
    rows = []
    for label, enrol, test in rows_t:
        a, b = wav / enrol, wav / test
        if not a.exists() or not b.exists():
            continue
        rows.append({"id": f"vox1{trials}-{enrol}-{test}".replace("/", "_"),
                     "group": test.split("/")[0], "split": "test",
                     "inputs": {"audio": _materialise(b, out, test, max_s),
                                "bank_00": _materialise(a, out, enrol, max_s)},
                     "refs": {"label": int(label)},
                     "meta": {"task": "trial", "trials": f"Vox1-{trials}", "max_s": max_s}})
    lic = (f"# VoxCeleb1-{trials} trials\n\nTrial list: {TRIAL_URL.format(TRIALS[trials])} "
           "(cleaned). Audio: VoxCeleb1 (CC BY 4.0; video copyright remains with the owners), "
           f"user-downloaded. {len(rows)} trials, seed {seed}"
           + (f", clips cropped to {max_s} s" if max_s else "") + ". Groups: test speaker.\n")
    return write_merged("A7", lang, rows, lic, f"voxceleb1-{trials}")


@builder("a7-cv-speakers", capabilities=["A7"], langs=["en", "hi", "ja", "ko", "ta"],
         license="CC0")
def a7_cv_speakers(lang: str, n: int | None = 3000, seed: int = 0, bank_items: int = 300,
                   bank_size: int = 10, impostor_rate: float = 0.2,
                   max_s: float | None = None, source: str | None = None) -> Path:
    """A7 pack: Common Voice speaker trials + voice-bank matching items."""
    root = need(Path(source) if source else source_dir("commonvoice") / lang,
                f"Common Voice ({lang})", "Download the locale from Mozilla Data Collective and "
                "extract validated.tsv + clips/ there.")
    by_spk: dict[str, list[Path]] = defaultdict(list)
    gender: dict[str, str] = {}
    with (root / "validated.tsv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            p = root / "clips" / r["path"]
            if p.exists():
                by_spk[r["client_id"]].append(p)
                if r.get("gender"):
                    gender[r["client_id"]] = r["gender"]
    spks = sorted(s for s, v in by_spk.items() if len(v) >= 2)
    if len(spks) < bank_size + 2:
        raise ValueError(f"only {len(spks)} speakers with ≥ 2 clips in {root}")
    pyr = random.Random(seed)
    out = paths.eval_dir() / "A7" / lang
    rows = []

    def rel(p: Path) -> str:
        return _materialise(p, out, f"cv/{p.name}", max_s)

    for k in range(n or 0):
        s = pyr.choice(spks)
        a, b = pyr.sample(by_spk[s], 2)
        if k % 2:
            pool = [x for x in spks if x != s and gender.get(x) == gender.get(s)] or \
                [x for x in spks if x != s]
            b = pyr.choice(by_spk[pyr.choice(pool)])
        rows.append({"id": f"cv-{lang}-t{k:05d}", "group": s[:16], "split": "test",
                     "inputs": {"audio": rel(b), "bank_00": rel(a)},
                     "refs": {"label": int(k % 2 == 0)}, "meta": {"task": "trial"}})
    for k in range(bank_items):
        bank = pyr.sample(spks, bank_size)
        impostor = pyr.random() < impostor_rate
        if impostor:
            s = pyr.choice([x for x in spks if x not in bank])
            test, match = pyr.choice(by_spk[s]), -1
        else:
            match = pyr.randrange(bank_size)
            s = bank[match]
        enrol = [pyr.choice(by_spk[x]) for x in bank]
        if not impostor:
            others = [p for p in by_spk[s] if p != enrol[match]]
            test = pyr.choice(others)
        inputs = {"audio": rel(test)}
        inputs.update({f"bank_{j:02d}": rel(p) for j, p in enumerate(enrol)})
        rows.append({"id": f"cv-{lang}-b{k:04d}", "group": s[:16], "split": "test",
                     "inputs": inputs, "refs": {"match": match},
                     "meta": {"task": "bank", "bank_size": bank_size}})
    lic = (f"# Common Voice speaker trials ({lang})\n\nSource: Mozilla Common Voice ({root}), CC0. "
           f"Speakers = client_id; seed {seed}. Groups: test speaker.\n")
    return write_merged("A7", lang, rows, lic, "commonvoice")
