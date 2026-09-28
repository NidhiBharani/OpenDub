"""Human-rated MT data → C4 packs (QE meta-evaluation). Downloads only.

- ``wmt23_qe_da``: WMT23 QE shared task, sentence-level DA, en→hi
  (github.com/WMT-QE-Task/wmt-qe-2023-data). The test DA gold labels are not in the repo, so the
  pack uses the released **dev** split (``task_1/en-hi/dev.enhi.df.short.tsv``: index, original,
  translation, scores, mean, z_scores, z_mean); human = ``z_mean``. One translation per source,
  so C4's ``acc_eq`` falls back to all pairs here. Licence not stated in the repo (unverified).
- ``mtme``: Google mt-metrics-eval (``mt-metrics-eval-v2.tgz``): WMT24 en→hi and en→ja with ESA
  segment scores for every system (plus WMT25 en→ja_JP, ``year="wmt25"``). Many systems per
  source, so ``acc_eq`` is within-source. group = document; ``inputs.reference`` holds refA for
  reference-based comparators (QE candidates ignore it).

Both write ``refs = {human, src_id}`` (see ``specs/c4.py``).
"""
from __future__ import annotations

import csv
import io
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
    split_direction,
    tar_members,
)

MTME_URL = "https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz"
MTME_LPS = {("wmt24", "en-hi"): "en-hi", ("wmt24", "en-ja"): "en-ja",
            ("wmt25", "en-ja"): "en-ja_JP"}
MTME_PROTOCOL = {"wmt24": "esa", "wmt25": "esa-human1"}


def _wmt23_tsv(split: str) -> str:
    tgz = github_tarball("WMT-QE-Task/wmt-qe-2023-data", ["main", "master"],
                         sources_dir("wmt23_qe"))
    name = f"{split}.enhi.df.short.tsv"
    found = tar_members(tgz, lambda m: m.endswith(name) and "en-hi" in m)
    if not found:
        raise FileNotFoundError(f"{name} not found in {tgz}")
    return read_text_bytes(next(iter(found.values())))


@builder("wmt23_qe_da", capabilities=["C4"], langs=["en-hi"], license="unverified (WMT23 QE)")
def wmt23_qe_da(lang: str = "en-hi", n: int | None = 1000, seed: int = 0, split: str = "dev",
                capability: str = "C4") -> Path:
    """WMT23 QE sentence-level DA en→hi (dev split; human = z_mean)."""
    src, tgt = split_direction(lang)
    text = _wmt23_tsv(split)
    reader = csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
    rows = [r for r in reader if r.get("original") and r.get("translation")
            and r.get("z_mean") not in (None, "")]
    rows = sample(rows, n, seed)
    items = [{"id": f"wmt23qe-{split}-{r['index']}", "group": str(r["index"]), "split": "test",
              "inputs": {"source": r["original"], "hypothesis": r["translation"],
                         "src_lang": src, "tgt_lang": tgt},
              "refs": {"human": float(r["z_mean"]), "src_id": int(r["index"])},
              "meta": {"src_lang": src, "tgt_lang": tgt, "source": "wmt23_qe",
                       "protocol": "da-z", "da_mean": float(r["mean"]) if r.get("mean") else None}}
             for r in rows]
    return write_pack(capability, lang, items, license_md(
        f"WMT23 QE DA {lang} ({split})", "https://github.com/WMT-QE-Task/wmt-qe-2023-data",
        "not stated in the repository (research data; treat as eval-only)",
        "human = z_mean (standardised DA). One MT output per source segment."))


def _mtme_files(year: str, lp: str) -> dict[str, bytes]:
    tgz = download(MTME_URL, sources_dir("mtme") / "mt-metrics-eval-v2.tgz", timeout=3600)
    prefix = f"mt-metrics-eval-v2/{year}/"

    def want(name: str) -> bool:
        rel = name.split(prefix, 1)[1] if prefix in name else ""
        return bool(rel) and (rel in (f"sources/{lp}.txt", f"documents/{lp}.docs")
                              or rel.startswith((f"references/{lp}.", f"system-outputs/{lp}/",
                                                 f"human-scores/{lp}.")))

    return {n.split(prefix, 1)[1]: b for n, b in tar_members(tgz, want).items()}


def parse_seg_scores(text: str) -> dict[str, list[float | None]]:
    """mt-metrics-eval ``*.seg.score``: "system<TAB>score" lines, in segment order per system."""
    out: dict[str, list[float | None]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        sysname, val = line.rsplit("\t", 1)
        out.setdefault(sysname, []).append(None if val.strip() in ("None", "") else float(val))
    return out


@builder("mtme", capabilities=["C4"], langs=["en-hi", "en-ja"], license="per WMT year (eval)")
def mtme(lang: str, n_sources: int | None = 200, seed: int = 0, year: str = "wmt24",
         capability: str = "C4") -> Path:
    """mt-metrics-eval human ESA scores (WMT24 en→hi/en→ja; WMT25 en→ja_JP)."""
    src, tgt = split_direction(lang)
    lp = MTME_LPS[(year, lang)]
    files = _mtme_files(year, lp)
    sources = read_text_bytes(files[f"sources/{lp}.txt"]).splitlines()
    docs_file = files.get(f"documents/{lp}.docs")
    docs = (read_text_bytes(docs_file).splitlines() if docs_file else [])
    protocol = MTME_PROTOCOL[year]
    score_key = next((k for k in files if k.startswith(f"human-scores/{lp}.{protocol}.seg")),
                     None) or next((k for k in files if k.startswith(f"human-scores/{lp}.")
                                    and k.endswith(".seg.score")), None)
    if score_key is None:
        raise FileNotFoundError(f"no segment-level human scores for {year} {lp}")
    scores = parse_seg_scores(read_text_bytes(files[score_key]))
    refs = sorted(k for k in files if k.startswith(f"references/{lp}."))
    ref_lines = read_text_bytes(files[refs[0]]).splitlines() if refs else []
    ref_name = refs[0].split(f"{lp}.", 1)[1].rsplit(".", 1)[0] if refs else None
    outputs = {k.rsplit("/", 1)[1].rsplit(".", 1)[0]: read_text_bytes(v).splitlines()
               for k, v in files.items() if k.startswith(f"system-outputs/{lp}/")}
    chosen = sample(list(range(len(sources))), n_sources, seed)
    items = []
    for i in sorted(chosen):
        doc = docs[i].split("\t")[-1] if i < len(docs) else str(i)
        for sysname, segs in sorted(scores.items()):
            if sysname == ref_name or sysname not in outputs or i >= len(segs):
                continue
            human, hyp = segs[i], outputs[sysname][i] if i < len(outputs[sysname]) else ""
            if human is None or not hyp.strip():
                continue
            inputs = {"source": sources[i], "hypothesis": hyp, "src_lang": src,
                      "tgt_lang": tgt}
            if i < len(ref_lines) and ref_lines[i].strip():
                inputs["reference"] = ref_lines[i]
            items.append({"id": f"{year}-{lp}-{i}-{sysname}", "group": doc, "split": "test",
                          "inputs": inputs, "refs": {"human": human, "src_id": i},
                          "meta": {"src_lang": src, "tgt_lang": tgt, "system": sysname,
                                   "source": f"mtme-{year}", "protocol": protocol}})
    return write_pack(capability, lang, items, license_md(
        f"mt-metrics-eval {year} {lp} ({protocol})",
        "https://github.com/google-research/mt-metrics-eval (mt-metrics-eval-v2.tgz)",
        "WMT shared-task data terms (research/evaluation)",
        f"Human {protocol} segment scores for every system on {len(chosen)} sampled source "
        f"segments; groups = documents; inputs.reference = {ref_name}."))
