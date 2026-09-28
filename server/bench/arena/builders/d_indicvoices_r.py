"""IndicVoices-R (Hindi) → D1 / D4 packs (CC BY 4.0; gated on HF: accept the terms, set HF_TOKEN).

Source: ``ai4bharat/indicvoices_r`` (revision pinned below), ``Hindi/test-0000{0,1}-of-00002.parquet``
(the test shards; ~600 MB). Columns (per the dataset viewer of the SPRINGLab mirror): ``text``,
``normalized``, ``verbatim``, ``speaker_id``, ``gender``, ``duration``, ``audio {bytes, path}``,
plus scenario/age/district metadata.

Items pair a target utterance (its text; its own recording = ``refs.human_audio``, the ASR
floor) with a *different* utterance of the SAME speaker as the reference voice — same-speaker
cloning, so SIM has a true same-speaker ceiling (cross-lingual ja→hi lives in ``fleurs_voice``).
Group = speaker_id.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from . import builder
from ._dvoice import finish, hf_file, pack_root, source_dir, write_audio_bytes

REPO, REVISION = "ai4bharat/indicvoices_r", "5f4495c91d500742a58d1be2ab07d77f73c0acf8"
SHARDS = ["Hindi/test-00000-of-00002.parquet", "Hindi/test-00001-of-00002.parquet"]

LICENSE = """# IndicVoices-R Hindi test ({capability})

Source: https://huggingface.co/datasets/{repo} (revision {rev}; gated, CC BY 4.0), Hindi test
shards. Built by bench/arena/builders/d_indicvoices_r.py: target = one utterance (its recording is
refs.human_audio), reference voice = another utterance of the same speaker. Group = speaker_id.
Attribution: AI4Bharat, IndicVoices-R (NeurIPS 2024 Datasets & Benchmarks).
"""


def _rows(shards: list[Path], columns: list[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    out: list[dict[str, Any]] = []
    for s in shards:
        t = pq.read_table(str(s))
        keep = [c for c in columns if c in t.column_names]
        out += t.select(keep).to_pylist()
    return out


def _gender(v: Any) -> str:
    s = str(v).lower()
    return "female" if s.startswith(("f", "0")) else "male" if s.startswith(("m", "1")) else s


@builder("indicvoices_r", capabilities=["D1", "D4"], langs=["hi"], license="CC BY 4.0")
def indicvoices_r(lang: str = "hi", n: int | None = 200, seed: int = 0, capability: str = "D1",
                  min_ref_s: float = 3.0, max_ref_s: float = 12.0, max_target_s: float = 15.0,
                  shards: list[str] | None = None) -> Path:
    """IndicVoices-R Hindi: ``n`` same-speaker cloning items (fixed seed)."""
    if lang != "hi":
        raise ValueError("this builder covers Hindi (other IndicVoices-R languages: add shards)")
    src = source_dir("indicvoices_r")
    paths_ = [hf_file(REPO, s, src, revision=REVISION, gated=True) for s in shards or SHARDS]
    rows = _rows(paths_, ["text", "normalized", "speaker_id", "gender", "duration", "audio"])
    by_spk: dict[str, list[dict[str, Any]]] = {}
    for i, r in enumerate(rows):
        r["_i"] = i
        by_spk.setdefault(str(r.get("speaker_id")), []).append(r)
    rng = random.Random(seed)
    cands = []
    for spk, rs in by_spk.items():
        refs = [r for r in rs if min_ref_s <= float(r.get("duration") or 0) <= max_ref_s]
        targets = [r for r in rs if 0.5 < float(r.get("duration") or 0) <= max_target_s]
        for t in targets:
            others = [r for r in refs if r["_i"] != t["_i"]]
            if others:
                cands.append((spk, t, rng.choice(others)))
    rng.shuffle(cands)
    cands = cands[:n] if n else cands
    root = pack_root(capability, lang)
    items = []
    for spk, t, ref in cands:
        t_rel, r_rel = f"audio/{spk}/{t['_i']}.wav", f"audio/{spk}/{ref['_i']}.wav"
        write_audio_bytes(t["audio"]["bytes"], root / t_rel)
        write_audio_bytes(ref["audio"]["bytes"], root / r_rel)
        items.append({"id": f"ivr-hi-{spk}-{t['_i']}", "group": spk, "split": "test",
                      "inputs": {"text": t["text"], "ref_audio": r_rel,
                                 "ref_text": ref.get("text") or ""},
                      "refs": {"text": t["text"], "human_audio": t_rel},
                      "meta": {"src_lang": "hi", "pair": "hi-hi", "gender": _gender(t["gender"]),
                               "duration_s": float(t.get("duration") or 0),
                               "source": "indicvoices_r"}})
    return finish(capability, lang, items,
                  LICENSE.format(capability=capability, repo=REPO, rev=REVISION))
