"""CV3-Eval → D1 (cross-lingual and zero-shot cloning) and D2 (emotion cloning) packs.

Source: CV3-Eval (FunAudioLLM/CV3-Eval, the CosyVoice 3 benchmark) via the HF parquet mirror
``yuekai/CV3-Eval`` (revision pinned below). Each split is one parquet file
``data/<split>-00000-of-00001.parquet`` with columns ``id, prompt_text, prompt_audio
{bytes, path}, target_text``. Splits used:

- D1 en: ``cross_lingual_zeroshot_to_en`` (600; prompts in zh/ja/ko…) → split test,
  ``zero_shot_en`` (500) → split control
- D1 ja: ``cross_lingual_zeroshot_to_ja`` (600) → test, ``zero_shot_ja`` (500) → control
- D2 en: ``emotion_zeroshot_en`` (150; emotional prompts) → test
There is no Hindi in CV3-Eval. No human recording of the target text (no ASR floor).
Licence: not stated on the mirror; CV3-Eval prompts come from public corpora — eval-only use.
"""
from __future__ import annotations

import random
from pathlib import Path

from . import builder
from ._dvoice import finish, hf_file, pack_root, source_dir, write_audio_bytes

REPO, REVISION = "yuekai/CV3-Eval", "6ea9d3650fffcbed7c6279e6d1546d01ef1d2796"
SPLITS = {("D1", "en"): [("cross_lingual_zeroshot_to_en", "test"), ("zero_shot_en", "control")],
          ("D1", "ja"): [("cross_lingual_zeroshot_to_ja", "test"), ("zero_shot_ja", "control")],
          ("D2", "en"): [("emotion_zeroshot_en", "test")]}

LICENSE = """# CV3-Eval ({capability}, {lang})

Source: CV3-Eval (github.com/FunAudioLLM/CV3-Eval) via {repo}@{rev}, splits: {splits}.
Licence not stated on the mirror: evaluation-only, never redistributed. CosyVoice 3 was tuned on
this benchmark's domain: read CosyVoice rows here with that in mind.
"""


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pylist()


@builder("cv3_eval", capabilities=["D1", "D2"], langs=["en", "ja"], license="unstated (eval-only)")
def cv3_eval(lang: str, n: int | None = 200, seed: int = 0, capability: str = "D1") -> Path:
    """CV3-Eval splits for ``capability`` in ``lang``; ``n`` items per split (fixed seed)."""
    cap = capability.upper()
    if (cap, lang) not in SPLITS:
        raise ValueError(f"cv3_eval has no {cap} split for {lang!r} (have {sorted(SPLITS)})")
    src = source_dir("cv3_eval")
    root = pack_root(cap, lang)
    items = []
    for split_name, split in SPLITS[(cap, lang)]:
        pq_path = hf_file(REPO, f"data/{split_name}-00000-of-00001.parquet", src,
                          revision=REVISION)
        rows = read_parquet(pq_path)
        random.Random(seed).shuffle(rows)
        for r in rows[:n] if n else rows:
            audio = r.get("prompt_audio") or {}
            if not audio.get("bytes"):
                continue
            rid = str(r["id"]).replace("/", "_").replace(".", "_")
            rel = f"audio/{split_name}/{rid}.wav"
            write_audio_bytes(audio["bytes"], root / rel)
            meta = {"pair": f"x-{lang}" if split == "test" else f"{lang}-{lang}",
                    "cv3_split": split_name, "source": "cv3_eval"}
            if split == "control":
                meta["src_lang"] = lang
            items.append({"id": f"cv3-{split_name}-{rid}", "group": rid, "split": split,
                          "inputs": {"text": r["target_text"], "ref_audio": rel,
                                     "ref_text": r.get("prompt_text") or ""},
                          "refs": {"text": r["target_text"]}, "meta": meta})
    splits = ", ".join(s for s, _ in SPLITS[(cap, lang)])
    return finish(cap, lang, items, LICENSE.format(capability=cap, lang=lang, repo=REPO,
                                                   rev=REVISION, splits=splits))
