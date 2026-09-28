"""Seed-TTS-eval test-en → D1 (same-language zero-shot cloning, the field's standard en board).

Source: the Seed-TTS-eval test set (github.com/BytedanceSpeech/seed-tts-eval, originally a Google
Drive zip) via the HF mirror ``hhqx/seedtts_testset`` (``data/en.tgz`` + ``data/en_meta.lst``,
revision pinned below). ``en_meta.lst`` lines: ``utt|prompt_text|prompt_wav|target_text``;
prompts and targets are Common Voice English (CC0). When the archive carries a ground-truth
recording for ``utt`` (``en/wavs/<utt>.wav``) it becomes ``refs.human_audio`` (ASR floor).
Items are same-language (src_lang en); split ``test``.
"""
from __future__ import annotations

import random
import tarfile
from pathlib import Path

from . import builder
from ._dvoice import finish, hf_file, pack_root, source_dir

REPO, REVISION = "hhqx/seedtts_testset", "9537716e9ed3f144f40d7e29075e2bdfe894ade3"

LICENSE = """# Seed-TTS-eval test-en ({capability})

Source: Seed-TTS-eval (BytedanceSpeech/seed-tts-eval) via the HF mirror {repo}@{rev}
(data/en.tgz, data/en_meta.lst). Prompts/targets from Mozilla Common Voice English (CC0-1.0).
Same-language zero-shot cloning; group = prompt speaker clip. Widely used in TTS papers: models
may have tuned on it — compare within this pack only.
"""


def parse_meta(text: str) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 4:
            rows.append({"utt": parts[0], "prompt_text": parts[1], "prompt_wav": parts[2],
                         "text": parts[3]})
    return rows


@builder("seedtts_eval", capabilities=["D1"], langs=["en"], license="CC0-1.0 (Common Voice)")
def seedtts_eval(lang: str = "en", n: int | None = 300, seed: int = 0,
                 capability: str = "D1") -> Path:
    """Seed-TTS-eval en: ``n`` items (fixed seed) of prompt clip + transcript → target text."""
    if lang != "en":
        raise ValueError("seedtts_eval has English only here (the zh set is out of scope)")
    src = source_dir("seedtts_eval")
    meta = hf_file(REPO, "data/en_meta.lst", src, revision=REVISION)
    tgz = hf_file(REPO, "data/en.tgz", src, revision=REVISION)
    rows = parse_meta(meta.read_text())
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if n else rows
    root = pack_root(capability, lang)
    out_dir = root / "audio"
    wanted = {r["prompt_wav"] for r in rows} | {f"en/wavs/{r['utt']}.wav" for r in rows}
    with tarfile.open(tgz) as tf:
        for m in tf:
            name = m.name.lstrip("./")
            if m.isfile() and name in wanted and not (out_dir / name).exists():
                (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as fsrc:  # type: ignore[union-attr]
                    (out_dir / name).write_bytes(fsrc.read())
    items = []
    for r in rows:
        if not (out_dir / r["prompt_wav"]).exists():
            continue
        refs = {"text": r["text"]}
        if (out_dir / f"en/wavs/{r['utt']}.wav").exists():
            refs["human_audio"] = f"audio/en/wavs/{r['utt']}.wav"
        items.append({"id": f"seedtts-en-{r['utt']}", "group": Path(r["prompt_wav"]).stem,
                      "split": "test",
                      "inputs": {"text": r["text"], "ref_audio": f"audio/{r['prompt_wav']}",
                                 "ref_text": r["prompt_text"]},
                      "refs": refs,
                      "meta": {"src_lang": "en", "pair": "en-en", "source": "seedtts_eval"}})
    return finish(capability, lang, items,
                  LICENSE.format(capability=capability, repo=REPO, rev=REVISION))
