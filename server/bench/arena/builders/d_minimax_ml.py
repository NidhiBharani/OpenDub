"""MiniMax multilingual TTS test set → D1 / D4 packs (24 languages incl. en, hi, ja).

Source: ``MiniMaxAI/TTS-Multilingual-Test-Set`` (CC BY-SA 4.0; revision pinned below). Layout
(dataset card): ``text/<language>.txt`` lines ``<speaker>|<text>`` (100 sentences per language),
``speaker/<language>/<language>_{female,male}/<clip>.mp3`` (two Common Voice speakers per
language) and ``speaker/<language>/prompt_text.txt`` with the clips' transcripts
(``<speaker>|<text>`` or ``<clip>|<text>``, parsed defensively).

Same-language cloning by default (the published protocol). ``cross_from`` (e.g. "japanese")
re-voices every sentence with that language's speakers instead → a cross-lingual split
``test-x`` next to the same-language ``test``. No human recording of the targets.
"""
from __future__ import annotations

from pathlib import Path

from . import builder
from ._dvoice import cut, finish, pack_root, source_dir

REPO, REVISION = "MiniMaxAI/TTS-Multilingual-Test-Set", "072144006475d33f8263481ab465c00747fe2c74"
NAMES = {"en": "english", "hi": "hindi", "ja": "japanese", "zh": "chinese", "ko": "korean",
         "es": "spanish", "fr": "french", "de": "german", "pt": "portuguese", "it": "italian",
         "ru": "russian", "ar": "arabic"}

LICENSE = """# MiniMax multilingual TTS test set ({capability}, {lang})

Source: https://huggingface.co/datasets/{repo} (revision {rev}), licence CC BY-SA 4.0 (dataset
card). Speaker prompts are Mozilla Common Voice clips. Built by
bench/arena/builders/d_minimax_ml.py. Group = prompt speaker. No human recordings of the targets.
"""


def _snapshot(language: str) -> Path:
    from huggingface_hub import snapshot_download

    local = source_dir("minimax_ml")
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=str(local),
                      allow_patterns=[f"text/{language}.txt", f"speaker/{language}/**"])
    return local


def parse_prompts(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "|" in line:
                k, v = line.split("|", 1)
                out[Path(k.strip()).stem] = v.strip()
    return out


def speakers(local: Path, language: str) -> dict[str, tuple[Path, str]]:
    """speaker name (e.g. hindi_female) → (clip path, transcript)."""
    base = local / "speaker" / language
    prompts = parse_prompts(base / "prompt_text.txt")
    out = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
        clips = sorted(f for f in d.iterdir() if f.suffix.lower() in (".mp3", ".wav", ".flac"))
        if clips:
            text = prompts.get(d.name) or prompts.get(clips[0].stem) or ""
            out[d.name] = (clips[0], text)
    return out


@builder("minimax_ml", capabilities=["D1", "D4"], langs=sorted(NAMES), license="CC BY-SA 4.0")
def minimax_ml(lang: str, n: int | None = None, capability: str = "D1",
               cross_from: str | None = None) -> Path:
    """MiniMax set for ``lang``: every sentence with its listed speaker (+ ``cross_from``)."""
    language = NAMES[lang]
    local = _snapshot(language) if not (source_dir("minimax_ml") / "text" /
                                        f"{language}.txt").exists() else source_dir("minimax_ml")
    root = pack_root(capability, lang)
    spk = speakers(local, language)
    lines = [ln.split("|", 1) for ln in (local / "text" / f"{language}.txt").read_text()
             .splitlines() if "|" in ln]
    lines = lines[:n] if n else lines
    items = []
    variants = [("test", spk, lang)]
    if cross_from:
        xl = next(k for k, v in NAMES.items() if v == cross_from)
        local_x = _snapshot(cross_from)
        variants.append(("test-x", speakers(local_x, cross_from), xl))
    for split, pool, src_lang in variants:
        names = sorted(pool)
        for k, (who, text) in enumerate(lines):
            name = who.strip()
            if split == "test-x":  # same gender as the listed speaker, from the other language
                gender = name.rsplit("_", 1)[-1]
                name = next((s for s in names if s.endswith(gender)), names[k % len(names)])
            if name not in pool:
                continue
            clip, ptext = pool[name]
            rel = f"audio/{name}.wav"
            cut(clip, root / rel)
            items.append({"id": f"minimax-{split}-{lang}-{k:03d}", "group": name, "split": split,
                          "inputs": {"text": text.strip(), "ref_audio": rel, "ref_text": ptext},
                          "refs": {"text": text.strip()},
                          "meta": {"src_lang": src_lang, "pair": f"{src_lang}-{lang}",
                                   "gender": name.rsplit("_", 1)[-1], "source": "minimax_ml"}})
    return finish(capability, lang, items,
                  LICENSE.format(capability=capability, lang=lang, repo=REPO, rev=REVISION))
