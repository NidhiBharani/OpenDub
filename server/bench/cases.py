"""Benchmark case registry.

A case is a source video under ``data/benchmarks/<name>/source.mp4`` plus any ground-truth files
that happen to exist next to it (all optional):

    data/benchmarks/<name>/
        source.mp4
        ref_transcript.jsonl     # {"start":..,"end":..,"text":".."} per line   (transcribe GT)
        ref_translation.jsonl    # {"text":".."} per line, aligned to ref_transcript (translate GT)
        ref_vocals.wav           # isolated vocal stem                            (separation GT)

The registry itself is ``data/benchmarks/cases.yaml``:

    cases:
      - name: moshi
        source_lang: ja
        target_lang: en
"""
from __future__ import annotations

from pathlib import Path

import yaml

from app import config

from .metrics.base import BenchCase

BENCH_DIR = config.DATA_DIR / "benchmarks"
REGISTRY = BENCH_DIR / "cases.yaml"


def _case_from_dir(name: str, source_lang: str, target_lang: str) -> BenchCase:
    d = BENCH_DIR / name

    def opt(fname: str) -> Path | None:
        p = d / fname
        return p if p.exists() else None

    return BenchCase(
        name=name,
        source=d / "source.mp4",
        source_lang=source_lang,
        target_lang=target_lang,
        ref_transcript=opt("ref_transcript.jsonl"),
        ref_translation=opt("ref_translation.jsonl"),
        ref_vocals=opt("ref_vocals.wav"),
    )


def load_cases(names: list[str] | None = None) -> list[BenchCase]:
    """Load cases from the registry, optionally filtered to ``names``."""
    if not REGISTRY.exists():
        raise FileNotFoundError(
            f"no benchmark registry at {REGISTRY}; create it with a 'cases:' list"
        )
    spec = yaml.safe_load(REGISTRY.read_text()) or {}
    cases = []
    for entry in spec.get("cases", []):
        name = entry["name"]
        if names and name not in names:
            continue
        case = _case_from_dir(name, entry.get("source_lang", "auto"),
                              entry.get("target_lang", "en"))
        if not case.source.exists():
            raise FileNotFoundError(f"case '{name}': missing {case.source}")
        cases.append(case)
    if names:
        found = {c.name for c in cases}
        missing = set(names) - found
        if missing:
            raise KeyError(f"cases not found in registry: {sorted(missing)}")
    return cases


# Re-export so callers can `from bench.cases import BenchCase`.
__all__ = ["BenchCase", "load_cases", "BENCH_DIR", "REGISTRY"]
