"""Shared metric interface for the benchmark harness.

Every stage metric module exposes a single ``score(ctx) -> list[Metric]`` coroutine. A metric is a
named number (or None when it could not be computed) plus provenance: whether it needed ground
truth, and a human note. The harness never crashes on a missing optional dependency or missing
ground-truth file — the metric is returned with ``value=None`` and an explanatory ``note`` instead.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.models import Project


@dataclass
class Metric:
    """One measured quantity. ``value=None`` means "could not compute" (see ``note``)."""

    name: str
    value: float | int | None
    unit: str = ""
    # "gt" = needs ground-truth reference; "free" = reference-free proxy.
    provenance: str = "free"
    higher_is_better: bool | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchCase:
    """One benchmark input: a source video plus whatever ground truth exists for it."""

    name: str
    source: Path  # absolute path to source.mp4
    source_lang: str = "auto"
    target_lang: str = "en"
    # Optional ground-truth files (absolute paths, present only when the user created them).
    ref_transcript: Path | None = None  # jsonl: {start,end,text} per line
    ref_translation: Path | None = None  # jsonl: {text} per line, aligned to ref_transcript
    ref_vocals: Path | None = None  # isolated vocal stem for separation SI-SDR

    def has(self, attr: str) -> bool:
        p = getattr(self, attr, None)
        return isinstance(p, Path) and p.exists()


@dataclass
class MetricContext:
    """Everything a metric module needs: the finished project, its on-disk dir, and the case."""

    project: Project
    project_dir: Path  # absolute data/projects/<id>/
    case: BenchCase
    # Per-stage timing + peak VRAM captured by the runner, keyed by StageKey.
    stage_timing: dict[str, dict[str, float]] = field(default_factory=dict)

    def path(self, rel: str) -> Path:
        """Absolute path for a project-relative media path."""
        return self.project_dir / rel


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a .jsonl ground-truth file into a list of dicts (blank lines skipped)."""
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def missing(name: str, reason: str, *, provenance: str = "free", unit: str = "") -> Metric:
    """A metric that could not be computed, with the reason recorded."""
    return Metric(name=name, value=None, unit=unit, provenance=provenance, note=reason)
