"""Candidate registry: which models compete in each capability, and how to run them.

``bench/candidates/<ID>.yaml``::

    capability: A4
    candidates:
      - id: fw-large-v3-turbo          # unique within the capability
        worker: asr_faster_whisper     # bench/arena/workers/<worker>.py
        env: server                    # bench/envs/<env>.yaml (which python runs the worker)
        kind: local                    # local | api | method (composition of other models)
        params: {model: large-v3-turbo, compute_type: float16}
        languages: "*"                 # or the codes the vendor declares
        license: MIT
        ship_ok: true                  # false = non-commercial weights (still allowed as default)
        requires: {vram_gb: 3}         # hardware.Requires: gpu, vram_gb, ram_gb, disk_gb, arch,
                                       #   min_compute_capability, api_keys, tools
        baseline: true                 # the current OpenDub default for this capability
        enabled: true                  # false = registered but never run (e.g. gated weights)
        sources: [compute-tiers:A4:spark]   # which research artifact row it came from
        url: https://huggingface.co/…  # where to get it: weights page, repo, or API docs
        params_b: 0.8                  # size in billions of parameters, if known
        verified: {license: true, languages: true, vram: false, runs: false}

Environments: ``bench/envs/<name>.yaml`` (one file per env, with setup recipes per architecture;
see ``envs.py``). The legacy ``bench/candidates/_envs.yaml`` map is still read.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import paths
from .hardware import Hardware, Requires, check


@dataclass
class Candidate:
    id: str
    capability: str
    worker: str
    env: str = "server"
    kind: str = "local"
    params: dict[str, Any] = field(default_factory=dict)
    languages: list[str] | str = "*"
    license: str = "unknown"
    ship_ok: bool | None = None
    requires: Requires = field(default_factory=Requires)
    vram_gb: float = 0.0               # legacy shorthand for requires.vram_gb
    profiles: list[str] = field(default_factory=list)  # informational: where research placed it
    baseline: bool = False
    enabled: bool = True
    sources: list[str] = field(default_factory=list)
    verified: dict[str, bool] = field(default_factory=dict)
    notes: str = ""
    url: str = ""                      # where to get it: weights page, repo, or API docs
    params_b: float | None = None      # model size in billions of parameters, if known

    def __post_init__(self) -> None:
        if not isinstance(self.requires, Requires):
            self.requires = Requires.parse(self.requires, self.vram_gb)
        elif self.vram_gb and not self.requires.vram_gb:
            self.requires = Requires.parse({}, self.vram_gb)
        self.vram_gb = self.vram_gb or self.requires.vram_gb

    def supports(self, lang: str) -> bool:
        return self.languages == "*" or lang in self.languages

    def blockers(self, hw: Hardware | None = None) -> list[str]:
        """Why this candidate cannot run here (disabled, hardware, missing env); [] = runnable."""
        why = [] if self.enabled else ["disabled in registry"]
        why += check(self.requires, hw)
        try:
            env_python(self.env)
        except (KeyError, FileNotFoundError) as exc:
            why.append(f"env {self.env!r} not set up ({exc.__class__.__name__})")
        if not (paths.WORKERS_DIR / f"{self.worker}.py").exists():
            why.append(f"worker {self.worker}.py missing")
        return why

    @property
    def key(self) -> str:
        """Stable identity of *what is generated*: worker + params (not licence/notes)."""
        blob = json.dumps({"worker": self.worker, "params": self.params}, sort_keys=True)
        return f"{self.id}@{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


def load_candidates(capability: str, root: Path | None = None) -> list[Candidate]:
    path = (root or paths.CANDIDATES_DIR) / f"{capability}.yaml"
    if not path.exists():
        return []
    doc = yaml.safe_load(path.read_text()) or {}
    cap = doc.get("capability", capability)
    out: list[Candidate] = []
    seen: set[str] = set()
    for raw in doc.get("candidates", []):
        cand = Candidate(capability=cap, **raw)
        if cand.id in seen:
            raise ValueError(f"{path}: duplicate candidate id {cand.id!r}")
        seen.add(cand.id)
        out.append(cand)
    return out


def capabilities(root: Path | None = None) -> list[str]:
    root = root or paths.CANDIDATES_DIR
    return sorted(p.stem for p in root.glob("*.yaml") if not p.stem.startswith("_"))


def select(candidates: list[Candidate], ids: list[str] | None) -> list[Candidate]:
    if not ids:
        return candidates
    by_id = {c.id: c for c in candidates}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown candidates: {', '.join(missing)} (known: {', '.join(by_id)})")
    return [by_id[i] for i in ids]


def env_python(env: str, root: Path | None = None) -> str:
    """Interpreter for a worker env. Isolated envs keep candidate deps out of the server venv."""
    from .envs import load_env

    spec = load_env(env, root)
    python = spec.python
    if python == "auto":
        return sys.executable
    resolved = Path(python).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"worker env {env!r}: interpreter {resolved} does not exist "
                                f"(run `python -m bench arena env setup {env}`)")
    return str(resolved)
