"""Filesystem layout for the arena. Every root can be overridden with an env var (tests do)."""
from __future__ import annotations

import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = SERVER_DIR.parent
CANDIDATES_DIR = SERVER_DIR / "bench" / "candidates"
WORKERS_DIR = Path(__file__).resolve().parent / "workers"


def data_dir() -> Path:
    return Path(os.environ.get("OPENDUB_ARENA_DATA", REPO_DIR / "data"))


def eval_dir() -> Path:
    """Eval packs: ``<eval>/<capability>/<lang>/manifest.jsonl`` plus downloaded sources."""
    return data_dir() / "eval"


def arena_dir() -> Path:
    """Arena state: ``arena.sqlite``, kept outputs, reports, audit marks."""
    return data_dir() / "arena"


def outputs_dir() -> Path:
    return arena_dir() / "outputs"


def db_path() -> Path:
    return arena_dir() / "arena.sqlite"
