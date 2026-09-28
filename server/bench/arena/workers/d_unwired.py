"""Placeholder worker for D-phase ledger rows that cannot run yet (no released weights, no public
API, paper-only method, or an open model whose inference path is not wired). Their candidates are
``enabled: false`` with a ``notes:`` line saying why; this worker only exists so every registry row
names a real file, and it refuses to run.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    raise RuntimeError(f"candidate not wired: {params.get('why', 'see bench/candidates notes')}")


def run(state: dict, item: dict, out: Path) -> dict:  # pragma: no cover - load always raises
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    serve(load, run)
