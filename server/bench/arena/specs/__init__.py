"""Evaluation specs, one module per capability (``a4.py`` …, lower-case id), each exporting ``SPEC``.

A spec module owns everything capability-specific about judging: its function judges, which
model judges apply (from ``bench.arena.judgelib``), the primary metric per language, direction,
practical threshold, gates, and the item/payload contract (``io``). Modules are discovered
automatically, so adding a capability never touches shared files.
"""
from __future__ import annotations

import importlib
import pkgutil

from ..judges import Spec

SPECS: dict[str, Spec] = {}


def _load_all() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        spec = getattr(importlib.import_module(f"{__name__}.{mod.name}"), "SPEC", None)
        if isinstance(spec, Spec):
            if spec.id in SPECS:
                raise ValueError(f"duplicate spec {spec.id}")
            SPECS[spec.id] = spec


_load_all()
