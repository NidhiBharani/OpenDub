"""Eval-pack builders, one module per data source. Each registers itself with :func:`builder`.

A builder turns downloaded public data (or local sources) into ``data/eval/<ID>/<lang>`` packs::

    @builder("fleurs", capabilities=["A4"], langs=["en", "hi", "ja"], license="CC BY 4.0")
    def fleurs(lang: str, n: int | None = 300, **opts) -> list[Path]: ...

Sources download into ``data/eval/_sources/<name>`` and are never committed. Every pack writes a
``LICENSE.md`` naming the source and its terms. Builders must not run models; a builder that
needs model output (e.g. transcripts of an official dub) takes it as a separate, opt-in step.
"""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Builder:
    name: str
    fn: Callable[..., list[Path] | Path]
    capabilities: list[str] = field(default_factory=list)
    langs: list[str] = field(default_factory=list)
    license: str = ""
    description: str = ""


BUILDERS: dict[str, Builder] = {}


def builder(name: str, *, capabilities: list[str], langs: list[str] | None = None,
            license: str = "", description: str = ""):
    def wrap(fn):
        if name in BUILDERS:
            raise ValueError(f"duplicate builder {name!r}")
        BUILDERS[name] = Builder(name, fn, capabilities, langs or [], license,
                                 description or (fn.__doc__ or "").strip().split("\n")[0])
        return fn
    return wrap


def _load_all() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


_load_all()
