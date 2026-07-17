"""Metric modules. Each exposes ``async score(ctx: MetricContext) -> list[Metric]``.

Ordered as the pipeline runs, with system metrics last. The harness calls each in turn and never
lets one module's failure abort the others.
"""
from __future__ import annotations

from . import (
    lipsync,
    mix,
    separation,
    system,
    transcribe,
    translate,
    tts,
)

# (stage label, module) in pipeline order.
MODULES = [
    ("separation", separation),
    ("transcribe", transcribe),
    ("translate", translate),
    ("tts", tts),
    ("mix", mix),
    ("lipsync", lipsync),
    ("system", system),
]
