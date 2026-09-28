"""T0 — test-only capability exercised with the model-free echo worker."""
from __future__ import annotations

from ..judges import Spec, text_errors

SPEC = Spec(
    id="T0",
    title="Test capability (echo worker)",
    judges={"text_errors@1": text_errors},
    primary={"*": "wer"},
    higher_is_better={"wer": False, "cer": False, "empty_output": False, "echo_len": True},
    threshold={"wer": 0.01},
    secondary=["cer"],
)
