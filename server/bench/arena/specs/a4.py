"""A4 — transcription with word-level timing."""
from __future__ import annotations

from ..judges import Spec, cost, speed, text_errors

SPEC = Spec(
    id="A4",
    title="Transcription with word-level timing",
    judges={"text_errors@3": text_errors, "speed@1": speed, "cost@1": cost},
    primary={"*": "wer", "ja": "cer", "zh": "cer", "ko": "cer"},
    higher_is_better={"wer": False, "cer": False, "empty_output": False, "rtfx": True,
                      "cost_usd": False},
    threshold={"wer": 0.01, "cer": 0.01},
    secondary=["wer", "cer", "empty_output", "rtfx", "cost_usd"],
    packs=["fleurs"],
    io="""item.inputs: {audio: wav/flac path}; item.refs: {text}; item.lang forced.
payload: {text, segments: [{start, end, text, words: [{start, end, word}]}], language,
duration_s}""",
)
