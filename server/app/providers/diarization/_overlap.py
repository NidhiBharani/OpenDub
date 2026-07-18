"""Map diarization turns onto ASR segments.

A diarizer produces speaker turns `(start, end, speaker)` on its own timeline; the pipeline needs
exactly one speaker label per ASR segment, in order. Assign each segment the speaker whose turn
overlaps it most, falling back to the previous segment's label when nothing overlaps (a short
segment inside a gap keeps the surrounding speaker rather than inventing a new one).
"""
from __future__ import annotations

from ...models import ASRSegment


def assign_labels_by_overlap(
    segments: list[ASRSegment], turns: list[tuple[float, float, str]]
) -> list[str]:
    labels: list[str] = []
    prev_label = "S0"
    for seg in segments:
        best_label: str | None = None
        best_overlap = 0.0
        for t_start, t_end, spk in turns:
            overlap = min(seg.end, t_end) - max(seg.start, t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = spk
        label = best_label if best_label is not None else prev_label
        labels.append(label)
        prev_label = label
    return labels
