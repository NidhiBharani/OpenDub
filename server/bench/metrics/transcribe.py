"""Transcribe-stage metrics.

Ground truth (when a ref_transcript.jsonl exists): WER / CER via jiwer over the concatenated
transcript. Reference-free: speech-coverage (fraction of media duration inside segments) and a
segments-per-minute sanity value that flags "0 usable segments" or runaway over-segmentation.
"""
from __future__ import annotations

from .base import Metric, MetricContext, load_jsonl, missing


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


async def score(ctx: MetricContext) -> list[Metric]:
    segs = ctx.project.segments
    duration = ctx.project.media.duration if ctx.project.media else 0.0
    out: list[Metric] = []

    # --- reference-free ---
    covered = sum(s.duration for s in segs)
    out.append(
        Metric(
            "transcribe.speech_coverage",
            round(covered / duration, 3) if duration else None, "ratio", "free",
            note=f"{covered:.1f}s of {duration:.1f}s inside segments; near-0 means ASR/VAD "
                 "dropped everything (the sung-vocals VAD bug)",
        )
    )
    out.append(
        Metric(
            "transcribe.segments_per_min",
            round(len(segs) / (duration / 60), 1) if duration else None, "1/min", "free",
            note=f"{len(segs)} segments; 0 or absurdly high both signal a broken transcribe",
        )
    )

    # --- ground truth (jiwer) ---
    if not ctx.case.has("ref_transcript"):
        out.append(missing("transcribe.wer", "no ref_transcript.jsonl", provenance="gt", unit="ratio"))
        return out
    try:
        import jiwer
    except ImportError:
        out.append(missing("transcribe.wer", "jiwer not installed (bench extra)",
                            provenance="gt", unit="ratio"))
        return out

    ref = " ".join(_normalize(r.get("text", "")) for r in load_jsonl(ctx.case.ref_transcript))
    hyp = " ".join(_normalize(s.source_text) for s in sorted(segs, key=lambda s: s.start))
    if not ref.strip():
        out.append(missing("transcribe.wer", "ref_transcript.jsonl empty", provenance="gt", unit="ratio"))
        return out
    out.append(
        Metric("transcribe.wer", round(jiwer.wer(ref, hyp), 3), "ratio", "gt",
               higher_is_better=False, note="word error rate vs ref_transcript")
    )
    out.append(
        Metric("transcribe.cer", round(jiwer.cer(ref, hyp), 3), "ratio", "gt",
               higher_is_better=False, note="character error rate vs ref_transcript")
    )
    return out
