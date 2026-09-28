"""ASR transcript and timing diagnostics."""
from __future__ import annotations

from app.pipeline.quality import coverage_diagnostics, text_error_rates

from .base import Metric, MetricContext, load_jsonl, missing


async def score(ctx: MetricContext) -> list[Metric]:
    segs = ctx.project.segments
    duration = ctx.project.media.duration if ctx.project.media else 0.0
    diag = coverage_diagnostics([(s.start, s.end) for s in segs], duration)
    out = [
        Metric('transcribe.speech_coverage', round(diag['coverage_ratio'], 3)
               if diag['coverage_ratio'] is not None else None, 'ratio', 'free',
               note='union of segment intervals divided by media duration; no VAD truth implied'),
        Metric('transcribe.segments_per_min', round(len(segs) / (duration / 60), 1)
               if duration > 0 else None, '1/min', 'free'),
        Metric('transcribe.overlap_seconds', round(diag['overlap_s'], 3), 's', 'free',
               higher_is_better=False),
        Metric('transcribe.overlap_pairs', diag['overlap_pairs'], 'count', 'free',
               higher_is_better=False),
        Metric('transcribe.invalid_intervals', diag['invalid_intervals'], 'count', 'free',
               higher_is_better=False),
    ]
    if not ctx.case.has('ref_transcript'):
        out.extend([missing('transcribe.wer', 'no ref_transcript.jsonl', provenance='gt', unit='ratio'),
                    missing('transcribe.cer', 'no ref_transcript.jsonl', provenance='gt', unit='ratio')])
        return out
    try:
        reference = ' '.join(str(r.get('text', '')) for r in load_jsonl(ctx.case.ref_transcript))
    except (OSError, ValueError) as exc:
        reason = f'unreadable ref_transcript.jsonl: {exc}'
        out.extend([missing('transcribe.wer', reason, provenance='gt', unit='ratio'),
                    missing('transcribe.cer', reason, provenance='gt', unit='ratio')])
        return out
    hypothesis = ' '.join(s.source_text for s in sorted(segs, key=lambda s: s.start))
    rates = text_error_rates(reference, hypothesis)
    for name in ('wer', 'cer'):
        if rates[name] is None:
            out.append(missing(f'transcribe.{name}', 'reference transcript has no text',
                               provenance='gt', unit='ratio'))
        else:
            out.append(Metric(f'transcribe.{name}', round(rates[name], 3), 'ratio', 'gt',
                              higher_is_better=False, note='vs reference transcript'))
    return out
