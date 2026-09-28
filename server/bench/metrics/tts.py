"""TTS quality metrics over every translated, unskipped segment."""
from __future__ import annotations

import json

from app.pipeline.quality import audio_stats

from .base import Metric, MetricContext, missing

_DEGENERATE_MAX_DUR = 0.25
_DEGENERATE_MIN_PEAK = -50.0
_RATE_CLAMP = 1.15


async def score(ctx: MetricContext) -> list[Metric]:
    segs = [s for s in ctx.project.segments if s.translated_text.strip() and not s.skipped]
    if not segs:
        return [missing('tts.degenerate_take_rate', 'no translated segments to synthesize')]

    degenerate = missing_count = unreadable_count = clamped = 0
    wers, cers, similarities, deliveries = [], [], [], []
    for segment in segs:
        take = segment.active_take()
        if take is None:
            missing_count += 1
            degenerate += 1
            continue
        stats = audio_stats(ctx.path(take.path))
        if stats['status'] != 'ok':
            degenerate += 1
            if stats['status'] == 'missing':
                missing_count += 1
            else:
                unreadable_count += 1
            continue
        if (stats['duration_s'] < _DEGENERATE_MAX_DUR or
                stats['peak_dbfs'] < _DEGENERATE_MIN_PEAK):
            degenerate += 1
        if take.rate_factor >= _RATE_CLAMP - 1e-3:
            clamped += 1
        report_path = ctx.path(f'quality/segments/{segment.id}.json')
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            report = {}
        synthesis = report.get('synthesis') or {}
        fitted = report.get('fitted') or {}
        if fitted.get('take_id') is not None and fitted['take_id'] != take.id:
            fitted = {}
        if synthesis.get('take_id') is not None and synthesis['take_id'] != take.id:
            synthesis = {}
        verification = (fitted.get('verification') or
                        (synthesis.get('attempts') or [{}])[-1].get('verification') or {})
        if verification.get('status') == 'ok':
            if isinstance(verification.get('wer'), (int, float)):
                wers.append(verification['wer'])
            if isinstance(verification.get('cer'), (int, float)):
                cers.append(verification['cer'])
        similarity = synthesis.get('speaker_similarity') or {}
        if similarity.get('status') == 'ok' and isinstance(similarity.get('cosine'), (int, float)):
            similarities.append(similarity['cosine'])
        delivery = fitted.get('delivery') or synthesis.get('delivery') or {}
        if delivery.get('status') == 'measured' and isinstance(delivery.get('score'), (int, float)):
            deliveries.append(delivery['score'])

    count = len(segs)
    return [
        Metric('tts.degenerate_take_rate', round(degenerate / count, 3), 'ratio', 'free',
               higher_is_better=False,
               note=f'{degenerate}/{count} degenerate, missing, or unreadable takes'),
        Metric('tts.missing_take_count', missing_count, 'count', 'free', higher_is_better=False),
        Metric('tts.unreadable_take_count', unreadable_count, 'count', 'free',
               higher_is_better=False),
        Metric('tts.duration_clamp_rate', round(clamped / count, 3), 'ratio', 'free',
               higher_is_better=False,
               note=f'{clamped}/{count} takes at the {_RATE_CLAMP} maximum speedup'),
        Metric('tts.round_trip_wer', round(sum(wers) / len(wers), 3) if wers else None,
               'ratio', 'free', higher_is_better=False,
               note=f'{len(wers)}/{count} takes verified by configured ASR'),
        Metric('tts.round_trip_cer', round(sum(cers) / len(cers), 3) if cers else None,
               'ratio', 'free', higher_is_better=False,
               note=f'{len(cers)}/{count} takes verified by configured ASR'),
        Metric('tts.verification_coverage', round(len(wers) / count, 3), 'ratio', 'free',
               higher_is_better=True, note='fraction of expected takes with numeric ASR WER'),
        Metric('tts.speaker_similarity', round(sum(similarities) / len(similarities), 3)
               if similarities else None, 'cosine', 'free', higher_is_better=True,
               note=f'{len(similarities)}/{count} takes verified by configured embedder'),
        Metric('tts.speaker_coverage', round(len(similarities) / count, 3), 'ratio', 'free',
               higher_is_better=True, note='fraction of expected takes with measured identity'),
        Metric('tts.delivery_score', round(sum(deliveries) / len(deliveries), 3)
               if deliveries else None, 'ratio', 'free', higher_is_better=True,
               note=f'{len(deliveries)}/{count} acoustic delivery comparisons; heuristic only'),
        Metric('tts.delivery_coverage', round(len(deliveries) / count, 3), 'ratio', 'free',
               higher_is_better=True),
    ]
