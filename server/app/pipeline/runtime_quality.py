"""Backend-only quality policy, provenance, and per-stage/per-line reports.

Configure under providers.quality.runtime in settings.yaml or OPENDUB_QUALITY_RUNTIME_*.
Optional model paths are local-only. Unknown measurements are never represented as passes.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .. import config, store
from ..models import Project, Segment
from ..providers._runtime import manager
from . import quality


def options() -> dict:
    return config.provider_options('quality.runtime')


def enabled(key: str, default: bool = False) -> bool:
    value = options().get(key, default)
    return str(value).lower() in ('true', '1', 'yes', 'on')


def number(key: str, default: float) -> float:
    try:
        value = float(options().get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read(project: Project, name: str) -> dict:
    path = store.resolve(project, f'quality/{name}.json')
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write(project: Project, name: str, data: dict) -> None:
    path = store.resolve(project, f'quality/{name}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**data, 'schema_version': 1, 'updated_at': datetime.now(UTC).isoformat()}
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def segment_report(project: Project, seg: Segment, section: str, data: dict) -> None:
    name = f'segments/{seg.id}'
    report = read(project, name)
    report.update(segment_id=seg.id, start=seg.start, end=seg.end)
    report[section] = data
    write(project, name, report)


def fingerprint(project: Project, seg: Segment) -> str:
    path = store.resolve(project, 'audio/vocals.wav')
    signature = (seg.start, seg.end, seg.source_text, project.source_lang,
                 path.stat().st_mtime_ns if path.exists() else None,
                 options().get('emotion_model'), options().get('emotion_languages'),
                 os.environ.get('OPENDUB_EMOTION_MODEL'), os.environ.get('OPENDUB_EMOTION_LANGUAGES'))
    return hashlib.sha256(repr(signature).encode()).hexdigest()


async def analyze_delivery(clip: Path, language: str) -> dict:
    from .emotion import analyze_audio
    opts = options()
    languages = opts.get('emotion_languages')
    if isinstance(languages, str):
        languages = [item.strip() for item in languages.split(',') if item.strip()]
    return await asyncio.to_thread(analyze_audio, clip, language=language,
                                   model=opts.get('emotion_model', 'auto'),
                                   supported_languages=languages)


async def source_delivery(project: Project, seg: Segment, clip: Path) -> dict:
    signature = fingerprint(project, seg)
    cached = read(project, f'segments/{seg.id}').get('source_delivery', {})
    if cached.get('fingerprint') == signature:
        return cached
    result = await analyze_delivery(clip, project.source_lang)
    result['fingerprint'] = signature
    segment_report(project, seg, 'source_delivery', result)
    return result


def emotion_hint(project: Project, seg: Segment) -> str:
    from .emotion import delivery_hint
    if seg.emotion.strip():
        return seg.emotion  # explicit human guidance takes precedence
    result = read(project, f'segments/{seg.id}').get('source_delivery', {})
    if result.get('fingerprint') != fingerprint(project, seg):
        return ''
    return delivery_hint(result)


def _load_verifier(path: str):
    if not Path(path).is_dir():
        raise ValueError('asr_model must be a local Faster-Whisper model directory')
    from faster_whisper import WhisperModel
    return WhisperModel(path, device='cpu', compute_type='int8', local_files_only=True)


class _ManagedVerifier:
    """Faster-Whisper verifier that holds its managed model only while transcribing, inside
    the worker thread, so work abandoned by a cancelled job keeps the model pinned until done."""

    def __init__(self, path: str) -> None:
        self.key = ('quality.verifier', path, 'cpu', 'int8')
        self.path = path

    def load(self):
        return _load_verifier(self.path)

    def transcribe(self, audio: str, language: str | None = None):
        with manager.acquire(self.key, self.load, est_vram_gb=0.0) as model:
            segments, info = model.transcribe(audio, language=language)
            return list(segments), info  # drain the lazy iterator while the model is held


async def verify(path: Path, text: str, language: str) -> dict:
    model_path = options().get('asr_model')
    if not model_path:
        return await quality.verify_speech(path, text, model=None, language=language)
    verifier = _ManagedVerifier(str(model_path))
    try:
        # Load up front so a missing model is 'unavailable'; verify_speech itself never raises.
        async with manager.acquire_async(verifier.key, verifier.load, est_vram_gb=0.0):
            return await quality.verify_speech(path, text, model=verifier, language=language)
    except Exception as exc:  # noqa: BLE001 - optional model load has an explicit unavailable result
        return {'status': 'unavailable', 'reason': str(exc), 'wer': None, 'cer': None}


async def speaker_similarity(reference: str | Path, take: str | Path) -> dict:
    """Identity cosine with the configured local SpeechBrain embedder, through the manager."""
    embedder_path = options().get('speaker_model')
    if not embedder_path:
        return {'status': 'unavailable', 'reason': 'no identity reference or local embedder'}
    path = str(embedder_path)
    key = ('quality.speaker', path, 'cpu')

    def load():
        return quality.load_speaker_embedder(path)

    def embed(audio_path: str):
        with manager.acquire(key, load, est_vram_gb=0.0) as embedder:
            return embedder(audio_path)

    try:
        async with manager.acquire_async(key, load, est_vram_gb=0.0):
            return await quality.speaker_similarity(reference, take, embed)
    except Exception as exc:  # noqa: BLE001 - optional model load has an explicit unavailable result
        return {'status': 'unavailable', 'reason': str(exc)}


def speech_failures(stats: dict, verification: dict, *, text: str) -> list[str]:
    failures = []
    if stats.get('status') != 'ok':
        return [stats.get('reason') or 'unreadable speech audio']
    if stats.get('duration_s', 0) < 0.08 or stats.get('peak_dbfs', -120) < -50:
        failures.append('missing or nearly silent speech')
    if stats.get('clipping_fraction', 0) > number('max_clipping_fraction', 0.01):
        failures.append('excessive clipped samples')
    if verification.get('status') == 'ok':
        # CER remains useful for unspaced scripts and short lines, whose WER is unstable.
        cer = verification.get('cer')
        if cer is None or cer > number('max_cer', 0.2):
            failures.append('generated speech does not match the script')
    elif enabled('require_verification'):
        failures.append('speech verification unavailable: ' + verification.get('reason', 'unknown'))
    return failures


def _redact_identifier(value: Any) -> Any:
    if isinstance(value, str) and '://' in value:
        parsed = urlsplit(value)
        host = parsed.hostname or ''
        return urlunsplit((parsed.scheme, host, parsed.path, '', ''))
    return value


def provenance(project: Project) -> dict:
    # Never persist API keys, endpoint credentials or arbitrary provider options.
    providers = {}
    for kind in ('separation', 'asr', 'diarization', 'translation', 'tts', 'lipsync'):
        choice = project.pipeline.choice(kind)
        from ..providers.base import get_provider_class
        try:
            defaults = {field.key: field.default for field in get_provider_class(choice.provider_id).meta.fields}
        except KeyError:
            defaults = {}
        opts = {**defaults, **config.provider_options(choice.provider_id), **choice.options}
        providers[kind] = {'id': choice.provider_id, 'settings': {
            k: _redact_identifier(opts[k]) for k in ('model', 'ckpt_file', 'speed', 'seed', 'device', 'compute_type')
            if k in opts}}
    policy = {key: _redact_identifier(value) for key, value in options().items()
              if key in ('tts_attempts', 'max_cer', 'require_verification', 'asr_model',
                         'emotion_model', 'emotion_languages', 'require_emotion_match',
                         'speaker_model', 'level_outliers', 'background_gain_db', 'ducking_mix')}
    return {'providers': providers, 'source_lang': project.source_lang,
            'target_lang': project.target_lang, 'quality_policy': policy}


async def stage_report(project: Project, stage: str, *, error: str | None = None) -> None:
    data: dict = {'stage': stage, 'status': 'error' if error else 'measured',
                  'error': error, **provenance(project)}
    paths = {'ingest': ['audio/original.wav'],
             'separate': ['audio/vocals.wav', 'audio/background.wav'],
             'mix': ['audio/dub_vocals.wav', 'audio/dub_mix.wav']}
    if stage in paths and not error:
        data['audio'] = {rel: await asyncio.to_thread(quality.audio_stats, store.resolve(project, rel))
                         for rel in paths[stage]}
    if error:
        data['artifact_status'] = 'not_assessed: stage failed; existing files may be from an older run'
    if stage == 'mix' and not error:
        data['mix'] = await quality.mix_statistics(
            store.resolve(project, 'audio/dub_mix.wav'), store.resolve(project, 'audio/dub_vocals.wav'),
            store.resolve(project, 'audio/background.wav'),
            [(s.start, s.end) for s in project.segments if not s.skipped],
            gains=read(project, 'mix_levels'),
            target_lufs=read(project, 'mix_levels').get('target_lufs', -16.0))
    if stage == 'transcribe':
        data['timing'] = quality.coverage_diagnostics(
            [(s.start, s.end) for s in project.segments], project.media.duration if project.media else 0)
        data['word_timing_missing'] = sum(not s.words for s in project.segments)
        data['word_boundary_errors'] = sum(
            not (s.start <= w.start < w.end <= s.end) for s in project.segments for w in s.words)
        data['wer_cer'] = {'status': 'unavailable', 'reason': 'requires reference transcript'}
    if stage == 'translate':
        data['segments'] = [{'id': s.id, 'empty': not s.translated_text.strip(),
                             'characters_per_second': len(s.translated_text) / max(.01, s.duration)}
                            for s in project.segments if not s.skipped and s.source_text.strip()]
        data['adequacy'] = {'status': 'unavailable', 'reason': 'requires bilingual reference or judge'}
    if stage in ('synthesize', 'mix'):
        section = 'synthesis' if stage == 'synthesize' else 'fitted'
        data['segments'] = [{'id': s.id, **read(project, f'segments/{s.id}').get(section, {
            'status': 'unavailable', 'reason': 'no measurement for this segment'})}
                            for s in project.segments if not s.skipped]
        data['failure_count'] = sum(x.get('status') == 'failed' for x in data['segments'])
        data['unmeasured_count'] = sum(x.get('status') == 'unavailable' for x in data['segments'])
    write(project, stage, data)
