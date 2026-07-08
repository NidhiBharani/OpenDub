"""Pipeline stage implementations.

One ``async def run_<stage>(project, job, progress) -> str`` per stage. Each mutates the in-memory
``Project`` (the orchestrator persists + publishes it around stage boundaries), reads/writes media
files strictly through ``app.media.ffmpeg`` / ``app.media.waveform`` / ``app.pipeline.audio``, and
reports progress 0..1 through the given callback. Stages raise:

- ``StageError``  — clear, user-facing failure message; the orchestrator marks the stage ``error``.
- ``StageSkipped`` — the stage intentionally did nothing; marked ``skipped``.
"""
from __future__ import annotations

import asyncio
import re
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

from .. import config, store
from ..jobs import engine
from ..media import ffmpeg, waveform
from ..models import (
    SPEAKER_PALETTE,
    ASRSegment,
    Job,
    Project,
    ProviderKind,
    Segment,
    Speaker,
    StageKey,
    Take,
    TranslationRequest,
    TTSRequest,
)
from ..providers.base import (
    ASRProvider,
    DiarizationProvider,
    LipSyncProvider,
    ProgressFn,
    Provider,
    SeparationProvider,
    TranslationProvider,
    TTSProvider,
    get_provider_class,
)
from . import audio as audio_utils

StageRunner = Callable[[Project, Job, ProgressFn], Awaitable[str]]

_P = TypeVar("_P", bound=Provider)

# speaker reference construction (transcribe stage)
_REF_MIN_SEGMENT = 1.0  # only use segments at least this long (seconds)
_REF_MAX_TOTAL = 30.0  # total reference length cap (seconds)
_REF_PAD = 0.05  # padding around each slice (seconds)

_MIN_SLOT = 0.01  # segments shorter than this are considered degenerate


class StageError(RuntimeError):
    """A stage failed with a clear, user-facing message."""


class StageSkipped(Exception):
    """A stage intentionally did nothing (e.g. lipsync provider set to 'none')."""


# ---- shared helpers ------------------------------------------------------------------------


def resolve_provider(project: Project, kind: ProviderKind) -> Provider:
    """Instantiate the provider chosen for `kind`: stored settings (with env overrides) overlaid
    with the project's per-choice options. Raises StageError with the reason when unusable."""
    choice = project.pipeline.choice(kind)
    try:
        cls = get_provider_class(choice.provider_id)
    except KeyError:
        raise StageError(
            f"unknown provider '{choice.provider_id}' configured for '{kind}'"
        ) from None
    options = dict(config.provider_options(choice.provider_id))
    options.update(choice.options or {})
    provider = cls(options)
    ok, reason = provider.available()
    if not ok:
        raise StageError(f"provider '{choice.provider_id}' is not available: {reason}")
    return provider


def _typed_provider(project: Project, kind: ProviderKind, abc: type[_P]) -> _P:
    provider = resolve_provider(project, kind)
    if not isinstance(provider, abc):
        raise StageError(
            f"provider '{provider.meta.id}' does not implement {abc.__name__} (wrong kind?)"
        )
    return provider


def _require(project: Project, rel: str, hint: str) -> Path:
    path = store.resolve(project, rel)
    if not path.exists():
        raise StageError(f"missing {rel} — {hint}")
    return path


def _out(project: Project, rel: str) -> Path:
    """Absolute path for an output artifact, with its parent directory created."""
    path = store.resolve(project, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sub(progress: ProgressFn, lo: float, hi: float) -> ProgressFn:
    """Rescale a 0..1 progress callback into the [lo, hi] slice of the stage."""
    span = hi - lo

    def scaled(fraction: float, message: str = "") -> None:
        try:
            f = float(fraction)
        except (TypeError, ValueError):
            f = 0.0
        if f != f:  # NaN
            f = 0.0
        progress(lo + span * min(1.0, max(0.0, f)), message)

    return scaled


async def _checkpoint(project: Project) -> None:
    """Persist + broadcast the project mid-stage (e.g. after each synthesized segment)."""
    async with store.lock(project.id):
        await asyncio.to_thread(store.save, project)
    engine.bus.publish_project(project)


# ---- ingest --------------------------------------------------------------------------------


async def run_ingest(project: Project, job: Job, progress: ProgressFn) -> str:
    source: Path | None = None
    if project.source_filename:
        candidate = store.resolve(project, project.source_filename)
        if candidate.exists():
            source = candidate
    if source is None:
        project_dir = store.project_dir(project.id)
        matches = sorted(p for p in project_dir.glob("source.*") if p.is_file())
        if not matches:
            raise StageError(
                "no source file found — expected the upload to be saved as source.<ext> "
                "in the project directory"
            )
        if len(matches) > 1:
            raise StageError(
                f"found {len(matches)} source.* files in the project directory; "
                "cannot determine which one to ingest"
            )
        source = matches[0]
        project.source_filename = source.name

    progress(0.02, "probing source media")
    info = await ffmpeg.probe(source)
    project.media = info
    if not info.has_audio:
        raise StageError(f"'{project.source_filename}' has no audio track; nothing to dub")
    if info.duration <= 0:
        raise StageError(f"could not determine the duration of '{project.source_filename}'")

    progress(0.05, "encoding playback video")
    await ffmpeg.make_playback(source, _out(project, "playback.mp4"))

    progress(0.65, "extracting original audio")
    original = _out(project, "audio/original.wav")
    await ffmpeg.extract_audio(source, original)

    progress(0.85, "computing waveform")
    await waveform.generate_peaks(original, _out(project, "waveforms/original.json"))

    project.mark_downstream_dirty("ingest")
    progress(1.0, "ingest complete")
    return f"{info.duration:.1f}s, {info.width}x{info.height} @ {info.fps:.4g} fps"


# ---- separate ------------------------------------------------------------------------------


async def run_separate(project: Project, job: Job, progress: ProgressFn) -> str:
    provider = _typed_provider(project, "separation", SeparationProvider)
    original = _require(project, "audio/original.wav", "run ingest first")
    vocals = _out(project, "audio/vocals.wav")
    background = _out(project, "audio/background.wav")

    progress(0.0, f"separating vocals with {provider.meta.id}")
    await provider.separate(original, vocals, background, _sub(progress, 0.0, 0.9))
    if not vocals.exists() or not background.exists():
        raise StageError(f"separation provider '{provider.meta.id}' did not produce output files")

    progress(0.9, "computing vocals waveform")
    await waveform.generate_peaks(vocals, _out(project, "waveforms/vocals.json"))

    project.mark_downstream_dirty("separate")
    progress(1.0, "separation complete")
    return f"separated with {provider.meta.id}"


# ---- transcribe ----------------------------------------------------------------------------


async def run_transcribe(project: Project, job: Job, progress: ProgressFn) -> str:
    asr = _typed_provider(project, "asr", ASRProvider)
    diarizer = _typed_provider(project, "diarization", DiarizationProvider)
    vocals = _require(project, "audio/vocals.wav", "run separate first")

    progress(0.0, f"transcribing with {asr.meta.id}")
    raw = await asr.transcribe(vocals, project.source_lang, _sub(progress, 0.0, 0.55))
    cleaned = [
        ASRSegment(start=max(0.0, s.start), end=s.end, text=s.text.strip())
        for s in raw
        if s.text.strip() and (s.end - s.start) > _MIN_SLOT
    ]
    if not cleaned:
        raise StageError(f"ASR ({asr.meta.id}) produced no usable segments")

    progress(0.55, f"diarizing with {diarizer.meta.id}")
    labels = await diarizer.diarize(vocals, cleaned, _sub(progress, 0.55, 0.75))
    if len(labels) != len(cleaned):
        raise StageError(
            f"diarization ({diarizer.meta.id}) returned {len(labels)} labels "
            f"for {len(cleaned)} segments"
        )

    # Speakers named in label order of first appearance, colors from the palette.
    speakers: list[Speaker] = []
    by_label: dict[str, Speaker] = {}
    for label in labels:
        if label not in by_label:
            speaker = Speaker(
                name=f"Speaker {len(speakers) + 1}",
                color=SPEAKER_PALETTE[len(speakers) % len(SPEAKER_PALETTE)],
            )
            by_label[label] = speaker
            speakers.append(speaker)

    segments = [
        Segment(start=a.start, end=a.end, source_text=a.text, speaker_id=by_label[label].id)
        for a, label in zip(cleaned, labels)
    ]
    segments.sort(key=lambda s: (s.start, s.end))

    # Segments are replaced wholesale — clear stale per-segment/per-speaker artifacts.
    for rel in ("speakers", "audio/segments"):
        stale = store.resolve(project, rel)
        if stale.exists():
            await asyncio.to_thread(shutil.rmtree, stale, True)

    project.segments = segments
    project.speakers = speakers

    progress(0.75, "building speaker references")
    for i, speaker in enumerate(speakers):
        await _build_speaker_reference(project, speaker, vocals)
        progress(0.75 + 0.25 * (i + 1) / len(speakers), f"reference for {speaker.name}")

    project.mark_downstream_dirty("transcribe")
    return f"{len(segments)} segments, {len(speakers)} speaker(s)"


async def _build_speaker_reference(project: Project, speaker: Speaker, vocals: Path) -> None:
    """Concat the speaker's longest segments (each ≥1 s, sliced from vocals.wav with a small pad)
    up to ~30 s total into speakers/<id>/reference.wav."""
    segs = [s for s in project.segments if s.speaker_id == speaker.id]
    if not segs:
        return
    candidates = sorted(
        (s for s in segs if s.duration >= _REF_MIN_SEGMENT),
        key=lambda s: s.duration,
        reverse=True,
    )
    if not candidates:  # nothing ≥1 s — fall back to the single longest utterance
        candidates = [max(segs, key=lambda s: s.duration)]
    picked: list[Segment] = []
    total = 0.0
    for s in candidates:
        if picked and total + s.duration > _REF_MAX_TOTAL:
            continue  # too big for the remaining budget; a shorter one may still fit
        picked.append(s)
        total += s.duration
        if total >= _REF_MAX_TOTAL:
            break
    picked.sort(key=lambda s: s.start)  # chronological order sounds most natural

    ref_rel = f"speakers/{speaker.id}/reference.wav"
    ref = _out(project, ref_rel)
    parts: list[Path] = []
    try:
        for i, s in enumerate(picked):
            part = ref.parent / f"part_{i}.wav"
            await ffmpeg.slice_audio(vocals, part, s.start, s.end, pad=_REF_PAD)
            parts.append(part)
        await ffmpeg.concat_audio(parts, ref)
        speaker.reference_path = ref_rel
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


# ---- translate -----------------------------------------------------------------------------


def _translation_request(
    project: Project, ordered: list[Segment], pos: dict[str, int], seg: Segment
) -> TranslationRequest:
    i = pos[seg.id]
    speaker = project.speaker(seg.speaker_id)
    return TranslationRequest(
        text=seg.source_text,
        emotion=seg.emotion,
        speaker_name=speaker.name if speaker else "",
        duration=seg.duration,
        context_before=[s.source_text for s in ordered[max(0, i - 2) : i]],
        context_after=[s.source_text for s in ordered[i + 1 : i + 3]],
    )


async def run_translate(project: Project, job: Job, progress: ProgressFn) -> str:
    provider = _typed_provider(project, "translation", TranslationProvider)
    targets = [s for s in project.segments if s.translate_dirty or not s.translated_text.strip()]
    for s in [t for t in targets if not t.source_text.strip()]:
        s.translate_dirty = False  # nothing to translate for empty lines
    targets = [s for s in targets if s.source_text.strip()]
    if not targets:
        return "nothing to translate"

    ordered = sorted(project.segments, key=lambda s: (s.start, s.end))
    pos = {s.id: i for i, s in enumerate(ordered)}
    requests = [_translation_request(project, ordered, pos, s) for s in targets]

    progress(0.02, f"translating {len(requests)} segment(s) with {provider.meta.id}")
    results = await provider.translate(
        requests, project.source_lang, project.target_lang, _sub(progress, 0.02, 0.95)
    )
    if len(results) != len(requests):
        raise StageError(
            f"translation provider '{provider.meta.id}' returned {len(results)} results "
            f"for {len(requests)} requests"
        )

    changed = 0
    for seg, text in zip(targets, results):
        new = (text or "").strip()
        if new != seg.translated_text:
            seg.translated_text = new
            seg.synth_dirty = True  # a new line must be re-voiced
            changed += 1
        seg.translate_dirty = False
    if changed:
        project.mark_downstream_dirty("translate")
    progress(1.0, "translation complete")
    return f"translated {len(targets)} segment(s)"


async def translate_one(
    project: Project, provider: Provider, segment: Segment, progress: ProgressFn
) -> None:
    """Force-translate a single segment (per-segment regeneration)."""
    if not isinstance(provider, TranslationProvider):
        raise StageError(f"provider '{provider.meta.id}' is not a translation provider")
    if not segment.source_text.strip():
        raise StageError("segment has no source text to translate")
    ordered = sorted(project.segments, key=lambda s: (s.start, s.end))
    pos = {s.id: i for i, s in enumerate(ordered)}
    request = _translation_request(project, ordered, pos, segment)
    results = await provider.translate(
        [request], project.source_lang, project.target_lang, progress
    )
    if not results:
        raise StageError(f"translation provider '{provider.meta.id}' returned no result")
    new = (results[0] or "").strip()
    if new != segment.translated_text:
        segment.translated_text = new
        segment.synth_dirty = True
    segment.translate_dirty = False


# ---- synthesize ----------------------------------------------------------------------------


def _next_take_number(seg_dir: Path) -> int:
    highest = 0
    if seg_dir.exists():
        for p in seg_dir.glob("take_*.wav"):
            m = re.fullmatch(r"take_(\d+)\.wav", p.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1


async def _synthesize_segment(
    project: Project, provider: TTSProvider, seg: Segment, vocals: Path, progress: ProgressFn
) -> Take:
    if not seg.translated_text.strip():
        raise StageError("segment has no translated text; run translate first")
    if seg.duration <= _MIN_SLOT:
        raise StageError("segment has zero duration")

    rel_dir = f"audio/segments/{seg.id}"
    seg_dir = store.resolve(project, rel_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)

    # The segment's own source audio doubles as the emotion/prosody style reference.
    source_ref = seg_dir / "source.wav"
    await ffmpeg.slice_audio(vocals, source_ref, seg.start, seg.end)

    n = _next_take_number(seg_dir)
    out_wav = seg_dir / f"take_{n}.wav"

    speaker = project.speaker(seg.speaker_id)
    speaker_ref: str | None = None
    if speaker and speaker.reference_path:
        ref = store.resolve(project, speaker.reference_path)
        if ref.exists():
            speaker_ref = str(ref)

    request = TTSRequest(
        text=seg.translated_text,
        language=project.target_lang,
        emotion=seg.emotion,
        target_duration=seg.duration,
        speaker_reference=speaker_ref,
        segment_reference=str(source_ref),
    )
    await provider.synthesize(request, out_wav, progress)
    if not out_wav.exists():
        raise StageError(f"TTS provider '{provider.meta.id}' produced no output file")

    duration = await ffmpeg.wav_duration(out_wav)
    take = Take(path=f"{rel_dir}/take_{n}.wav", duration=duration, provider_id=provider.meta.id)
    seg.takes.append(take)
    seg.active_take_id = take.id
    seg.synth_dirty = False
    return take


async def run_synthesize(project: Project, job: Job, progress: ProgressFn) -> str:
    provider = _typed_provider(project, "tts", TTSProvider)
    vocals = _require(project, "audio/vocals.wav", "run separate first")
    targets = [s for s in project.segments if s.synth_dirty or not s.takes]
    if not targets:
        return "nothing to synthesize"

    total = len(targets)
    errors: list[str] = []
    done = 0
    for i, seg in enumerate(targets):
        progress(i / total, f"synthesizing segment {i + 1}/{total}")
        try:
            await _synthesize_segment(
                project, provider, seg, vocals, _sub(progress, i / total, (i + 1) / total)
            )
            done += 1
            await _checkpoint(project)  # persist per segment so the UI updates live
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            errors.append(f"{seg.id}: {str(exc) or type(exc).__name__}")
    if done == 0:
        raise StageError(errors[0] if errors else "no segments to synthesize")

    project.mark_downstream_dirty("synthesize")
    progress(1.0, f"synthesized {done}/{total}")
    detail = f"synthesized {done}/{total} segment(s)"
    if errors:
        detail += f"; {len(errors)} failed (first: {errors[0]})"
    return detail


async def synthesize_one(
    project: Project, provider: Provider, segment: Segment, progress: ProgressFn
) -> Take:
    """Force-synthesize a single segment (per-segment regeneration)."""
    if not isinstance(provider, TTSProvider):
        raise StageError(f"provider '{provider.meta.id}' is not a TTS provider")
    vocals = _require(project, "audio/vocals.wav", "run separate first")
    return await _synthesize_segment(project, provider, segment, vocals, progress)


# ---- mix -----------------------------------------------------------------------------------


async def run_mix(project: Project, job: Job, progress: ProgressFn) -> str:
    if project.media is None or project.media.duration <= 0:
        raise StageError("project media info is missing; run ingest first")
    background = _require(project, "audio/background.wav", "run separate first")

    entries = [
        (seg, take)
        for seg in sorted(project.segments, key=lambda s: (s.start, s.end))
        if (take := seg.active_take()) is not None
    ]
    if not entries:
        raise StageError("no segments have an active take; run synthesize first")

    placements: list[tuple[float, Path]] = []
    skipped = 0
    for i, (seg, take) in enumerate(entries):
        take_path = store.resolve(project, take.path)
        if not take_path.exists() or seg.duration <= _MIN_SLOT:
            skipped += 1
            continue
        fitted = _out(project, f"audio/segments/{seg.id}/fitted.wav")
        rate = await audio_utils.fit_to_duration(take_path, fitted, seg.duration)
        take.rate_factor = rate
        placements.append((seg.start, fitted))
        progress(0.7 * (i + 1) / len(entries), f"fitting take {i + 1}/{len(entries)}")
    if not placements:
        raise StageError("no usable take files found on disk; re-run synthesize")

    progress(0.72, "assembling dub vocal track")
    dub_vocals = _out(project, "audio/dub_vocals.wav")
    await audio_utils.assemble_track(placements, project.media.duration, dub_vocals)

    progress(0.85, "mixing with background")
    dub_mix = _out(project, "audio/dub_mix.wav")
    await audio_utils.mix_tracks(dub_vocals, background, dub_mix)

    progress(0.93, "computing waveform")
    await waveform.generate_peaks(dub_mix, _out(project, "waveforms/dub_mix.json"))

    project.mark_downstream_dirty("mix")
    progress(1.0, "mix complete")
    detail = f"mixed {len(placements)} take(s) over {project.media.duration:.1f}s"
    if skipped:
        detail += f"; {skipped} skipped (missing take file or zero-length slot)"
    return detail


# ---- lipsync -------------------------------------------------------------------------------


async def run_lipsync(project: Project, job: Job, progress: ProgressFn) -> str:
    if project.pipeline.choice("lipsync").provider_id == "lipsync.none":
        raise StageSkipped("lipsync provider is set to 'none'")
    provider = _typed_provider(project, "lipsync", LipSyncProvider)
    playback = _require(project, "playback.mp4", "run ingest first")
    dub_mix = _require(project, "audio/dub_mix.wav", "run mix first")
    out_video = _out(project, "render/lipsync.mp4")

    progress(0.0, f"lip-syncing with {provider.meta.id}")
    await provider.sync(playback, dub_mix, out_video, _sub(progress, 0.0, 1.0))
    if not out_video.exists():
        raise StageError(f"lipsync provider '{provider.meta.id}' produced no output file")
    return f"lip-synced with {provider.meta.id}"


# ---- render --------------------------------------------------------------------------------


async def run_render(project: Project, job: Job, progress: ProgressFn) -> str:
    dub_mix = _require(project, "audio/dub_mix.wav", "run mix first")
    lipsync_video = store.resolve(project, "render/lipsync.mp4")
    use_lipsync = project.stage("lipsync").status == "done" and lipsync_video.exists()
    video = lipsync_video if use_lipsync else _require(project, "playback.mp4", "run ingest first")

    progress(0.05, "muxing final video")
    await ffmpeg.mux(video, dub_mix, _out(project, "render/dubbed.mp4"))
    progress(1.0, "render complete")
    return f"render/dubbed.mp4 (video: {'lipsync' if use_lipsync else 'playback'})"


STAGE_RUNNERS: dict[StageKey, StageRunner] = {
    "ingest": run_ingest,
    "separate": run_separate,
    "transcribe": run_transcribe,
    "translate": run_translate,
    "synthesize": run_synthesize,
    "mix": run_mix,
    "lipsync": run_lipsync,
    "render": run_render,
}
