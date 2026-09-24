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
import contextlib
import re
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

from .. import capabilities, config, store
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
    StepState,
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
    StepContext,
    StepProvider,
    StepSkipped,
    TranslationProvider,
    TTSProvider,
    get_provider_class,
)
from . import analysis, regions, segmentation
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
    """Persist + broadcast the project mid-stage (e.g. after each synthesized segment).
    Uses store.save_merged so user edits made on disk while the stage ran are preserved."""
    async with store.lock(project.id):
        await asyncio.to_thread(store.save_merged, project)
    engine.bus.publish_project(project)


# ---- capability steps ----------------------------------------------------------------------


async def run_steps(
    project: Project, stage: StageKey, slot: capabilities.Slot, progress: ProgressFn
) -> list[str]:
    """Run the enabled capability steps registered for (stage, slot), in order. A failing step
    is recorded and the stage carries on: only the stage body itself can fail a stage. Returns
    one short note per step that ran, for the stage detail."""
    caps = capabilities.steps(stage, slot)
    if not caps:
        return []
    resolved = capabilities.resolve(project.pipeline)
    state = project.stage(stage)
    notes: list[str] = []
    todo = [c for c in caps if resolved[c.id].enabled]
    for cap in caps:
        if cap not in todo:
            state.steps.pop(cap.id, None)
    for i, cap in enumerate(todo):
        choice = resolved[cap.id]
        step = StepState(provider_id=choice.provider_id)
        sub = _sub(progress, i / len(todo), (i + 1) / len(todo))
        try:
            cls = get_provider_class(choice.provider_id)
            options = dict(config.provider_options(choice.provider_id))
            options.update(choice.options)
            provider = cls(options)
            if not isinstance(provider, StepProvider):
                raise StageError(f"'{choice.provider_id}' is not a step provider")
            ok, reason = provider.available()
            if not ok:
                raise StageError(f"not available: {reason}")
            sub(0.0, f"{cap.id} · {cap.name}")
            ctx = StepContext(
                project=project,
                project_dir=store.project_dir(project.id),
                capability_id=cap.id,
                params=choice.params,
                enabled={k: v.enabled for k, v in resolved.items()},
            )
            step.detail = await provider.run(ctx, sub)
            notes.append(f"{cap.id} ok")
        except StepSkipped as skip:
            step.status, step.detail = "skipped", str(skip)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - optional steps degrade, never fail the stage
            step.status, step.detail = "error", str(exc) or type(exc).__name__
            notes.append(f"{cap.id} failed")
        state.steps[cap.id] = step
    return notes


def _with_steps(stage: StageKey, body: StageRunner | None) -> StageRunner:
    """Wrap a stage body with its pre/post capability steps. A stage with no body (analyze,
    review) is skipped when none of its steps are enabled."""

    async def run(project: Project, job: Job, progress: ProgressFn) -> str:
        has_pre = bool(capabilities.steps(stage, "pre"))
        has_post = bool(capabilities.steps(stage, "post"))
        if body is None:
            resolved = capabilities.resolve(project.pipeline)
            if not any(resolved[c.id].enabled for c in capabilities.steps(stage, "post")):
                project.stage(stage).steps.clear()
                raise StageSkipped("no capabilities enabled for this stage")
            notes = await run_steps(project, stage, "post", progress)
            return ", ".join(notes)
        lo, hi = (0.1 if has_pre else 0.0), (0.85 if has_post else 1.0)
        notes = await run_steps(project, stage, "pre", _sub(progress, 0.0, lo))
        detail = await body(project, job, _sub(progress, lo, hi))
        notes += await run_steps(project, stage, "post", _sub(progress, hi, 1.0))
        failed = [n for n in notes if n.endswith("failed")]
        return f"{detail}; {', '.join(failed)}" if failed else detail

    return run


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

    progress(0.92, "detecting shot changes")
    # Anchors and the filmstrip are conveniences: the routes recompute them on demand.
    with contextlib.suppress(RuntimeError):
        await analysis.detect_scenes(project, refresh=True)
        await analysis.filmstrip(project, refresh=True)

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
    with contextlib.suppress(RuntimeError):
        await analysis.detect_silences(project, refresh=True)

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
        ASRSegment(start=max(0.0, s.start), end=s.end, text=s.text.strip(), words=s.words)
        for s in raw
        if s.text.strip() and (s.end - s.start) > _MIN_SLOT
    ]
    if not cleaned:
        raise StageError(f"ASR ({asr.meta.id}) produced no usable segments")

    c1 = capabilities.resolve(project.pipeline)["C1"]
    if c1.provider_id == "segmentation.pause":
        cleaned = segmentation.resegment(
            cleaned,
            min_pause=float(c1.params.get("min_pause") or 0.35),
            max_line_seconds=float(c1.params.get("max_line_seconds") or 12.0),
        )

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
        Segment(
            start=a.start,
            end=a.end,
            source_text=a.text,
            speaker_id=by_label[label].id,
            words=a.words,
        )
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
    regions.apply_skip_ranges(project)

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


# ---- work selection ------------------------------------------------------------------------


def translate_targets(project: Project) -> list[Segment]:
    """Lines the translate stage will process: not skipped, have source text, and are either
    flagged dirty or still untranslated. Shared with the orchestrator's run selection so a line
    without output (a fresh insert) is never left behind."""
    return [
        s
        for s in project.segments
        if not s.skipped
        and s.source_text.strip()
        and (s.translate_dirty or not s.translated_text.strip())
    ]


def synth_targets(project: Project) -> list[Segment]:
    return [
        s
        for s in project.segments
        if not s.skipped
        and s.translated_text.strip()
        and (s.synth_dirty or not s.takes)
    ]


_UNSPACED_LANGS = {"ja", "zh", "th", "yue", "lo", "km", "my"}


def _joiner(lang: str) -> str:
    """Separator between joined lines of text: nothing for scripts written without spaces."""
    return "" if lang.split("-")[0].lower() in _UNSPACED_LANGS else " "


async def transcribe_one(
    project: Project, provider: Provider, segment: Segment, progress: ProgressFn, pad: float = 0.15
) -> None:
    """Re-run ASR over one segment's time range (per-segment correction). Timing is kept — the
    user chose the range; words are offset back into project time."""
    if not isinstance(provider, ASRProvider):
        raise StageError(f"provider '{provider.meta.id}' is not an ASR provider")
    vocals = _require(project, "audio/vocals.wav", "run separate first")
    clip = _out(project, f"audio/segments/{segment.id}/asr_input.wav")
    offset = max(0.0, segment.start - pad)
    try:
        await ffmpeg.slice_audio(vocals, clip, segment.start, segment.end, pad=pad)
        raw = await provider.transcribe(clip, project.source_lang, progress)
    finally:
        clip.unlink(missing_ok=True)
    pieces = [r for r in raw if r.text.strip()]
    if not pieces:
        raise StageError("speech recognition found no speech in this line")
    text = _joiner(project.source_lang).join(r.text.strip() for r in pieces)
    words = []
    for r in pieces:
        for w in r.words:
            start = min(segment.end, max(segment.start, w.start + offset))
            end = min(segment.end, max(start, w.end + offset))
            words.append(w.model_copy(update={"start": round(start, 3), "end": round(end, 3)}))
    if text != segment.source_text:
        segment.source_text = text
        segment.translate_dirty = True
        segment.synth_dirty = True
    segment.words = words


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
    for s in project.segments:
        if s.skipped or not s.source_text.strip():
            s.translate_dirty = False  # nothing to translate: kept original, or an empty line
    targets = translate_targets(project)
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
    if segment.skipped:
        raise StageError("this line keeps the original audio (inside a skip range)")
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
    if seg.skipped:
        raise StageError("this line keeps the original audio (inside a skip range)")
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
        segment_reference_text=seg.source_text,
    )
    await provider.synthesize(request, out_wav, progress)
    if not out_wav.exists():
        raise StageError(f"TTS provider '{provider.meta.id}' produced no output file")

    duration = await ffmpeg.wav_duration(out_wav)
    take = Take(
        path=f"{rel_dir}/take_{n}.wav",
        duration=duration,
        provider_id=provider.meta.id,
        lang=project.target_lang,
    )
    seg.takes.append(take)
    seg.active_take_id = take.id
    seg.synth_dirty = False
    return take


async def run_synthesize(project: Project, job: Job, progress: ProgressFn) -> str:
    provider = _typed_provider(project, "tts", TTSProvider)
    vocals = _require(project, "audio/vocals.wav", "run separate first")
    for s in project.segments:
        if s.skipped:
            s.synth_dirty = False
    targets = synth_targets(project)
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


def _delivery_lufs(params: dict) -> float:
    """E4's delivery preset ("web -16", "EBU R128 -23", …) → its integrated loudness target."""
    try:
        return float(str(params.get("delivery") or "").split()[-1])
    except (IndexError, ValueError):
        return -16.0


async def run_mix(project: Project, job: Job, progress: ProgressFn) -> str:
    if project.media is None or project.media.duration <= 0:
        raise StageError("project media info is missing; run ingest first")
    background = _require(project, "audio/background.wav", "run separate first")

    entries = [
        (seg, take)
        for seg in sorted(project.segments, key=lambda s: (s.start, s.end))
        if not seg.skipped and (take := seg.active_take()) is not None
    ]
    if not entries and not project.skip_ranges:
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
    if not placements and not project.skip_ranges:
        raise StageError("no usable take files found on disk; re-run synthesize")

    progress(0.72, "assembling dub vocal track")
    dub_vocals = _out(project, "audio/dub_vocals.wav")
    await audio_utils.assemble_track(placements, project.media.duration, dub_vocals)

    progress(0.85, "mixing with background")
    dub_mix = _out(project, "audio/dub_mix.wav")
    await audio_utils.mix_tracks(
        dub_vocals,
        background,
        dub_mix,
        reference_vocals=store.resolve(project, "audio/vocals.wav"),
        target_lufs=_delivery_lufs(capabilities.resolve(project.pipeline)["E4"].params),
        keep_original=regions.spans(project),
        original=store.resolve(project, "audio/original.wav"),
    )

    progress(0.91, "muxing dub preview")
    # The editor previews the dub by swapping its <video> source to this file. One file, one
    # media clock: a separate <audio> element kept in sync with the muted video by re-seeking
    # stutters and repeats as soon as either stream hiccups (e.g. streaming over a LAN).
    playback = _require(project, "playback.mp4", "run ingest first")
    await ffmpeg.mux(playback, dub_mix, _out(project, "playback_dub.mp4"))

    progress(0.93, "computing waveform")
    await waveform.generate_peaks(dub_mix, _out(project, "waveforms/dub_mix.json"))

    project.mark_downstream_dirty("mix")
    progress(1.0, "mix complete")
    detail = f"mixed {len(placements)} take(s) over {project.media.duration:.1f}s"
    if skipped:
        detail += f"; {skipped} skipped (missing take file or zero-length slot)"
    if project.skip_ranges:
        detail += f"; original kept in {len(project.skip_ranges)} range(s)"
    return detail


# ---- lipsync -------------------------------------------------------------------------------


async def run_lipsync(project: Project, job: Job, progress: ProgressFn) -> str:
    f1 = capabilities.resolve(project.pipeline)["F1"]
    if not f1.enabled:
        raise StageSkipped(f1.reason or "lip sync is off")
    provider = _typed_provider(project, "lipsync", LipSyncProvider)
    playback = _require(project, "playback.mp4", "run ingest first")
    dub_mix = _require(project, "audio/dub_mix.wav", "run mix first")
    out_video = _out(project, "render/lipsync.mp4")
    spans = regions.spans(project)
    raw = _out(project, "render/lipsync_raw.mp4") if spans else out_video

    progress(0.0, f"lip-syncing with {provider.meta.id}")
    await provider.sync(playback, dub_mix, raw, _sub(progress, 0.0, 0.9 if spans else 1.0))
    if not raw.exists():
        raise StageError(f"lipsync provider '{provider.meta.id}' produced no output file")
    if spans:
        # Skip ranges keep the original picture: overlay it back over the synced video.
        progress(0.9, "restoring the original picture in kept ranges")
        await ffmpeg.overlay_ranges(raw, playback, spans, out_video)
        raw.unlink(missing_ok=True)
        return f"lip-synced with {provider.meta.id}; original picture kept in {len(spans)} range(s)"
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
    "separate": _with_steps("separate", run_separate),
    "analyze": _with_steps("analyze", None),
    "transcribe": _with_steps("transcribe", run_transcribe),
    "translate": _with_steps("translate", run_translate),
    "synthesize": _with_steps("synthesize", run_synthesize),
    "mix": _with_steps("mix", run_mix),
    "lipsync": _with_steps("lipsync", run_lipsync),
    "review": _with_steps("review", None),
    "render": _with_steps("render", run_render),
}
