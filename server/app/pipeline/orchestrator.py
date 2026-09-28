"""Pipeline orchestrator — turns API requests into jobs on the engine.

``start_pipeline_job`` resolves which stages to run, snapshots + queues their StageStates and
submits a runner that executes them in order: each stage goes running → done/skipped/error with a
detail message and timestamps, the project is saved + broadcast after every transition, and overall
job progress is ``(completed_stages + stage_fraction) / total_stages``.

``start_segment_job`` re-runs translate and/or synthesize for a single segment, then marks
mix/lipsync/render dirty (it does NOT auto-run mix).

Model lifecycle (see providers/_runtime.py): each stage runs inside a model-manager hold, so the
models it loads stay warm for the whole stage. At every stage boundary, idle models the next stage
does not use are unloaded, and when a pipeline job ends (done, error or cancelled) every idle model
is unloaded. Segment jobs keep local models warm for the idle TTL (quick successive regenerations)
but release LLM servers as soon as they finish.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import Callable, Sequence

from .. import store
from ..jobs import Runner, engine
from ..models import STAGE_ORDER, Job, Project, StageKey, StageStatus, now
from ..providers._runtime import family, manager
from ..providers.base import ProgressFn
from . import stages as stage_impl
from . import versions
from .stages import StageSkipped, resolve_provider

log = logging.getLogger(__name__)

_SEGMENT_STAGES: tuple[StageKey, ...] = ("transcribe", "translate", "synthesize")

# Model families (first part of a model-manager key) each stage loads in-process; stages not
# listed load none (demucs and the lip-sync models run as subprocesses). 'llm' is deliberately in
# no set: an Ollama model left resident after translation (~9 GB) is what starves the next GPU stage.
_STAGE_MODELS: dict[StageKey, frozenset[str]] = {
    "transcribe": frozenset({"asr", "diarization", "quality"}),
    "synthesize": frozenset({"tts", "asr", "quality"}),  # F5's verifier shares asr's whisper
    "mix": frozenset({"quality"}),
}


# ---- public API ------------------------------------------------------------------------------


def start_pipeline_job(project_id: str, stages: list[StageKey] | None = None) -> Job:
    """Queue a pipeline job. ``stages=None`` runs every stage whose status is not ``done``;
    an explicit list runs exactly those stages, always in STAGE_ORDER order."""
    project = store.load(project_id)
    if project is None:
        raise KeyError(f"unknown project '{project_id}'")

    if stages is None:
        selected = [k for k in STAGE_ORDER if _stage_needs_run(project, k)]
        if not selected:
            raise ValueError("all stages are already done; pass explicit stages to re-run")
    else:
        bad = sorted({s for s in stages if s not in STAGE_ORDER})
        if bad:
            raise ValueError(f"unknown stage(s): {', '.join(bad)}")
        wanted = set(stages)
        selected = [k for k in STAGE_ORDER if k in wanted]
        if not selected:
            raise ValueError("no stages requested")

    kind = "ingest" if selected == ["ingest"] else "pipeline"
    job = Job(project_id=project_id, kind=kind, stages=selected)

    # Snapshot prior statuses (to restore stages that end up never running) and mark queued.
    prior: dict[StageKey, StageStatus] = {k: project.stage(k).status for k in selected}
    for key in selected:
        state = project.stage(key)
        state.status = "queued"
        state.detail = ""
        state.updated_at = now()
    store.save(project)
    engine.bus.publish_project(project)

    engine.submit(
        job,
        _pipeline_runner(project_id, prior),
        cleanup=_abandoned_cleanup(project_id, selected, prior),
    )
    return job


def start_segment_job(project_id: str, segment_id: str, stages: list[str]) -> Job:
    """Queue a per-segment regeneration job (translate and/or synthesize for ONE segment)."""
    project = store.load(project_id)
    if project is None:
        raise KeyError(f"unknown project '{project_id}'")
    segment = project.segment(segment_id)
    if segment is None:
        raise KeyError(f"unknown segment '{segment_id}' in project '{project_id}'")
    if segment.skipped and any(s in ("translate", "synthesize") for s in stages):
        raise ValueError("this line keeps the original audio (inside a skip range)")

    bad = sorted({s for s in stages if s not in _SEGMENT_STAGES})
    if bad:
        raise ValueError(
            f"segment jobs support stages {list(_SEGMENT_STAGES)}; got: {', '.join(bad)}"
        )
    wanted = set(stages)
    selected: list[StageKey] = [k for k in _SEGMENT_STAGES if k in wanted]
    if not selected:
        raise ValueError("no stages requested; pass 'transcribe', 'translate' and/or 'synthesize'")

    job = Job(project_id=project_id, kind="segment", stages=selected, segment_id=segment_id)
    engine.submit(job, _segment_runner(project_id, segment_id))
    return job


# ---- runners ---------------------------------------------------------------------------------


def _pipeline_runner(project_id: str, prior: dict[StageKey, StageStatus]) -> Runner:
    async def run(job: Job) -> None:
        try:
            await _run_pipeline(job)
        finally:
            await _release_models(None)

    async def _run_pipeline(job: Job) -> None:
        total = len(job.stages)
        project: Project | None = None
        try:
            for index, key in enumerate(job.stages):
                # Reload per stage so edits made between stages are picked up, and so each stage
                # persists against the freshest manifest.
                project = store.load(project_id, track=True)
                if project is None:
                    raise RuntimeError("project was deleted while the job was running")

                state = project.stage(key)
                job.stage = key
                job.progress = index / total
                job.message = f"running {key}"
                state.status = "running"
                state.detail = ""
                state.updated_at = now()
                await _checkpoint(project)
                engine.bus.publish_job(job)

                progress = _stage_progress(job, index, total)
                try:
                    async with manager.hold_async():
                        detail = await stage_impl.STAGE_RUNNERS[key](project, job, progress)
                    state.status = "done"
                    state.detail = detail or ""
                except StageSkipped as skip:
                    state.status = "skipped"
                    state.detail = str(skip)
                except asyncio.CancelledError:
                    raise  # stage/queued statuses restored + persisted by the outer handler
                except Exception as exc:
                    message = str(exc) or type(exc).__name__
                    state.status = "error"
                    state.detail = message
                    state.updated_at = now()
                    # A failing stage stops the run; the outer handler restores + persists.
                    raise RuntimeError(f"stage '{key}' failed: {message}") from exc

                state.updated_at = now()
                if index + 1 < total:
                    await _release_models(job.stages[index + 1])
                job.progress = (index + 1) / total
                job.message = f"{key}: {state.status}"
                await _checkpoint(project)
                engine.bus.publish_job(job)
            job.message = f"completed {total} stage(s)"
            # Every run that completes Mix becomes a browsable version (snapshot of the
            # on-disk state, taken after the last stage so a render is included).
            if "mix" in job.stages and project is not None and project.stage("mix").status == "done":
                job.message = "saving version"
                engine.bus.publish_job(job)
                async with store.lock(project_id):
                    try:
                        await asyncio.shield(versions.snapshot(project_id, kind="auto", job_id=job.id))
                    except versions.VersionError:
                        pass
                fresh = store.load(project_id)
                if fresh is not None:
                    engine.bus.publish_project(fresh)
                job.message = f"completed {total} stage(s)"
        except BaseException as exc:
            # Single cleanup point: a cancel/error landing ANYWHERE in the loop (including the
            # inter-stage checkpoints) must not leave stages stuck on "running"/"queued".
            if project is not None:
                if job.stage is not None:
                    state = project.stage(job.stage)
                    if state.status == "running":
                        state.status = "error"
                        state.detail = (
                            "cancelled"
                            if isinstance(exc, asyncio.CancelledError)
                            else (str(exc) or type(exc).__name__)
                        )
                        state.updated_at = now()
                _restore_queued(project, job.stages, prior)
                # Shield the persist so a second cancel (e.g. engine shutdown while this handler
                # runs) can't skip it — the save completes in the background regardless.
                try:
                    await asyncio.shield(_checkpoint(project))
                except asyncio.CancelledError:
                    if not isinstance(exc, asyncio.CancelledError):
                        raise
            raise

    return run


def _segment_runner(project_id: str, segment_id: str) -> Runner:
    async def run(job: Job) -> None:
        try:
            async with manager.hold_async():
                await _run_segment(job)
        finally:
            # Local models stay warm for the idle TTL (the user often regenerates line after
            # line); LLM servers are released now.
            await _release_models(None, keep=lambda key: family(key) != "llm")

    async def _run_segment(job: Job) -> None:
        project = store.load(project_id, track=True)
        if project is None:
            raise RuntimeError("project was deleted while the job was running")
        segment = project.segment(segment_id)
        if segment is None:
            raise RuntimeError(f"segment '{segment_id}' no longer exists")

        total = len(job.stages)
        completed = 0
        downstream_from: StageKey = "translate" if "transcribe" in job.stages else "synthesize"
        try:
            for index, key in enumerate(job.stages):
                job.stage = key
                job.progress = index / total
                job.message = f"{key} for segment {segment_id}"
                engine.bus.publish_job(job)
                progress = _stage_progress(job, index, total)
                if key == "transcribe":
                    provider = resolve_provider(project, "asr")
                    await stage_impl.transcribe_one(project, provider, segment, progress)
                elif key == "translate":
                    provider = resolve_provider(project, "translation")
                    await stage_impl.translate_one(project, provider, segment, progress)
                else:
                    provider = resolve_provider(project, "tts")
                    await stage_impl.synthesize_one(project, provider, segment, progress)
                completed += 1
                job.progress = (index + 1) / total
                await _checkpoint(project)
                engine.bus.publish_job(job)
        except asyncio.CancelledError:
            if completed:
                project.mark_downstream_dirty(downstream_from)
            # Shielded so a second cancel can't skip persisting what was already produced.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(_checkpoint(project))
            raise
        except Exception:
            if completed:
                project.mark_downstream_dirty(downstream_from)
            await _checkpoint(project)
            raise

        # Regenerated text/audio invalidates everything after it.
        project.mark_downstream_dirty(downstream_from)
        await _checkpoint(project)
        job.message = "segment updated"

    return run


# ---- helpers ---------------------------------------------------------------------------------


async def _release_models(
    next_stage: StageKey | None, *, keep: Callable[[tuple], bool] | None = None
) -> None:
    """Unload idle models. Between stages: those ``next_stage`` does not load (models another
    project's job is using or holding are spared). At job end (``next_stage=None``): everything
    idle, and anything still busy (e.g. an uncancellable worker thread) as soon as it's released.
    Best effort: a model that fails to unload must never fail the job."""
    if next_stage is not None:
        wanted = _STAGE_MODELS.get(next_stage, frozenset())

        def keep(key: tuple) -> bool:
            return family(key) in wanted

    try:
        await asyncio.shield(asyncio.to_thread(
            manager.unload_idle, keep=keep, defer_busy=next_stage is None
        ))
    except asyncio.CancelledError:
        # The shielded unload still completes in its thread. Mid-run the cancel must stop the
        # job; at job end the work is already finished (or already being cancelled).
        if next_stage is not None:
            raise
    except Exception as exc:  # noqa: BLE001
        log.warning("releasing models failed: %s", exc)


async def _checkpoint(project: Project) -> None:
    """Persist and broadcast the project under its lock, re-applying any user edits that
    landed on disk since the job loaded this copy (see store.save_merged)."""
    async with store.lock(project.id):
        await asyncio.to_thread(store.save_merged, project)
    engine.bus.publish_project(project)


def _stage_needs_run(project: Project, key: StageKey) -> bool:
    """Default 'Run pipeline' selection: any non-done stage, plus a 'done' translate/synthesize
    that still owes per-segment work (segment edits set translate_dirty/synth_dirty without
    flipping the stage status itself)."""
    if project.stage(key).status != "done":
        return True
    if key == "translate":
        return bool(stage_impl.translate_targets(project))
    if key == "synthesize":
        return bool(stage_impl.synth_targets(project))
    return False


def _abandoned_cleanup(
    project_id: str, selected: Sequence[StageKey], prior: dict[StageKey, StageStatus]
) -> Callable[[Job], None]:
    """Cleanup run by the engine when the job dies without its runner ever starting
    (cancelled while queued, or engine shutdown): restore the stage statuses that
    start_pipeline_job persisted as 'queued'."""

    def cleanup(job: Job) -> None:
        project = store.load(project_id)
        if project is None:
            return
        before = {k: project.stage(k).status for k in selected}
        _restore_queued(project, selected, prior)
        if any(project.stage(k).status != before[k] for k in selected):
            store.save(project)
            engine.bus.publish_project(project)

    return cleanup


def _stage_progress(job: Job, index: int, total: int) -> ProgressFn:
    """Map a stage's 0..1 progress into overall job progress; publish throttled (~4/sec)."""

    def progress(fraction: float, message: str = "") -> None:
        try:
            f = float(fraction)
        except (TypeError, ValueError):
            f = 0.0
        if math.isnan(f):
            f = 0.0
        f = min(1.0, max(0.0, f))
        job.progress = min((index + f) / max(total, 1), 0.999)
        if message:
            job.message = message
        engine.bus.publish_job(job, throttle=True)

    return progress


def _restore_queued(
    project: Project, keys: Sequence[StageKey], prior: dict[StageKey, StageStatus]
) -> None:
    """Put never-run stages (still 'queued' from submit) back to their pre-submit status."""
    for key in keys:
        state = project.stage(key)
        if state.status != "queued":
            continue
        previous = prior.get(key, "pending")
        state.status = "pending" if previous in ("queued", "running") else previous
        state.updated_at = now()
