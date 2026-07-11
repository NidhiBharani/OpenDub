"""Pipeline orchestrator — turns API requests into jobs on the engine.

``start_pipeline_job`` resolves which stages to run, snapshots + queues their StageStates and
submits a runner that executes them in order: each stage goes running → done/skipped/error with a
detail message and timestamps, the project is saved + broadcast after every transition, and overall
job progress is ``(completed_stages + stage_fraction) / total_stages``.

``start_segment_job`` re-runs translate and/or synthesize for a single segment, then marks
mix/lipsync/render dirty (it does NOT auto-run mix).
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence

from .. import store
from ..jobs import Runner, engine
from ..models import STAGE_ORDER, Job, Project, StageKey, StageStatus, now
from ..providers.base import ProgressFn
from . import stages as stage_impl
from .stages import StageSkipped, resolve_provider

_SEGMENT_STAGES: tuple[StageKey, ...] = ("translate", "synthesize")


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
    if project.segment(segment_id) is None:
        raise KeyError(f"unknown segment '{segment_id}' in project '{project_id}'")

    bad = sorted({s for s in stages if s not in _SEGMENT_STAGES})
    if bad:
        raise ValueError(
            f"segment jobs support stages {list(_SEGMENT_STAGES)}; got: {', '.join(bad)}"
        )
    wanted = set(stages)
    selected: list[StageKey] = [k for k in _SEGMENT_STAGES if k in wanted]
    if not selected:
        raise ValueError("no stages requested; pass 'translate' and/or 'synthesize'")

    job = Job(project_id=project_id, kind="segment", stages=selected, segment_id=segment_id)
    engine.submit(job, _segment_runner(project_id, segment_id))
    return job


# ---- runners ---------------------------------------------------------------------------------


def _pipeline_runner(project_id: str, prior: dict[StageKey, StageStatus]) -> Runner:
    async def run(job: Job) -> None:
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
                    if key == "lipsync" and (
                        project.pipeline.choice("lipsync").provider_id == "lipsync.none"
                    ):
                        raise StageSkipped("lipsync provider is set to 'none'")
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
                job.progress = (index + 1) / total
                job.message = f"{key}: {state.status}"
                await _checkpoint(project)
                engine.bus.publish_job(job)
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
        project = store.load(project_id, track=True)
        if project is None:
            raise RuntimeError("project was deleted while the job was running")
        segment = project.segment(segment_id)
        if segment is None:
            raise RuntimeError(f"segment '{segment_id}' no longer exists")

        total = len(job.stages)
        completed = 0
        try:
            for index, key in enumerate(job.stages):
                job.stage = key
                job.progress = index / total
                job.message = f"{key} for segment {segment_id}"
                engine.bus.publish_job(job)
                progress = _stage_progress(job, index, total)
                if key == "translate":
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
                project.mark_downstream_dirty("synthesize")
            # Shielded so a second cancel can't skip persisting what was already produced.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(_checkpoint(project))
            raise
        except Exception:
            if completed:
                project.mark_downstream_dirty("synthesize")
            await _checkpoint(project)
            raise

        # Regenerated audio invalidates the mixdown and everything after it.
        project.mark_downstream_dirty("synthesize")
        await _checkpoint(project)
        job.message = "segment updated"

    return run


# ---- helpers ---------------------------------------------------------------------------------


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
        return any(s.translate_dirty for s in project.segments)
    if key == "synthesize":
        return any(s.synth_dirty for s in project.segments)
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
        if f != f:  # NaN
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
