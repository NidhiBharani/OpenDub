"""Async job engine + event bus.

The engine serializes jobs PER PROJECT (one running job per project at a time; different projects
run concurrently), supports cancellation of queued and running jobs, and keeps a bounded history of
finished jobs. The bus fans job/project updates out to per-project SSE subscribers through bounded
queues (drop-oldest) so a stalled client can never block a job.

Cancellation semantics: cancelling a running job cancels its runner task; awaited subprocesses
(ffmpeg, demucs, wav2lip, latentsync) kill their child process on CancelledError, but work inside
``asyncio.to_thread`` (whisper/f5-tts/pyannote inference) cannot be interrupted — the thread runs to
completion in the background. Such abandoned work never persists anything (only the cancelled
runner saved manifests, and provider temp filenames are unique per call), and its late progress
callbacks are dropped by ``publish_job`` once the job is finished.

This module is one of the two allowed global-mutable-state singletons (see ARCHITECTURE.md).
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable

from .models import Job, Project, now

Runner = Callable[[Job], Awaitable[None]]

_SUB_QUEUE_MAX = 256  # per-subscriber event buffer; oldest events dropped when a client stalls
_FINISHED_KEEP = 50  # finished jobs kept in history
_PUBLISH_MIN_INTERVAL = 0.25  # throttled publishes: at most ~4/sec per job

_FINISHED_STATUSES = ("done", "error", "cancelled")


class EventBus:
    """Fan-out of ``(event_name, json_payload)`` tuples to per-project subscribers.

    Safe to publish from worker threads (e.g. provider progress callbacks running inside
    ``asyncio.to_thread``): payloads are serialized in the calling thread and handed to the
    event loop via ``call_soon_threadsafe``.
    """

    def __init__(self) -> None:
        self._subs: dict[str, dict[str, asyncio.Queue[tuple[str, str]]]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_pub: dict[str, float] = {}  # job_id -> monotonic time of last throttled publish

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # -- subscriptions ---------------------------------------------------------------------

    def subscribe(self, project_id: str) -> tuple[str, asyncio.Queue]:
        """Register an SSE listener. Returns ``(sub_id, queue)`` of ``(event_name, json_str)``."""
        sub_id = uuid.uuid4().hex[:8]
        queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=_SUB_QUEUE_MAX)
        self._subs.setdefault(project_id, {})[sub_id] = queue
        return sub_id, queue

    def unsubscribe(self, project_id: str, sub_id: str) -> None:
        subs = self._subs.get(project_id)
        if subs is None:
            return
        subs.pop(sub_id, None)
        if not subs:
            self._subs.pop(project_id, None)

    # -- publishing ------------------------------------------------------------------------

    def publish_job(self, job: Job, *, throttle: bool = False) -> None:
        """Emit a ``job`` event. With ``throttle=True`` rate-limit to ~4/sec per job
        (used by high-frequency progress callbacks; state transitions publish unthrottled)."""
        if throttle:
            if job.status in _FINISHED_STATUSES:
                # Orphaned progress callback from abandoned (uncancellable) work: don't emit
                # events for a finished job or resurrect its throttling entry after forget().
                return
            if not self._should_publish(job.id):
                return
        self._dispatch(job.project_id, "job", job.model_dump_json())

    def publish_project(self, project: Project) -> None:
        """Emit a ``project`` event (full project JSON)."""
        self._dispatch(project.id, "project", project.model_dump_json())

    def forget(self, job_id: str) -> None:
        """Drop throttling state for a finished/pruned job."""
        self._last_pub.pop(job_id, None)

    # -- internals ---------------------------------------------------------------------------

    def _should_publish(self, key: str) -> bool:
        t = time.monotonic()
        if t - self._last_pub.get(key, 0.0) < _PUBLISH_MIN_INTERVAL:
            return False
        self._last_pub[key] = t
        return True

    def _dispatch(self, project_id: str, event: str, payload: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Called from a worker thread — hop onto the server loop.
            loop = self._loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(self._dispatch_now, project_id, event, payload)
            return
        self._dispatch_now(project_id, event, payload)

    def _dispatch_now(self, project_id: str, event: str, payload: str) -> None:
        item = (event, payload)
        for queue in list((self._subs.get(project_id) or {}).values()):
            while True:
                try:
                    queue.put_nowait(item)
                    break
                except asyncio.QueueFull:
                    # Bounded queue: drop the oldest event so a stalled client can't block us.
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass  # a consumer raced us; retry the put


class JobEngine:
    """Runs submitted jobs, one at a time per project, on lazily-created worker tasks."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self._jobs: dict[str, Job] = {}  # insertion order == creation order
        self._runners: dict[str, Runner] = {}
        self._cleanups: dict[str, Callable[[Job], None]] = {}  # run if the job never starts
        self._queues: dict[str, deque[str]] = {}  # project_id -> pending job ids
        self._workers: dict[str, asyncio.Task] = {}  # project_id -> drain task
        self._running: dict[str, asyncio.Task] = {}  # job_id -> runner task
        self._started = False

    # -- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        """Called from the app lifespan. Worker tasks are created lazily on submit."""
        self.bus.attach_loop(asyncio.get_running_loop())
        self._started = True

    async def shutdown(self) -> None:
        self._started = False
        for task in list(self._running.values()):
            task.cancel()
        workers = [w for w in self._workers.values() if not w.done()]
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        leftovers = [t for t in self._running.values() if not t.done()]
        for task in leftovers:
            task.cancel()
        if leftovers:
            await asyncio.gather(*leftovers, return_exceptions=True)
        for job in list(self._jobs.values()):
            if job.status in ("queued", "running"):
                never_ran = job.status == "queued"
                job.status = "cancelled"
                job.message = "server shutting down"
                job.finished_at = now()
                if never_ran:
                    self._run_cleanup(job)
                self.bus.publish_job(job)
                self.bus.forget(job.id)
        self._queues.clear()
        self._workers.clear()
        self._running.clear()

    # -- public API --------------------------------------------------------------------------

    def submit(
        self, job: Job, runner: Runner, *, cleanup: Callable[[Job], None] | None = None
    ) -> Job:
        """Queue a job. Runners execute sequentially per project; projects run concurrently.
        ``cleanup`` (optional) runs if the job dies without its runner ever starting —
        cancelled while still queued, or discarded at shutdown."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:  # pragma: no cover - programming error
            raise RuntimeError(
                "JobEngine.submit must be called from within a running event loop"
            ) from exc
        self.bus.attach_loop(loop)
        if job.id in self._jobs:
            raise ValueError(f"job '{job.id}' was already submitted")
        job.status = "queued"
        self._jobs[job.id] = job
        self._runners[job.id] = runner
        if cleanup is not None:
            self._cleanups[job.id] = cleanup
        self._queues.setdefault(job.project_id, deque()).append(job.id)
        self.bus.publish_job(job)
        worker = self._workers.get(job.project_id)
        if worker is None or worker.done():
            self._workers[job.project_id] = loop.create_task(
                self._drain(job.project_id), name=f"opendub-jobs-{job.project_id}"
            )
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, project_id: str | None = None) -> list[Job]:
        """All known jobs (active + bounded finished history), newest first."""
        jobs = [j for j in self._jobs.values() if project_id is None or j.project_id == project_id]
        jobs.reverse()
        return jobs

    async def cancel(self, job_id: str) -> Job | None:
        """Cancel a queued or running job. Finished jobs are returned unchanged."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status == "queued":
            job.status = "cancelled"
            job.message = "cancelled"
            job.finished_at = now()
            self._run_cleanup(job)  # runner never ran: restore whatever submit persisted
            self.bus.publish_job(job)
            self._finalize(job)
        elif job.status == "running":
            task = self._running.get(job_id)
            if task is not None and not task.done():
                task.cancel()
                await asyncio.wait([task])
                # The project worker (which awaits the same task) sets the final status;
                # give it a moment to settle so callers see the terminal state.
                for _ in range(50):
                    if job.status != "running":
                        break
                    await asyncio.sleep(0.01)
        return job

    # -- internals ---------------------------------------------------------------------------

    async def _drain(self, project_id: str) -> None:
        """Per-project worker: run queued jobs one at a time until the queue is empty."""
        try:
            while True:
                queue = self._queues.get(project_id)
                if not queue:
                    break
                job_id = queue.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue  # cancelled (or pruned) while waiting in the queue
                await self._run_job(job)
        finally:
            self._workers.pop(project_id, None)
            queue = self._queues.get(project_id)
            if queue is not None and not queue:
                self._queues.pop(project_id, None)

    async def _run_job(self, job: Job) -> None:
        runner = self._runners.get(job.id)
        self._cleanups.pop(job.id, None)  # runner is starting: it owns cleanup from here on
        job.status = "running"
        job.started_at = now()
        self.bus.publish_job(job)
        if runner is None:  # pragma: no cover - defensive
            job.status = "error"
            job.error = "internal error: job runner is missing"
            job.finished_at = now()
            self.bus.publish_job(job)
            self._finalize(job)
            return
        task = asyncio.ensure_future(runner(job))
        self._running[job.id] = task
        try:
            # asyncio.wait never propagates the task's exception — inspect it below.
            await asyncio.wait([task])
        except asyncio.CancelledError:
            # The worker itself was cancelled (engine shutdown): cancel the runner too.
            task.cancel()
            try:
                await asyncio.wait([task])
            except asyncio.CancelledError:
                pass
            job.status = "cancelled"
            job.message = "cancelled"
            job.finished_at = now()
            self.bus.publish_job(job)
            self._finalize(job)
            raise
        finally:
            self._running.pop(job.id, None)
        if task.cancelled():
            job.status = "cancelled"
            job.message = "cancelled"
        else:
            exc = task.exception()
            if exc is not None:
                job.status = "error"
                job.error = str(exc) or type(exc).__name__
            else:
                job.status = "done"
                job.progress = 1.0
        job.finished_at = now()
        self.bus.publish_job(job)
        self._finalize(job)

    def _run_cleanup(self, job: Job) -> None:
        """Best-effort cleanup for a job whose runner never started."""
        cleanup = self._cleanups.pop(job.id, None)
        if cleanup is None:
            return
        try:
            cleanup(job)
        except Exception:  # pragma: no cover - cleanup must never break engine state
            pass

    def _finalize(self, job: Job) -> None:
        self._cleanups.pop(job.id, None)
        self.bus.forget(job.id)
        self._prune()

    def _prune(self) -> None:
        """Keep only the newest ~50 finished jobs; never drop queued/running ones."""
        finished = [jid for jid, j in self._jobs.items() if j.status in _FINISHED_STATUSES]
        for jid in finished[: max(0, len(finished) - _FINISHED_KEEP)]:
            self._jobs.pop(jid, None)
            self._runners.pop(jid, None)
            self._cleanups.pop(jid, None)
            self.bus.forget(jid)


engine = JobEngine()
