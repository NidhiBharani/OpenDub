"""Benchmark runner: drive the real pipeline over a case, capture timing + peak VRAM, then
compute every metric module against the finished project.

The pipeline runs IN-PROCESS via the app's ASGI transport (same pattern as tests) so the bench
needs no separate server and uses the real configured providers from configs/settings.yaml. A
background thread samples nvidia-smi so per-stage peak VRAM can be attributed by timestamp.
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from app import store
from app.main import app
from app.models import STAGE_ORDER, Project

from .cases import BenchCase
from .metrics import MODULES
from .metrics.base import Metric, MetricContext


@dataclass
class CaseResult:
    case: str
    project_id: str
    metrics: list[Metric] = field(default_factory=list)
    error: str | None = None


class _VramSampler:
    """Samples total GPU memory (MiB) on a thread; supports peak-in-window queries."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._samples: list[tuple[float, float]] = []  # (monotonic-ish wall time, MiB)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read() -> float:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return float(out.stdout.strip().splitlines()[0])
        except Exception:
            return 0.0

    def _loop(self):
        while not self._stop.is_set():
            self._samples.append((time.time(), self._read()))
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def peak_between(self, start: float, end: float) -> float:
        vals = [v for (t, v) in self._samples if start <= t <= end]
        return max(vals) if vals else 0.0


def _as_epoch(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


async def _wait_for_job(client: httpx.AsyncClient, pid: str, job_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = (await client.get(f"/api/jobs?project_id={pid}")).json()
        job = next((j for j in jobs if j["id"] == job_id), None)
        if job and job["status"] in ("done", "error", "cancelled"):
            return job
        await asyncio.sleep(1.0)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def _stage_windows(project: Project, run_start: float) -> dict[str, tuple[float, float]]:
    """Approximate [start, end] wall-clock window per stage from StageState.updated_at.

    updated_at is the stage's completion time; a stage's start is the prior stage's completion
    (or run_start for the first). Good enough to attribute peak VRAM to a stage."""
    windows: dict[str, tuple[float, float]] = {}
    prev_end = run_start
    for stage in STAGE_ORDER:
        st = project.stages.get(stage)
        end = _as_epoch(st.updated_at) if st else None
        if end is None or st.status in ("skipped", "pending"):
            continue
        windows[stage] = (prev_end, end)
        prev_end = end
    return windows


async def run_case(case: BenchCase, timeout: float = 1800.0) -> CaseResult:
    sampler = _VramSampler()
    sampler.start()
    run_start = time.time()
    transport = httpx.ASGITransport(app=app)
    try:
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=transport, base_url="http://bench"
        ) as client:
            # 1. create project (auto-runs ingest)
            with case.source.open("rb") as f:
                resp = await client.post(
                    "/api/projects",
                    files={"file": (case.source.name, f, "video/mp4")},
                    data={"name": f"bench:{case.name}", "source_lang": case.source_lang,
                          "target_lang": case.target_lang},
                )
            resp.raise_for_status()
            project_json = resp.json()
            pid = project_json["id"]

            # 2. wait out the auto-started ingest job, then run the rest
            for j in (await client.get(f"/api/jobs?project_id={pid}")).json():
                await _wait_for_job(client, pid, j["id"], timeout)
            run = (await client.post(f"/api/projects/{pid}/pipeline/run", json={})).json()
            final = await _wait_for_job(client, pid, run["id"], timeout)

            project = store.load(pid)
            sampler.stop()
            if project is None:
                return CaseResult(case.name, pid, error="project vanished after run")

            windows = _stage_windows(project, run_start)
            stage_timing = {
                s: {"seconds": e - b, "peak_vram_mib": sampler.peak_between(b, e)}
                for s, (b, e) in windows.items()
            }
            ctx = MetricContext(
                project=project, project_dir=store.project_dir(pid),
                case=case, stage_timing=stage_timing,
            )
            metrics = await _score_all(ctx)
            err = final.get("error") if final["status"] == "error" else None
            return CaseResult(case.name, pid, metrics=metrics, error=err)
    finally:
        sampler.stop()


async def _score_all(ctx: MetricContext) -> list[Metric]:
    metrics: list[Metric] = []
    for label, module in MODULES:
        try:
            metrics.extend(await module.score(ctx))
        except Exception as e:  # a metric module must never abort the whole run
            metrics.append(Metric(f"{label}.error", None, note=f"metric module crashed: {e!r}"))
    return metrics
