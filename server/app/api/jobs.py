"""Job listing + cancellation routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..jobs import engine
from ..models import Job

router = APIRouter()


@router.get("/jobs")
async def list_jobs(project_id: str | None = None) -> list[Job]:
    return engine.list(project_id)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Job:
    job = await engine.cancel(job_id)
    if job is None:
        raise HTTPException(404, f"job '{job_id}' not found")
    return job
