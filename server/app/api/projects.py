"""Project CRUD + pipeline-run routes.

Mounted under /api by main.py. See ARCHITECTURE.md "HTTP API" and specs/backend-core.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .. import config, store
from ..jobs import engine
from ..models import (
    Job,
    Project,
    ProjectSummary,
    ProviderChoice,
    ProviderKind,
    StageKey,
    now,
)
from ..pipeline import orchestrator

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}

# kind -> stage that consumes that provider kind (ARCHITECTURE.md pipeline table)
KIND_STAGE: dict[str, StageKey] = {
    "separation": "separate",
    "asr": "transcribe",
    "diarization": "transcribe",
    "translation": "translate",
    "tts": "synthesize",
    "lipsync": "lipsync",
}

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class ProjectPatchBody(BaseModel):
    name: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    pipeline: dict[ProviderKind, ProviderChoice] | None = None


class PipelineRunBody(BaseModel):
    stages: list[StageKey] | None = None


def _load_or_404(pid: str) -> Project:
    project = store.load(pid)
    if project is None:
        raise HTTPException(404, f"project '{pid}' not found")
    return project


@router.get("/projects")
async def list_projects() -> list[ProjectSummary]:
    return store.list_summaries()


@router.post("/projects")
async def create_project(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    source_lang: str = Form("ja"),
    target_lang: str = Form("en"),
) -> Project:
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"unsupported file extension '{ext or '(none)'}'; expected one of "
            f"{sorted(ALLOWED_EXTENSIONS)}",
        )

    stem = Path(filename).stem
    project_name = (name or "").strip() or stem or "Untitled"
    project = Project(name=project_name, source_lang=source_lang, target_lang=target_lang)

    # Apply any global default pipeline (configs/settings.yaml -> defaults: {kind: provider_id}).
    defaults: dict[str, Any] = config.load_settings().get("defaults") or {}
    for kind, provider_id in defaults.items():
        if hasattr(project.pipeline, kind) and isinstance(provider_id, str) and provider_id:
            setattr(project.pipeline, kind, ProviderChoice(provider_id=provider_id))

    dest_dir = store.project_dir(project.id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"source{ext}"
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
    finally:
        await file.close()

    project.source_filename = filename

    store.save(project)
    orchestrator.start_pipeline_job(project.id, ["ingest"])
    return project


@router.get("/projects/{pid}")
async def get_project(pid: str) -> Project:
    return _load_or_404(pid)


@router.delete("/projects/{pid}")
async def delete_project(pid: str) -> dict[str, bool]:
    _load_or_404(pid)
    for job in engine.list(pid):
        if job.status in ("queued", "running"):
            await engine.cancel(job.id)
    ok = store.delete(pid)
    return {"ok": ok}


@router.patch("/projects/{pid}")
async def update_project(pid: str, body: ProjectPatchBody) -> Project:
    project = _load_or_404(pid)

    if body.name is not None:
        project.name = body.name
    if body.source_lang is not None:
        project.source_lang = body.source_lang
    if body.target_lang is not None:
        project.target_lang = body.target_lang

    if body.pipeline:
        for kind, choice in body.pipeline.items():
            current = project.pipeline.choice(kind)
            if current.provider_id == choice.provider_id and current.options == choice.options:
                continue
            setattr(project.pipeline, kind, choice)
            stage_key = KIND_STAGE[kind]
            stage = project.stage(stage_key)
            if stage.status == "done":
                stage.status = "dirty"
                stage.updated_at = now()
            project.mark_downstream_dirty(stage_key)

    store.save(project)
    engine.bus.publish_project(project)
    return project


@router.post("/projects/{pid}/pipeline/run")
async def run_pipeline(pid: str, body: PipelineRunBody) -> Job:
    _load_or_404(pid)
    try:
        return orchestrator.start_pipeline_job(pid, body.stages)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
