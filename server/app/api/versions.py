"""Version routes: list / snapshot / rename / delete / restore dubbing attempts.

See app/pipeline/versions.py for the on-disk layout. Structural operations (snapshot, delete,
restore) are refused while a job runs for the project: the job's checkpoint merge would fight
them (ARCHITECTURE.md "Dirty tracking").
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import store
from ..jobs import engine
from ..models import Project, Version
from ..pipeline import versions

router = APIRouter()


class SnapshotBody(BaseModel):
    label: str = ""


class RenameBody(BaseModel):
    label: str


class RestoreBody(BaseModel):
    restore_point: bool = True


def _load_or_404(pid: str) -> Project:
    project = store.load(pid)
    if project is None:
        raise HTTPException(404, f"project '{pid}' not found")
    return project


def _refuse_if_busy(pid: str) -> None:
    job = engine.active(pid)
    if job is not None:
        raise HTTPException(409, f"a job is {job.status} for this project; wait for it or cancel it")


@router.get("/projects/{pid}/versions")
async def list_versions(pid: str, lang: str | None = None) -> list[Version]:
    _load_or_404(pid)
    out = versions.list_versions(pid)
    if lang:
        out = [v for v in out if v.target_lang == lang]
    return out


@router.post("/projects/{pid}/versions")
async def snapshot_version(pid: str, body: SnapshotBody | None = None) -> Version:
    _load_or_404(pid)
    _refuse_if_busy(pid)
    async with store.lock(pid):
        try:
            version = await versions.snapshot(pid, kind="manual", label=(body.label if body else ""))
        except versions.VersionError as e:
            raise HTTPException(400, str(e)) from e
        project = store.load(pid)
    if project is not None:
        engine.bus.publish_project(project)
    return version


@router.patch("/projects/{pid}/versions/{vid}")
async def rename_version(pid: str, vid: str, body: RenameBody) -> Version:
    _load_or_404(pid)
    try:
        return versions.rename(pid, vid, body.label)
    except versions.VersionError as e:
        raise HTTPException(404, str(e)) from e


@router.delete("/projects/{pid}/versions/{vid}")
async def delete_version(pid: str, vid: str) -> dict[str, bool]:
    _refuse_if_busy(pid)
    async with store.lock(pid):
        project = _load_or_404(pid)
        ok = versions.delete(pid, vid)
        if not ok:
            raise HTTPException(404, f"version '{vid}' not found")
        if project.active_version_id == vid:
            project.active_version_id = None
            store.save(project)
    engine.bus.publish_project(project)
    return {"ok": True}


@router.post("/projects/{pid}/versions/{vid}/restore")
async def restore_version(pid: str, vid: str, body: RestoreBody | None = None) -> Project:
    _load_or_404(pid)
    _refuse_if_busy(pid)
    async with store.lock(pid):
        try:
            project = await versions.restore(
                pid, vid, restore_point=body.restore_point if body else True
            )
        except versions.VersionError as e:
            raise HTTPException(404, str(e)) from e
    engine.bus.publish_project(project)
    return project
