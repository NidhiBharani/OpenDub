"""Project CRUD + pipeline-run routes.

Mounted under /api by main.py. See ARCHITECTURE.md "HTTP API".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .. import capabilities, config, store
from ..jobs import engine
from ..models import (
    LEGACY_KINDS,
    CapabilityChoice,
    Job,
    Project,
    ProjectSummary,
    ProviderChoice,
    StageKey,
    TimeRange,
    now,
)
from ..pipeline import analysis, orchestrator, regions, versions

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
    pipeline: dict[str, ProviderChoice] | None = None  # legacy kinds: separation, asr, …
    mode: Literal["simple", "advanced"] | None = None
    # keyed by capability id; each value is a partial CapabilityChoice merged over the current one
    capabilities: dict[str, dict[str, Any]] | None = None


class PresetBody(BaseModel):
    preset: Literal["minimal", "balanced", "max"]
    runtime: Literal["builtin", "local", "cloud"]
    lipsync: bool | None = None
    subtitles: bool | None = None


class PresetResult(BaseModel):
    project: Project
    notes: list[str]


class PipelineRunBody(BaseModel):
    stages: list[StageKey] | None = None


class SkipRangeIn(BaseModel):
    id: str | None = None
    start: float
    end: float
    label: str = ""


class SkipRangesBody(BaseModel):
    ranges: list[SkipRangeIn]


def _refuse_if_busy(pid: str) -> None:
    job = engine.active(pid)
    if job is not None:
        raise HTTPException(409, f"a job is {job.status} for this project; wait for it or cancel it")


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
        if kind in LEGACY_KINDS and isinstance(provider_id, str) and provider_id:
            setattr(project.pipeline, kind, ProviderChoice(provider_id=provider_id))
            if kind == "lipsync":
                project.pipeline.lipsync_enabled = provider_id != "lipsync.none"

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
    async with store.lock(pid):
        project = _load_or_404(pid)

        if body.name is not None:
            project.name = body.name
        if body.source_lang is not None and body.source_lang != project.source_lang:
            project.source_lang = body.source_lang
            project.active_version_id = None
        if body.target_lang is not None and body.target_lang != project.target_lang:
            # A new target language starts a new dub lineage: keep the old one as a version
            # (when it was never snapshotted) and re-translate every line.
            _refuse_if_busy(pid)
            if project.active_version_id is None and store.resolve(project, "audio/dub_mix.wav").exists():
                try:
                    await versions.snapshot(
                        pid, kind="auto", label=f"Before switching to {body.target_lang.upper()}"
                    )
                except versions.VersionError:
                    pass
                project = _load_or_404(pid)
            project.target_lang = body.target_lang
            project.active_version_id = None
            for seg in project.segments:
                if not seg.skipped and seg.source_text.strip():
                    seg.translate_dirty = True
                    seg.synth_dirty = True  # even an identical line must be voiced in the new language
            _mark_config_changed(project, "translate", "")

        if body.mode is not None:
            project.pipeline.mode = body.mode
        if body.pipeline or body.capabilities:
            project.active_version_id = None

        if body.pipeline:
            bad = sorted(set(body.pipeline) - set(LEGACY_KINDS))
            if bad:
                raise HTTPException(
                    422, f"unknown provider kind(s) {bad}; use 'capabilities' for the rest"
                )
            for kind, choice in body.pipeline.items():
                current = project.pipeline.choice(kind)
                if current.provider_id == choice.provider_id and current.options == choice.options:
                    continue
                setattr(project.pipeline, kind, choice)
                if kind == "lipsync":
                    project.pipeline.lipsync_enabled = choice.provider_id != "lipsync.none"
                _mark_config_changed(project, KIND_STAGE[kind], kind)
                project.pipeline.preset = "custom"
                project.pipeline.runtime = "custom"

        if body.capabilities:
            bad = sorted(set(body.capabilities) - set(capabilities.BY_ID))
            if bad:
                raise HTTPException(422, f"unknown capability id(s) {bad}")
            resolved = capabilities.resolve(project.pipeline)
            for cap_id, patch in body.capabilities.items():
                cap = capabilities.BY_ID[cap_id]
                current = project.pipeline.capabilities.get(cap_id) or CapabilityChoice(
                    enabled=resolved[cap_id].enabled, provider_id=resolved[cap_id].provider_id
                )
                try:
                    choice = CapabilityChoice.model_validate({**current.model_dump(), **patch})
                except ValueError as e:
                    raise HTTPException(422, f"invalid choice for {cap_id}: {e}") from e
                if project.pipeline.capabilities.get(cap_id) == choice:
                    continue
                project.pipeline.capabilities[cap_id] = choice
                _mark_config_changed(project, cap.stage, cap.kind)
                project.pipeline.preset = "custom"

        store.save(project)
    engine.bus.publish_project(project)
    return project


def _mark_config_changed(project: Project, stage_key: StageKey, kind: str) -> None:
    stage = project.stage(stage_key)
    if stage.status in ("done", "skipped"):
        stage.status = "dirty"
        stage.updated_at = now()
    project.mark_downstream_dirty(stage_key)
    # translate/synthesize only process segments carrying their dirty flag, so a provider
    # change must also flag every segment — otherwise the re-run is a no-op that flips the
    # stage back to "done" without re-processing anything.
    if kind == "translation":
        for seg in project.segments:
            seg.translate_dirty = True
    elif kind == "tts":
        for seg in project.segments:
            seg.synth_dirty = True


@router.post("/projects/{pid}/pipeline/preset")
async def apply_preset(pid: str, body: PresetBody) -> PresetResult:
    """Simple mode: write concrete provider choices for every capability from a preset + runtime."""
    async with store.lock(pid):
        project = _load_or_404(pid)
        before = project.pipeline.model_dump()
        notes = capabilities.apply_preset(
            project.pipeline, body.preset, body.runtime, body.lipsync, body.subtitles
        )
        project.pipeline.mode = "simple"
        after = project.pipeline.model_dump()
        for cap in capabilities.CAPABILITIES:
            field = cap.legacy_field
            changed = (
                before.get(field) != after.get(field)
                if field
                else before["capabilities"].get(cap.id) != after["capabilities"].get(cap.id)
            )
            if changed and project.stage(cap.stage).status != "pending":
                _mark_config_changed(project, cap.stage, cap.kind)
        store.save(project)
    engine.bus.publish_project(project)
    return PresetResult(project=project, notes=notes)


@router.put("/projects/{pid}/skip-ranges")
async def put_skip_ranges(pid: str, body: SkipRangesBody) -> Project:
    """Replace the skip ranges (spans that keep the original audio/picture). Lines inside a
    range stop being translated/voiced; the mix and everything after it must re-run."""
    _refuse_if_busy(pid)
    async with store.lock(pid):
        project = _load_or_404(pid)
        existing = {r.id: r for r in project.skip_ranges}
        incoming: list[TimeRange] = []
        for r in body.ranges:
            if r.end <= r.start:
                raise HTTPException(400, f"range end ({r.end}) must be after start ({r.start})")
            prev = existing.get(r.id or "")
            incoming.append(
                TimeRange(
                    id=prev.id if prev else TimeRange(start=r.start, end=r.end).id,
                    start=r.start,
                    end=r.end,
                    label=r.label,
                    source=prev.source if prev else "user",
                )
            )
        duration = project.media.duration if project.media else None
        new_ranges = regions.normalize_ranges(incoming, duration)
        changed = [r.model_dump() for r in new_ranges] != [r.model_dump() for r in project.skip_ranges]
        project.skip_ranges = new_ranges
        _newly_skipped, newly_unskipped = regions.apply_skip_ranges(project)
        for seg in project.segments:
            if seg.skipped:
                seg.translate_dirty = False
                seg.synth_dirty = False
            elif seg.id in newly_unskipped:
                seg.translate_dirty = not seg.translated_text.strip()
                seg.synth_dirty = not seg.takes
        if changed:
            project.mark_downstream_dirty("synthesize")
            project.active_version_id = None
        store.save(project)
    engine.bus.publish_project(project)
    return project


@router.get("/projects/{pid}/anchors")
async def get_anchors(pid: str, refresh: bool = False) -> analysis.AnchorSet:
    """Suggested cut points: shot changes, dialogue silences and segment edges. Cached; pass
    refresh=1 to run the detectors again."""
    project = _load_or_404(pid)
    if project.media is None:
        raise HTTPException(409, "run ingest first")
    try:
        return await analysis.anchors(project, refresh=refresh)
    except RuntimeError as e:
        raise HTTPException(500, f"analysis failed: {e}") from e


@router.get("/projects/{pid}/filmstrip")
async def get_filmstrip(pid: str, refresh: bool = False) -> analysis.Filmstrip:
    project = _load_or_404(pid)
    try:
        strip = await analysis.filmstrip(project, refresh=refresh)
    except RuntimeError as e:
        raise HTTPException(500, f"filmstrip failed: {e}") from e
    if strip is None:
        raise HTTPException(409, "run ingest first")
    return strip


@router.post("/projects/{pid}/pipeline/run")
async def run_pipeline(pid: str, body: PipelineRunBody) -> Job:
    _load_or_404(pid)
    try:
        return orchestrator.start_pipeline_job(pid, body.stages)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
