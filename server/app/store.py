"""Project persistence. One directory per project under data/projects/<id>/ containing all media
plus manifest.json (the serialized Project). Only this module deals in absolute paths; every path
stored on models is relative to the project dir.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

from . import config
from .models import Project, ProjectSummary

_locks: dict[str, asyncio.Lock] = {}


def lock(project_id: str) -> asyncio.Lock:
    return _locks.setdefault(project_id, asyncio.Lock())


def project_dir(project_id: str) -> Path:
    d = config.PROJECTS_DIR / project_id
    if not d.resolve().is_relative_to(config.PROJECTS_DIR.resolve()):
        raise ValueError("invalid project id")
    return d


def resolve(project: Project | str, rel_path: str) -> Path:
    """Absolute path for a project-relative media path (validated against traversal)."""
    pid = project if isinstance(project, str) else project.id
    base = project_dir(pid).resolve()
    p = (base / rel_path).resolve()
    if not p.is_relative_to(base):
        raise ValueError("invalid media path")
    return p


def save(project: Project) -> None:
    d = project_dir(project.id)
    d.mkdir(parents=True, exist_ok=True)
    # Unique tmp name per save: concurrent saves (route on the loop thread vs. a job checkpoint
    # in a worker thread) can never interleave writes to the same tmp file; each replace is atomic.
    tmp = d / f"manifest.json.{uuid.uuid4().hex[:8]}.tmp"
    try:
        tmp.write_text(project.model_dump_json(indent=2))
        tmp.replace(d / "manifest.json")
    finally:
        tmp.unlink(missing_ok=True)


def load(project_id: str, *, track: bool = False) -> Project | None:
    """Load a project. With ``track=True`` (used by job runners) the loaded state is snapshotted
    so a later ``save_merged`` can detect and preserve user edits made on disk in the meantime."""
    manifest = project_dir(project_id) / "manifest.json"
    if not manifest.exists():
        return None
    project = Project.model_validate(json.loads(manifest.read_text()))
    if track:
        project._checkpoint_base = project.model_dump(mode="json")
    return project


# Segment fields owned by user edits (PATCH /segments); job-generated data (takes) is excluded.
_USER_SEGMENT_FIELDS = (
    "start",
    "end",
    "speaker_id",
    "source_text",
    "translated_text",
    "emotion",
    "notes",
    "translate_dirty",
    "synth_dirty",
)


def save_merged(project: Project) -> None:
    """Persist a job's (potentially stale) project copy without clobbering user edits.

    Jobs hold a Project in memory for a whole stage while routes keep editing the on-disk
    manifest. Before saving, reload the manifest and re-apply anything that changed on disk
    since this copy's snapshot (segment text/timing/flags, speaker names, project settings);
    the job keeps ownership of stage-state transitions and the takes it generated. Call under
    ``store.lock(project.id)``. Falls back to a plain save when no snapshot is tracked.
    """
    base = project._checkpoint_base
    if base is not None:
        fresh = load(project.id)
        if fresh is not None:
            _reapply_user_edits(project, base, fresh)
    save(project)
    project._checkpoint_base = project.model_dump(mode="json")


def _reapply_user_edits(ours: Project, base: dict, fresh: Project) -> None:
    # Project-level fields jobs never write: the on-disk value is authoritative.
    ours.name = fresh.name
    ours.source_lang = fresh.source_lang
    ours.target_lang = fresh.target_lang
    ours.pipeline = fresh.pipeline

    base_segments = {s["id"]: s for s in base.get("segments", [])}
    for seg in ours.segments:
        b = base_segments.get(seg.id)
        f = fresh.segment(seg.id)
        if b is None or f is None:
            continue  # segment created/replaced by the job itself: ours wins
        if f.model_dump(mode="json") == b:
            continue  # untouched on disk since our snapshot: ours wins
        # The user edited this segment while the job ran: their editable fields and dirty
        # flags win over the job's stale copy (so e.g. an edit made mid-synthesis keeps its
        # translate/synth_dirty and is re-processed); the job's generated takes are kept.
        for field in _USER_SEGMENT_FIELDS:
            setattr(seg, field, getattr(f, field))
        if f.active_take_id != b.get("active_take_id"):
            seg.active_take_id = f.active_take_id

    base_speakers = {s["id"]: s for s in base.get("speakers", [])}
    for spk in ours.speakers:
        b = base_speakers.get(spk.id)
        f = fresh.speaker(spk.id)
        if b is None or f is None:
            continue
        if f.model_dump(mode="json") != b:
            spk.name = f.name
            spk.color = f.color

    # Stage states: the job wins for stages it transitioned; otherwise on-disk changes
    # (e.g. a route marking downstream stages dirty) win.
    base_stages = base.get("stages", {})
    for key, f_state in fresh.stages.items():
        ours_state = ours.stages.get(key)
        b = base_stages.get(key)
        if ours_state is None:
            ours.stages[key] = f_state
            continue
        if b is not None and ours_state.model_dump(mode="json") != b:
            continue  # job changed this stage
        if b is None or f_state.model_dump(mode="json") != b:
            ours.stages[key] = f_state


def delete(project_id: str) -> bool:
    d = project_dir(project_id)
    if not d.exists():
        return False
    shutil.rmtree(d)
    _locks.pop(project_id, None)
    return True


def list_summaries() -> list[ProjectSummary]:
    out: list[ProjectSummary] = []
    if not config.PROJECTS_DIR.exists():
        return out
    for d in sorted(config.PROJECTS_DIR.iterdir()):
        if (d / "manifest.json").exists():
            try:
                p = Project.model_validate(json.loads((d / "manifest.json").read_text()))
            except Exception:
                continue
            out.append(
                ProjectSummary(
                    id=p.id,
                    name=p.name,
                    created_at=p.created_at,
                    duration=p.media.duration if p.media else 0.0,
                    target_lang=p.target_lang,
                    segment_count=len(p.segments),
                    stages=p.stages,
                )
            )
    out.sort(key=lambda s: s.created_at, reverse=True)
    return out
