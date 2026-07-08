"""Segment + speaker mutation routes (dirty-flag propagation lives here per specs/backend-core.md
"Dirty rules").
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import store
from ..jobs import engine
from ..models import Job, Project, Segment, Speaker
from ..pipeline import orchestrator

router = APIRouter()


class SegmentPatchBody(BaseModel):
    source_text: str | None = None
    translated_text: str | None = None
    speaker_id: str | None = None
    start: float | None = None
    end: float | None = None
    emotion: str | None = None
    active_take_id: str | None = None
    notes: str | None = None


class RegenerateBody(BaseModel):
    stages: list[Literal["translate", "synthesize"]]


class SpeakerPatchBody(BaseModel):
    name: str | None = None
    color: str | None = None


def _load_project_or_404(pid: str) -> Project:
    project = store.load(pid)
    if project is None:
        raise HTTPException(404, f"project '{pid}' not found")
    return project


def _load_segment_or_404(project: Project, sid: str) -> Segment:
    segment = project.segment(sid)
    if segment is None:
        raise HTTPException(404, f"segment '{sid}' not found in project '{project.id}'")
    return segment


@router.patch("/projects/{pid}/segments/{sid}")
async def update_segment(pid: str, sid: str, body: SegmentPatchBody) -> Segment:
    project = _load_project_or_404(pid)
    segment = _load_segment_or_404(project, sid)

    data = body.model_dump(exclude_unset=True)

    # --- validation (before any mutation) ---------------------------------
    if "speaker_id" in data and data["speaker_id"] and project.speaker(data["speaker_id"]) is None:
        raise HTTPException(400, f"unknown speaker '{data['speaker_id']}'")

    new_start = data["start"] if "start" in data else segment.start
    new_end = data["end"] if "end" in data else segment.end
    duration = project.media.duration if project.media else None
    if duration is not None and duration > 0:
        new_start = max(0.0, min(new_start, duration))
        new_end = max(0.0, min(new_end, duration))
    else:
        new_start = max(0.0, new_start)
        new_end = max(0.0, new_end)
    if new_start >= new_end:
        raise HTTPException(400, f"start ({new_start}) must be before end ({new_end})")

    if "active_take_id" in data and data["active_take_id"] is not None:
        if not any(t.id == data["active_take_id"] for t in segment.takes):
            raise HTTPException(400, f"unknown take '{data['active_take_id']}'")

    # --- apply dirty rules (specs/backend-core.md "Dirty rules") -----------
    translate_dirty = False
    synth_dirty = False

    if "source_text" in data and data["source_text"] != segment.source_text:
        segment.source_text = data["source_text"]
        translate_dirty = True
        synth_dirty = True

    for field in ("translated_text", "speaker_id", "emotion"):
        if field in data and data[field] != getattr(segment, field):
            setattr(segment, field, data[field])
            synth_dirty = True

    if new_start != segment.start or new_end != segment.end:
        segment.start = new_start
        segment.end = new_end
        synth_dirty = True

    if translate_dirty:
        segment.translate_dirty = True
    if synth_dirty:
        segment.synth_dirty = True

    if translate_dirty:
        project.mark_downstream_dirty("translate")
    elif synth_dirty:
        project.mark_downstream_dirty("synthesize")

    if "active_take_id" in data and data["active_take_id"] != segment.active_take_id:
        segment.active_take_id = data["active_take_id"]
        # No segment dirty flags for this one, but mix (and downstream) must rerun.
        project.mark_downstream_dirty("synthesize")

    if "notes" in data:
        segment.notes = data["notes"]

    store.save(project)
    engine.bus.publish_project(project)
    return segment


@router.post("/projects/{pid}/segments/{sid}/regenerate")
async def regenerate_segment(pid: str, sid: str, body: RegenerateBody) -> Job:
    project = _load_project_or_404(pid)
    _load_segment_or_404(project, sid)

    if not body.stages:
        raise HTTPException(400, "stages must be a non-empty subset of ('translate', 'synthesize')")

    try:
        return orchestrator.start_segment_job(pid, sid, list(body.stages))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.patch("/projects/{pid}/speakers/{spid}")
async def update_speaker(pid: str, spid: str, body: SpeakerPatchBody) -> Speaker:
    project = _load_project_or_404(pid)
    speaker = project.speaker(spid)
    if speaker is None:
        raise HTTPException(404, f"speaker '{spid}' not found in project '{pid}'")

    if body.name is not None:
        speaker.name = body.name
    if body.color is not None:
        speaker.color = body.color

    store.save(project)
    engine.bus.publish_project(project)
    return speaker
