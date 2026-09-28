"""Segment + speaker mutation routes (dirty-flag propagation lives here — see ARCHITECTURE.md
"Dirty tracking / invalidation").
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import store
from ..jobs import engine
from ..models import Job, Project, Segment, Speaker, Word
from ..pipeline import orchestrator, regions

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


class SplitBody(BaseModel):
    at: float | None = None
    word_index: int | None = None  # split before this word (alternative to `at`)


class MergeBody(BaseModel):
    with_: Literal["next", "prev"] = "next"


class CreateSegmentBody(BaseModel):
    start: float
    end: float
    speaker_id: str | None = None
    source_text: str = ""
    emotion: str = ""


def _refuse_if_busy(pid: str) -> None:
    job = engine.active(pid)
    if job is not None:
        raise HTTPException(409, f"a job is {job.status} for this project; wait for it or cancel it")


def _touch(project: Project) -> None:
    """Any edit makes the working state differ from every saved version."""
    project.active_version_id = None


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
    async with store.lock(pid):
        segment, project = _update_segment_locked(pid, sid, body)
    engine.bus.publish_project(project)
    return segment


def _update_segment_locked(pid: str, sid: str, body: SegmentPatchBody) -> tuple[Segment, Project]:
    project = _load_project_or_404(pid)
    segment = _load_segment_or_404(project, sid)

    data = body.model_dump(exclude_unset=True)

    # --- validation (before any mutation) ---------------------------------
    # Explicit JSON nulls are only meaningful for active_take_id; on any other field they would
    # write None into a non-nullable str/float field and brick the manifest on the next load.
    for field in ("source_text", "translated_text", "speaker_id", "start", "end", "emotion", "notes"):
        if field in data and data[field] is None:
            raise HTTPException(400, f"'{field}' cannot be null")

    if "speaker_id" in data and data["speaker_id"] and project.speaker(data["speaker_id"]) is None:
        raise HTTPException(400, f"unknown speaker '{data['speaker_id']}'")

    new_start = data.get("start", segment.start)
    new_end = data.get("end", segment.end)
    duration = project.media.duration if project.media else None
    if duration is not None and duration > 0:
        new_start = max(0.0, min(new_start, duration))
        new_end = max(0.0, min(new_end, duration))
    else:
        new_start = max(0.0, new_start)
        new_end = max(0.0, new_end)
    if new_start >= new_end:
        raise HTTPException(400, f"start ({new_start}) must be before end ({new_end})")

    if data.get("active_take_id") is not None and not any(
            t.id == data["active_take_id"] for t in segment.takes):
        raise HTTPException(400, f"unknown take '{data['active_take_id']}'")

    # --- apply dirty rules (ARCHITECTURE.md "Dirty tracking / invalidation") -----------
    translate_dirty = False
    synth_dirty = False

    if "source_text" in data and data["source_text"] != segment.source_text:
        segment.words = _remap_words(segment.words, segment.source_text, data["source_text"])
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

    if translate_dirty or synth_dirty or ("active_take_id" in data):
        _touch(project)
    store.save(project)
    return segment, project


def _remap_words(words: list[Word], old_text: str, new_text: str) -> list[Word]:
    """Keep word timings across a text edit when the token count is unchanged (a corrected
    spelling keeps its slot); otherwise the timings are unknown and the words are dropped."""
    if not words:
        return []
    old_tokens = old_text.split()
    new_tokens = new_text.split()
    if len(old_tokens) != len(new_tokens) or len(words) != len(old_tokens):
        return []
    out = []
    for w, tok in zip(words, new_tokens):
        lead = " " if w.text.startswith(" ") else ""
        out.append(w.model_copy(update={"text": lead + tok}))
    return out


def _ordered(project: Project) -> list[Segment]:
    return sorted(project.segments, key=lambda s: (s.start, s.end))


@router.post("/projects/{pid}/segments/{sid}/split")
async def split_segment(pid: str, sid: str, body: SplitBody) -> Project:
    """Split one line into two at `at` seconds (or before word `word_index`). Text is divided
    at the nearest word boundary; both halves need translating and voicing again."""
    _refuse_if_busy(pid)
    async with store.lock(pid):
        project = _load_project_or_404(pid)
        seg = _load_segment_or_404(project, sid)
        if body.word_index is not None:
            if not (0 < body.word_index < len(seg.words)):
                raise HTTPException(400, "word_index must be strictly inside the line's words")
            at = seg.words[body.word_index].start
        elif body.at is not None:
            at = body.at
        else:
            raise HTTPException(400, "pass 'at' (seconds) or 'word_index'")
        if not (seg.start + 0.05 < at < seg.end - 0.05):
            raise HTTPException(400, f"split point {at:.3f} must be inside the line ({seg.start:.3f}–{seg.end:.3f})")

        if seg.words:
            left_words = [w for w in seg.words if (w.start + w.end) / 2 < at]
            right_words = [w for w in seg.words if (w.start + w.end) / 2 >= at]
            left_text = "".join(w.text for w in left_words).strip()
            right_text = "".join(w.text for w in right_words).strip()
            if not left_text or not right_text:
                # the split lands before the first / after the last word: fall back to text ratio
                left_words, right_words = [], []
        else:
            left_words, right_words = [], []
            left_text = right_text = ""
        if not left_words and not right_words:
            text = seg.source_text
            cut = round(len(text) * (at - seg.start) / max(seg.duration, 1e-6))
            # prefer the nearest whitespace/punctuation boundary
            candidates = [i for i, ch in enumerate(text) if ch in " 、。,.!?！？　"]
            if candidates:
                cut = min(candidates, key=lambda i: abs(i - cut)) + 1
            left_text, right_text = text[:cut].strip(), text[cut:].strip()

        right = Segment(
            start=round(at, 3),
            end=seg.end,
            speaker_id=seg.speaker_id,
            source_text=right_text,
            emotion=seg.emotion,
            notes=seg.notes,
            words=right_words,
            translate_dirty=True,
            synth_dirty=True,
        )
        seg.end = round(at, 3)
        seg.source_text = left_text
        seg.words = left_words
        seg.translated_text = ""
        seg.active_take_id = None
        seg.translate_dirty = True
        seg.synth_dirty = True
        idx = project.segments.index(seg)
        project.segments.insert(idx + 1, right)
        regions.apply_skip_ranges(project)
        project.mark_downstream_dirty("translate")
        _touch(project)
        store.save(project)
    engine.bus.publish_project(project)
    return project


@router.post("/projects/{pid}/segments/{sid}/merge-next")
async def merge_segment_with_next(pid: str, sid: str) -> Project:
    """Merge a line with the next one in time order. The earlier line survives (its id and
    speaker); the merged text needs translating and voicing again."""
    _refuse_if_busy(pid)
    async with store.lock(pid):
        project = _load_project_or_404(pid)
        seg = _load_segment_or_404(project, sid)
        ordered = _ordered(project)
        i = ordered.index(seg)
        if i + 1 >= len(ordered):
            raise HTTPException(400, "this is the last line; nothing to merge with")
        other = ordered[i + 1]
        joiner = "" if (other.words and not other.words[0].text.startswith(" ")) else " "
        if not other.words and not seg.words:
            joiner = "" if _unspaced(project.source_lang) else " "
        seg.end = max(seg.end, other.end)
        seg.source_text = (seg.source_text.strip() + joiner + other.source_text.strip()).strip()
        seg.words = seg.words + other.words
        seg.emotion = seg.emotion or other.emotion
        seg.notes = "\n".join(x for x in (seg.notes, other.notes) if x)
        seg.translated_text = ""
        seg.active_take_id = None
        seg.translate_dirty = True
        seg.synth_dirty = True
        project.segments = [x for x in project.segments if x.id != other.id]
        regions.apply_skip_ranges(project)
        project.mark_downstream_dirty("translate")
        _touch(project)
        store.save(project)
    engine.bus.publish_project(project)
    return project


_UNSPACED = {"ja", "zh", "th", "yue", "lo", "km", "my"}


def _unspaced(lang: str) -> bool:
    return lang.split("-")[0].lower() in _UNSPACED


@router.post("/projects/{pid}/segments")
async def create_segment(pid: str, body: CreateSegmentBody) -> Project:
    _refuse_if_busy(pid)
    async with store.lock(pid):
        project = _load_project_or_404(pid)
        duration = project.media.duration if project.media else None
        start = max(0.0, body.start)
        end = body.end if duration is None else min(body.end, duration)
        if start >= end:
            raise HTTPException(400, f"start ({start}) must be before end ({end})")
        speaker_id = body.speaker_id
        if speaker_id and project.speaker(speaker_id) is None:
            raise HTTPException(400, f"unknown speaker '{speaker_id}'")
        if not speaker_id:
            before = [s for s in _ordered(project) if s.start <= start]
            speaker_id = before[-1].speaker_id if before else (project.speakers[0].id if project.speakers else "")
        seg = Segment(
            start=round(start, 3),
            end=round(end, 3),
            speaker_id=speaker_id,
            source_text=body.source_text.strip(),
            emotion=body.emotion,
            translate_dirty=bool(body.source_text.strip()),
        )
        project.segments.append(seg)
        project.segments.sort(key=lambda s: (s.start, s.end))
        regions.apply_skip_ranges(project)
        if seg.translate_dirty and not seg.skipped:
            project.mark_downstream_dirty("translate")
        _touch(project)
        store.save(project)
    engine.bus.publish_project(project)
    return project


@router.delete("/projects/{pid}/segments/{sid}")
async def delete_segment(pid: str, sid: str) -> Project:
    _refuse_if_busy(pid)
    async with store.lock(pid):
        project = _load_project_or_404(pid)
        seg = _load_segment_or_404(project, sid)
        had_audio = seg.active_take() is not None and not seg.skipped
        project.segments = [x for x in project.segments if x.id != sid]
        if had_audio:
            project.mark_downstream_dirty("synthesize")
        _touch(project)
        store.save(project)
    engine.bus.publish_project(project)
    return project


@router.post("/projects/{pid}/segments/{sid}/transcribe")
async def transcribe_segment(pid: str, sid: str) -> Job:
    """Re-run speech recognition over just this line's time range."""
    project = _load_project_or_404(pid)
    _load_segment_or_404(project, sid)
    try:
        return orchestrator.start_segment_job(pid, sid, ["transcribe"])
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/projects/{pid}/segments/{sid}/regenerate")
async def regenerate_segment(pid: str, sid: str, body: RegenerateBody) -> Job:
    project = _load_project_or_404(pid)
    segment = _load_segment_or_404(project, sid)

    if not body.stages:
        raise HTTPException(400, "stages must be a non-empty subset of ('translate', 'synthesize')")
    if segment.skipped:
        raise HTTPException(409, "this line keeps the original audio (inside a skip range)")

    try:
        return orchestrator.start_segment_job(pid, sid, list(body.stages))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.patch("/projects/{pid}/speakers/{spid}")
async def update_speaker(pid: str, spid: str, body: SpeakerPatchBody) -> Speaker:
    async with store.lock(pid):
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
