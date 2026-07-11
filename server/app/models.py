"""Domain models — the single source of truth for OpenDub's data shapes.

Mirrored field-for-field (snake_case) in web/src/types.ts. Change both together.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

StageKey = Literal[
    "ingest", "separate", "transcribe", "translate", "synthesize", "mix", "lipsync", "render"
]
STAGE_ORDER: list[StageKey] = [
    "ingest", "separate", "transcribe", "translate", "synthesize", "mix", "lipsync", "render"
]

ProviderKind = Literal["separation", "asr", "diarization", "translation", "tts", "lipsync"]

StageStatus = Literal["pending", "queued", "running", "done", "dirty", "error", "skipped"]
JobStatus = Literal["queued", "running", "done", "error", "cancelled"]


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class StageState(BaseModel):
    status: StageStatus = "pending"
    detail: str = ""
    updated_at: datetime | None = None


class Take(BaseModel):
    id: str = Field(default_factory=lambda: new_id("take"))
    path: str  # relative to project dir, e.g. audio/segments/seg_ab12cd34/take_1.wav
    duration: float = 0.0
    provider_id: str = ""
    rate_factor: float = 1.0  # atempo factor applied during mix to fit the slot
    created_at: datetime = Field(default_factory=now)


class Segment(BaseModel):
    # Defense in depth: an invalid assignment (e.g. None into a str field) fails immediately
    # instead of being persisted and bricking the manifest on the next load.
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: new_id("seg"))
    start: float
    end: float
    speaker_id: str = ""
    source_text: str = ""
    translated_text: str = ""
    emotion: str = ""  # free-text style hint, e.g. "angry, shouting"
    notes: str = ""
    takes: list[Take] = Field(default_factory=list)
    active_take_id: str | None = None
    translate_dirty: bool = False
    synth_dirty: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def active_take(self) -> Take | None:
        for t in self.takes:
            if t.id == self.active_take_id:
                return t
        return None


class Speaker(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: new_id("spk"))
    name: str
    color: str  # hex from SPEAKER_PALETTE
    reference_path: str | None = None  # speakers/<id>/reference.wav


# NOTE: deliberately avoids the UI accent (#e8604c) so speaker colors never collide with the
# selection highlight.
SPEAKER_PALETTE = [
    "#4c9be8", "#50c878", "#e8b84c", "#b47ce8",
    "#4ccfe8", "#e87cb0", "#a3e84c", "#e89a4c",
]


class ProviderChoice(BaseModel):
    provider_id: str
    options: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    separation: ProviderChoice = ProviderChoice(provider_id="separation.passthrough")
    asr: ProviderChoice = ProviderChoice(provider_id="asr.mock")
    diarization: ProviderChoice = ProviderChoice(provider_id="diarization.single_speaker")
    translation: ProviderChoice = ProviderChoice(provider_id="translation.mock")
    tts: ProviderChoice = ProviderChoice(provider_id="tts.mock")
    lipsync: ProviderChoice = ProviderChoice(provider_id="lipsync.none")

    def choice(self, kind: ProviderKind) -> ProviderChoice:
        return getattr(self, kind)


class MediaInfo(BaseModel):
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = True


class Project(BaseModel):
    # Snapshot of the on-disk state when this instance was loaded with store.load(track=True)
    # (and refreshed on every store.save_merged). Lets job checkpoints re-apply user edits that
    # landed on disk while the job held a stale copy. Not serialized; not part of the contract.
    _checkpoint_base: dict[str, Any] | None = PrivateAttr(default=None)

    id: str = Field(default_factory=lambda: new_id("prj"))
    name: str
    created_at: datetime = Field(default_factory=now)
    source_filename: str = ""
    source_lang: str = "ja"
    target_lang: str = "en"
    media: MediaInfo | None = None
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    stages: dict[str, StageState] = Field(default_factory=dict)
    speakers: list[Speaker] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)

    def stage(self, key: StageKey) -> StageState:
        return self.stages.setdefault(key, StageState())

    def segment(self, sid: str) -> Segment | None:
        return next((s for s in self.segments if s.id == sid), None)

    def speaker(self, spid: str) -> Speaker | None:
        return next((s for s in self.speakers if s.id == spid), None)

    def mark_downstream_dirty(self, after: StageKey) -> None:
        """Mark every stage after `after` dirty (if it had completed)."""
        for key in STAGE_ORDER[STAGE_ORDER.index(after) + 1 :]:
            st = self.stage(key)
            if st.status in ("done", "error"):
                st.status = "dirty"
                st.updated_at = now()


class ProjectSummary(BaseModel):
    id: str
    name: str
    created_at: datetime
    duration: float = 0.0
    target_lang: str = "en"
    segment_count: int = 0
    stages: dict[str, StageState] = Field(default_factory=dict)


class Job(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    project_id: str
    kind: Literal["pipeline", "segment", "ingest"]
    stages: list[StageKey] = Field(default_factory=list)
    segment_id: str | None = None
    status: JobStatus = "queued"
    progress: float = 0.0  # 0..1 overall
    stage: StageKey | None = None  # currently running stage
    message: str = ""
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


# ---- provider-facing DTOs ----------------------------------------------------


class ASRSegment(BaseModel):
    start: float
    end: float
    text: str


class TranslationRequest(BaseModel):
    text: str
    emotion: str = ""
    speaker_name: str = ""
    duration: float = 0.0  # seconds available in the slot
    context_before: list[str] = Field(default_factory=list)
    context_after: list[str] = Field(default_factory=list)


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    emotion: str = ""
    target_duration: float = 0.0
    speaker_reference: str | None = None  # absolute path to speaker identity reference wav
    segment_reference: str | None = None  # absolute path to this line's source audio (style/emotion)
