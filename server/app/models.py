"""Domain models — the single source of truth for OpenDub's data shapes.

Mirrored field-for-field (snake_case) in web/src/types.ts. Change both together.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

StageKey = Literal[
    "ingest", "separate", "analyze", "transcribe", "translate", "synthesize", "mix", "lipsync",
    "review", "render",
]
STAGE_ORDER: list[StageKey] = [
    "ingest", "separate", "analyze", "transcribe", "translate", "synthesize", "mix", "lipsync",
    "review", "render",
]

# One provider kind per capability (see app/capabilities.py). The original six double as
# PipelineConfig fields; the rest are configured through PipelineConfig.capabilities.
ProviderKind = str
LEGACY_KINDS = ("separation", "asr", "diarization", "translation", "tts", "lipsync")

StageStatus = Literal["pending", "queued", "running", "done", "dirty", "error", "skipped"]
JobStatus = Literal["queued", "running", "done", "error", "cancelled"]


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class StepState(BaseModel):
    """Outcome of one capability step inside a stage (keyed by capability id, e.g. "A3")."""

    status: Literal["done", "skipped", "error"] = "done"
    provider_id: str = ""
    detail: str = ""


class StageState(BaseModel):
    status: StageStatus = "pending"
    detail: str = ""
    updated_at: datetime | None = None
    steps: dict[str, StepState] = Field(default_factory=dict)


class Take(BaseModel):
    id: str = Field(default_factory=lambda: new_id("take"))
    path: str  # relative to project dir, e.g. audio/segments/seg_ab12cd34/take_1.wav
    duration: float = 0.0
    provider_id: str = ""
    rate_factor: float = 1.0  # atempo factor applied during mix to fit the slot
    created_at: datetime = Field(default_factory=now)
    lang: str = ""  # target language the take was voiced in ("" on takes from before this field)


class Word(BaseModel):
    """One recognized word with its timing. `text` keeps the recognizer's own spacing (a leading
    space in spaced languages, none in e.g. Japanese) so words join back with ''."""

    start: float
    end: float
    text: str
    confidence: float | None = None


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
    words: list[Word] = Field(default_factory=list)  # source-side word timing (A4), may be empty
    takes: list[Take] = Field(default_factory=list)
    active_take_id: str | None = None
    translate_dirty: bool = False
    synth_dirty: bool = False
    # Derived from Project.skip_ranges (pipeline/regions.apply_skip_ranges): a skipped line keeps
    # the original audio and is neither translated nor voiced. Persisted so stages, the checkpoint
    # merge and the UI agree without recomputing.
    skipped: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def active_take(self) -> Take | None:
        for t in self.takes:
            if t.id == self.active_take_id:
                return t
        return None


class TimeRange(BaseModel):
    """A span the user excludes from dubbing: the mix keeps the original audio (and lip sync the
    original picture) between start and end."""

    id: str = Field(default_factory=lambda: new_id("rng"))
    start: float
    end: float
    label: str = ""
    source: Literal["user", "A3", "D7"] = "user"  # who created it; always user-editable


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


class CapabilityChoice(BaseModel):
    """Per-capability configuration, keyed by capability id in PipelineConfig.capabilities.
    For the six legacy kinds the provider lives in the PipelineConfig field instead and
    `provider_id` here is informational."""

    enabled: bool = False
    provider_id: str = ""
    options: dict[str, Any] = Field(default_factory=dict)  # provider options override
    params: dict[str, Any] = Field(default_factory=dict)  # capability-level knobs


class PipelineConfig(BaseModel):
    # Simple mode applies a preset+runtime (capabilities.apply_preset writes concrete choices);
    # any per-capability edit in Advanced mode flips preset to "custom".
    mode: Literal["simple", "advanced"] = "simple"
    preset: Literal["minimal", "balanced", "max", "custom"] = "minimal"
    runtime: Literal["builtin", "local", "cloud", "custom"] = "builtin"
    lipsync_enabled: bool = False
    subtitles_enabled: bool = False
    capabilities: dict[str, CapabilityChoice] = Field(default_factory=dict)

    separation: ProviderChoice = ProviderChoice(provider_id="separation.passthrough")
    asr: ProviderChoice = ProviderChoice(provider_id="asr.mock")
    diarization: ProviderChoice = ProviderChoice(provider_id="diarization.single_speaker")
    translation: ProviderChoice = ProviderChoice(provider_id="translation.mock")
    tts: ProviderChoice = ProviderChoice(provider_id="tts.mock")
    lipsync: ProviderChoice = ProviderChoice(provider_id="lipsync.none")

    @model_validator(mode="after")
    def _legacy_lipsync(self) -> PipelineConfig:
        # Manifests written before the capability map have no lipsync_enabled; a configured
        # lip-sync provider there means the user wants lip sync.
        if "lipsync_enabled" not in self.model_fields_set and (
            self.lipsync.provider_id != "lipsync.none"
        ):
            self.lipsync_enabled = True
        return self

    def choice(self, kind: ProviderKind) -> ProviderChoice:
        if kind not in LEGACY_KINDS:
            raise KeyError(f"'{kind}' is not a pipeline-field provider kind")
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
    skip_ranges: list[TimeRange] = Field(default_factory=list)
    # Version the working state was last snapshotted to / restored from; any mutation route
    # clears it (the working state then differs from every version).
    active_version_id: str | None = None

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
    source_lang: str = "ja"
    target_lang: str = "en"
    segment_count: int = 0
    stages: dict[str, StageState] = Field(default_factory=dict)
    version_count: int = 0
    languages: list[str] = Field(default_factory=list)  # distinct target languages dubbed so far


VersionKind = Literal["auto", "manual", "pre_restore"]


class Version(BaseModel):
    """Metadata of one dubbing attempt, stored as versions/<id>/meta.json. The full state
    (segments, speakers, pipeline, skip ranges) lives next to it in state.json."""

    id: str = Field(default_factory=lambda: new_id("ver"))
    label: str = ""
    kind: VersionKind = "auto"
    created_at: datetime = Field(default_factory=now)
    source_lang: str = "ja"
    target_lang: str = "en"
    preset: str = "minimal"
    runtime: str = "builtin"
    providers: dict[str, str] = Field(default_factory=dict)  # legacy kind -> provider id
    stages: dict[str, StageState] = Field(default_factory=dict)
    segment_count: int = 0
    voiced_count: int = 0
    skipped_ranges: int = 0
    has_mix: bool = False
    has_render: bool = False
    outputs: dict[str, str] = Field(default_factory=dict)  # 'dub_mix' | 'playback_dub' | 'dubbed' -> rel path
    summary: str = ""
    bytes: int = 0
    job_id: str | None = None
    parent_id: str | None = None  # active_version_id when this one was taken (lineage)


class VersionState(BaseModel):
    """versions/<id>/state.json — everything needed to restore the working state."""

    pipeline: PipelineConfig
    media: MediaInfo | None = None
    speakers: list[Speaker]
    segments: list[Segment]  # Take.path rewritten to versions/<id>/takes/...
    skip_ranges: list[TimeRange] = Field(default_factory=list)


class Job(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    project_id: str
    kind: Literal["pipeline", "segment", "ingest", "restore"]
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
    words: list[Word] = Field(default_factory=list)  # empty when the recognizer has no word timing


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
    # Transcript of segment_reference (the line's source_text). Lets zero-shot TTS skip its
    # internal ASR pass over the reference clip — which fails on hard audio (e.g. sung lines),
    # collapsing the output-duration estimate to near zero.
    segment_reference_text: str = ""
