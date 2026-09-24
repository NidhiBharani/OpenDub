"""Provider abstraction — every model integration (OSS-local or cloud-API) implements one of the
kind-specific ABCs below and registers itself. The server must import this module and every provider
module WITHOUT any optional ML dependency installed: do heavy imports lazily inside methods, and keep
`available()` cheap (import probe / key presence check — no model loading, no network unless `deep=True`).
"""
from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from ..models import ASRSegment, ProviderKind, TranslationRequest, TTSRequest

# progress(fraction 0..1, message)
ProgressFn = Callable[[float, str], None]


class ConfigField(BaseModel):
    key: str
    label: str
    type: Literal["string", "secret", "number", "boolean", "select"] = "string"
    default: Any = None
    options: list[str] = Field(default_factory=list)  # for type=select
    placeholder: str = ""
    help: str = ""


class ProviderMeta(BaseModel):
    id: str  # "<kind>.<slug>", e.g. "tts.f5_tts"
    kind: ProviderKind
    name: str  # display name, e.g. "F5-TTS"
    description: str = ""
    runtime: Literal["local", "cloud"] = "local"
    fields: list[ConfigField] = Field(default_factory=list)


class Provider(ABC):
    meta: ProviderMeta  # class attribute on every subclass

    def __init__(self, options: dict[str, Any] | None = None):
        merged: dict[str, Any] = {f.key: f.default for f in self.meta.fields}
        merged.update({k: v for k, v in (options or {}).items() if v not in (None, "")})
        self.options = merged

    def opt(self, key: str, default: Any = None) -> Any:
        v = self.options.get(key)
        return default if v in (None, "") else v

    def opt_bool(self, key: str, default: bool = False) -> bool:
        """Boolean option, coercing env-var/YAML string values ('false', '0', 'off', ...) —
        plain bool() would treat any non-empty string, including 'false', as True."""
        v = self.opt(key)
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        text = str(v).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return default

    def opt_float(self, key: str, default: float) -> float:
        """Numeric option that only falls back when unset — unlike `float(opt(...) or default)`,
        an explicitly configured 0 (or 0.0) is preserved."""
        v = self.opt(key)
        if v is None:
            return float(default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def available(self, deep: bool = False) -> tuple[bool, str]:
        """(is_usable, human_reason). Cheap by default; `deep=True` may hit the network/API."""
        return True, "ready"

    @staticmethod
    def _can_import(*modules: str) -> tuple[bool, str]:
        for m in modules:
            try:
                spec = importlib.util.find_spec(m)
            except (ModuleNotFoundError, ImportError, ValueError):
                # find_spec imports parent packages for dotted names (e.g. 'pyannote.audio')
                # and raises when the parent itself is missing — that still just means
                # "not installed" for available()'s cheap-probe contract.
                spec = None
            if spec is None:
                return False, f"python package '{m}' is not installed"
        return True, "ready"


class SeparationProvider(Provider):
    @abstractmethod
    async def separate(
        self, audio: Path, vocals_out: Path, background_out: Path, progress: ProgressFn
    ) -> None: ...


class ASRProvider(Provider):
    @abstractmethod
    async def transcribe(
        self, audio: Path, language: str, progress: ProgressFn
    ) -> list[ASRSegment]: ...


class DiarizationProvider(Provider):
    @abstractmethod
    async def diarize(
        self, audio: Path, segments: list[ASRSegment], progress: ProgressFn
    ) -> list[str]:
        """Return one opaque speaker label (e.g. 'S0', 'S1') per input segment, same order."""
        ...


class TranslationProvider(Provider):
    @abstractmethod
    async def translate(
        self,
        requests: list[TranslationRequest],
        source_lang: str,
        target_lang: str,
        progress: ProgressFn,
    ) -> list[str]:
        """Return one translation per request, same order. Respect req.duration: translations are
        spoken dubs and should fit the slot (roughly ≤ duration * 15 chars/sec for English)."""
        ...


class TTSProvider(Provider):
    @abstractmethod
    async def synthesize(self, req: TTSRequest, out_wav: Path, progress: ProgressFn) -> None:
        """Write mono or stereo wav (any sample rate; pipeline resamples) to out_wav."""
        ...


class LipSyncProvider(Provider):
    @abstractmethod
    async def sync(self, video: Path, audio: Path, out_video: Path, progress: ProgressFn) -> None: ...


class StepContext(BaseModel):
    """What a capability step gets to work with. Steps read/write project artifacts under
    `project_dir` and mutate `project` in place; the stage persists it afterwards."""

    model_config = {"arbitrary_types_allowed": True}

    project: Any  # app.models.Project (Any: avoids a models↔base import cycle in annotations)
    project_dir: Path
    capability_id: str
    params: dict[str, Any] = Field(default_factory=dict)  # capability-level knobs, defaults merged
    enabled: dict[str, bool] = Field(default_factory=dict)  # capability id → resolved on/off


class StepProvider(Provider):
    """Base for every capability beyond the original six kinds (see app/capabilities.py).
    `run` does the capability's work and returns a one-line detail for the run view. Raise
    StepSkipped when there is nothing to do."""

    @abstractmethod
    async def run(self, ctx: StepContext, progress: ProgressFn) -> str: ...


class StepSkipped(Exception):
    """A capability step intentionally did nothing."""


class OffProvider(StepProvider):
    """The '<kind>.off' choice: selecting it disables the capability (capabilities.resolve)."""

    async def run(self, ctx: StepContext, progress: ProgressFn) -> str:
        raise StepSkipped("capability is off")


class InlineBuiltin(StepProvider):
    """Builtin whose behaviour is implemented by the stage body itself (e.g. atempo timing fit,
    the static loudness chain). Registered so it is selectable and documented in the map."""

    async def run(self, ctx: StepContext, progress: ProgressFn) -> str:
        return "handled by the stage"


REGISTRY: dict[str, type[Provider]] = {}


def register(cls: type[Provider]) -> type[Provider]:
    REGISTRY[cls.meta.id] = cls
    return cls


def get_provider_class(provider_id: str) -> type[Provider]:
    if provider_id not in REGISTRY:
        raise KeyError(f"unknown provider '{provider_id}'")
    return REGISTRY[provider_id]


def load_all() -> None:
    """Import every module in the kind sub-packages so @register side effects run."""
    from . import asr, diarization, lipsync, separation, translation, tts  # noqa: F401

    for pkg in (asr, diarization, lipsync, separation, translation, tts):
        for mod in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg.__name__}.{mod.name}")

    from . import steps  # capability step providers, one sub-package per kind

    for mod in pkgutil.walk_packages(steps.__path__, f"{steps.__name__}."):
        importlib.import_module(mod.name)
    _register_builtins()


_INLINE_BUILTINS: dict[str, tuple[str, str]] = {
    "segmentation.pause": ("Pause + punctuation splitter", "Regroups recognized words into speakable lines: splits on real pauses and sentence ends, and caps line length. Needs word timing; falls back to the recognizer's segments."),
    "segmentation.verbatim": ("ASR segments as lines", "Uses the recognizer's segments as dub lines, unchanged."),
    "style.segment_reference": ("Line audio as style reference", "Passes each line's own source audio to the voice model as the style prompt."),
    "timing_fit.atempo": ("ffmpeg atempo", "Uniform tempo stretch, clamped to 0.6–1.6×."),
    "loudness.static": ("Static gain chain", "Dialogue −18 LUFS, bed ducked 8 dB, master −16 LUFS, −1.5 dBTP limiter."),
}


def _register_builtins() -> None:
    """Make sure every capability has its always-available builtin registered: the documented
    inline ones above, and a '<kind>.off' for every kind that can be switched off."""
    from ..capabilities import CAPABILITIES

    for provider_id, (name, description) in _INLINE_BUILTINS.items():
        if provider_id not in REGISTRY:
            kind, slug = provider_id.split(".", 1)
            meta = ProviderMeta(id=provider_id, kind=kind, name=name, description=description)
            register(type(f"Inline_{kind}_{slug}", (InlineBuiltin,), {"meta": meta}))
    for cap in CAPABILITIES:
        off_id = f"{cap.kind}.off"
        if not cap.legacy_field and cap.tier != "core" and off_id not in REGISTRY:
            meta = ProviderMeta(
                id=off_id, kind=cap.kind, name="Off", description="This capability does not run."
            )
            register(type(f"Off_{cap.kind}", (OffProvider,), {"meta": meta}))
