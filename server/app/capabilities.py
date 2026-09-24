"""Capability map — the declarative registry of the 41 jobs in OpenDub's inference map (phases
A–G), plus the presets that decide which of them run.

A *capability* is one job ("A3 · speech / music / singing classification"). Each has a provider
kind, the macro stage it runs inside, a tier, and its dependencies. The six original provider
kinds (separation, asr, diarization, translation, tts, lipsync) are capabilities too: their
provider choice lives in the matching ``PipelineConfig`` field (``legacy_field``); every other
capability is configured through ``PipelineConfig.capabilities[<id>]``.

Simple mode = pick a preset + runtime and ``apply_preset`` writes concrete choices into the
config. Advanced mode edits those choices one capability at a time (preset becomes "custom").
``resolve`` turns a config into what will actually run, applying dependency rules.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import CapabilityChoice, PipelineConfig, ProviderChoice, StageKey
from .providers.base import REGISTRY, ConfigField

Phase = Literal["A", "B", "C", "D", "E", "F", "G"]
Tier = Literal["core", "recommended", "advanced", "experimental", "deferred"]
# pre/post: run by the stage's step loop before/after the stage body. inline: the stage body
# itself consults the capability (it is the body, or is woven into its per-segment loop).
Slot = Literal["pre", "post", "inline"]
Preset = Literal["minimal", "balanced", "max"]
Runtime = Literal["builtin", "local", "cloud"]

PHASES: dict[str, str] = {
    "A": "Source analysis · audio",
    "B": "Source analysis · picture",
    "C": "Text transformation",
    "D": "Speech generation",
    "E": "Audio post-production",
    "F": "Picture generation",
    "G": "Automatic quality judges",
}


class Capability(BaseModel):
    id: str  # "A1" … "G6"
    kind: str  # provider kind slug; providers register as "<kind>.<slug>"
    phase: Phase
    name: str
    summary: str
    stage: StageKey
    slot: Slot = "post"
    order: int = 0  # position among the stage's steps
    tier: Tier = "advanced"
    needs: list[str] = Field(default_factory=list)  # hard deps: auto-enabled with this one
    uses: list[str] = Field(default_factory=list)  # soft deps: better when present
    requires: Literal["", "lipsync", "subtitles"] = ""  # feature toggle that gates it
    legacy_field: str = ""  # PipelineConfig field holding the provider choice (original six)
    default_provider: str = ""  # always-available builtin; "" ⇒ "<kind>.off"
    params: list[ConfigField] = Field(default_factory=list)  # capability-level knobs

    @property
    def builtin(self) -> str:
        return self.default_provider or f"{self.kind}.off"


def _num(key: str, label: str, default: float, help: str = "") -> ConfigField:
    return ConfigField(key=key, label=label, type="number", default=default, help=help)


def _sel(key: str, label: str, default: str, options: list[str], help: str = "") -> ConfigField:
    return ConfigField(
        key=key, label=label, type="select", default=default, options=options, help=help
    )


def _flag(key: str, label: str, default: bool, help: str = "") -> ConfigField:
    return ConfigField(key=key, label=label, type="boolean", default=default, help=help)


_C = Capability

CAPABILITIES: list[Capability] = [
    # ---- A · source analysis · audio ---------------------------------------------------------
    _C(id="A1", kind="separation", phase="A", name="Dialogue / M&E separation",
       summary="Split the mix into a dialogue stem and an intact music-and-effects bed.",
       stage="separate", slot="inline", tier="core", legacy_field="separation",
       default_provider="separation.passthrough",
       params=[_flag("bed_by_subtraction", "Build bed as mix − dialogue", True,
                     "Separator error becomes quiet dialogue bleed instead of a hollow bed.")]),
    _C(id="A2", kind="enhance", phase="A", name="Reference-clip enhancement",
       summary="Denoise/dereverb the analysis path to build clean cloning references.",
       stage="transcribe", order=40, tier="recommended", uses=["A7"],
       params=[_flag("identity_gate", "Revert when enhancement shifts identity", True)]),
    _C(id="A3", kind="region_detect", phase="A", name="Speech / music / singing regions",
       summary="Detect songs, crowd walla, laughter beds and background TV so they are passed "
               "through instead of transcribed and re-voiced.",
       stage="analyze", order=10, tier="recommended", needs=["A1"],
       params=[_sel("song_action", "Song regions", "passthrough", ["passthrough", "dub", "flag"])]),
    _C(id="A4", kind="asr", phase="A", name="Transcription + word timing",
       summary="Transcribe the dialogue stem with word-level timestamps.",
       stage="transcribe", slot="inline", tier="core", needs=["A1"], legacy_field="asr",
       default_provider="asr.mock",
       params=[_flag("word_timestamps", "Request word timestamps", True)]),
    _C(id="A5", kind="align", phase="A", name="Forced alignment / subtitle import",
       summary="Align known text (e.g. an official subtitle file) to audio at word level.",
       stage="transcribe", order=10, tier="recommended", needs=["A4"]),
    _C(id="A6", kind="diarization", phase="A", name="Diarization + overlap",
       summary="Who speaks when, with overlapped speech detected rather than guessed.",
       stage="transcribe", slot="inline", tier="core", needs=["A4"], legacy_field="diarization",
       default_provider="diarization.single_speaker",
       params=[_flag("word_level", "Attribute speakers per word", True)]),
    _C(id="A7", kind="speaker_embed", phase="A", name="Speaker embeddings / voice bank",
       summary="Link speakers across files, match consented voice profiles, and pick reference "
               "clips by similarity and cleanliness instead of raw length.",
       stage="transcribe", order=30, tier="recommended", needs=["A6"], uses=["A2", "A3"],
       params=[_num("ref_min_seconds", "Reference clip min (s)", 4),
               _num("ref_max_seconds", "Reference clip max (s)", 10)]),
    _C(id="A8", kind="delivery", phase="A", name="Delivery / paralinguistic tagging",
       summary="Per-line rate, loudness, pitch and a free-text delivery note, produced "
               "automatically instead of typed by hand.",
       stage="transcribe", order=50, tier="advanced", needs=["A4"], uses=["A5"]),
    # ---- B · source analysis · picture -------------------------------------------------------
    _C(id="B1", kind="active_speaker", phase="B", name="Active speaker + face tracks",
       summary="Know which on-screen face is speaking and where a mouth is worth editing.",
       stage="analyze", order=40, tier="recommended", needs=["B2"], requires="lipsync",
       params=[_num("min_face_height", "Minimum face height (px)", 96)]),
    _C(id="B2", kind="shots", phase="B", name="Shot boundaries",
       summary="Cut list for per-shot lip sync, subtitle timing and drift control.",
       stage="analyze", order=20, tier="recommended"),
    _C(id="B3", kind="ocr", phase="B", name="On-screen text reading",
       summary="Read slides, lower thirds and burned-in captions; seeds the glossary.",
       stage="analyze", order=50, tier="advanced", uses=["B2"]),
    _C(id="B4", kind="content_type", phase="B", name="Content type gate",
       summary="Classify live action vs animation per shot; blocks human-face lip sync on 2D.",
       stage="analyze", order=30, tier="recommended", uses=["B2"], requires="lipsync"),
    # ---- C · text ----------------------------------------------------------------------------
    _C(id="C1", kind="segmentation", phase="C", name="Dub-line segmentation",
       summary="Merge ASR fragments and split run-ons into speakable lines on real pauses.",
       stage="transcribe", slot="inline", tier="core", needs=["A4"],
       default_provider="segmentation.pause",
       params=[_num("max_line_seconds", "Longest line (s)", 12),
               _num("min_pause", "Split on pauses ≥ (s)", 0.35)]),
    _C(id="C2", kind="translation", phase="C", name="Translation + dubbing adaptation",
       summary="Context-aware, length-budgeted translation written to be spoken.",
       stage="translate", slot="inline", tier="core", needs=["A4"], legacy_field="translation",
       default_provider="translation.mock",
       params=[_num("context_lines", "Context lines each side", 2),
               _num("candidates", "Candidates per line", 1)]),
    _C(id="C3", kind="viseme_rerank", phase="C", name="Viseme-aware re-ranking",
       summary="Prefer translations whose lip closures land where the original mouth closes.",
       stage="translate", order=20, tier="experimental", needs=["C2", "B1", "A5"],
       requires="lipsync", params=[_num("weight", "Re-ranking weight", 0.1)]),
    _C(id="C4", kind="qe", phase="C", name="Translation quality estimation",
       summary="Score every line without a reference; decides which lines a human sees.",
       stage="translate", order=30, tier="recommended", needs=["C2"]),
    _C(id="C5", kind="pronunciation", phase="C", name="Normalization / G2P / lexicon",
       summary="Numbers, acronyms and a per-project pronunciation lexicon for names and jargon.",
       stage="translate", order=40, tier="recommended", needs=["C2"]),
    _C(id="C6", kind="subtitles", phase="C", name="Subtitle condensation",
       summary="Reading-speed-compliant subtitles derived from the same translation.",
       stage="translate", order=50, tier="recommended", needs=["C2"], uses=["B2"],
       requires="subtitles",
       params=[_num("cps", "Characters per second", 17), _num("cpl", "Characters per line", 42)]),
    # ---- D · voice ---------------------------------------------------------------------------
    _C(id="D1", kind="tts", phase="D", name="Zero-shot voice cloning",
       summary="The identity engine: each speaker's own voice in the target language.",
       stage="synthesize", slot="inline", tier="core", needs=["C2"], legacy_field="tts",
       default_provider="tts.mock",
       params=[_num("candidates", "Takes per line", 1)]),
    _C(id="D2", kind="style", phase="D", name="Expressive style transfer",
       summary="Carry the source line's performance into the dub.",
       stage="synthesize", slot="inline", tier="core", needs=["D1"],
       default_provider="style.segment_reference",
       params=[_num("min_reference_seconds", "Shortest usable style clip (s)", 1.5)]),
    _C(id="D3", kind="duration_control", phase="D", name="Duration-controlled synthesis",
       summary="Ask the voice model to hit the slot natively instead of stretching afterwards.",
       stage="synthesize", slot="inline", tier="recommended", needs=["D1"],
       params=[_num("safe_band", "Safe compression band (±)", 0.15)]),
    _C(id="D4", kind="lang_router", phase="D", name="Regional / low-resource routing",
       summary="Route each target language to the engine that is production-grade for it.",
       stage="synthesize", slot="inline", tier="advanced", needs=["D1"]),
    _C(id="D5", kind="stock_voice", phase="D", name="Licensed non-cloning voices",
       summary="Curated stock voices for speakers who decline cloning, and for narration.",
       stage="synthesize", slot="inline", tier="advanced", needs=["D1"]),
    _C(id="D6", kind="voice_convert", phase="D", name="Voice conversion",
       summary="Convert a human or stock take into the target speaker's timbre.",
       stage="synthesize", slot="inline", tier="advanced", needs=["D1"]),
    _C(id="D7", kind="nonverbal", phase="D", name="Non-verbal vocalizations",
       summary="Laughs, gasps and screams: pass the original through instead of speaking them.",
       stage="synthesize", slot="inline", tier="recommended", needs=["A3"]),
    _C(id="D8", kind="s2st", phase="D", name="Direct speech-to-speech (bench only)",
       summary="End-to-end S2ST as a benchmark comparison; never a pipeline route.",
       stage="review", order=90, tier="experimental"),
    # ---- E · audio out -----------------------------------------------------------------------
    _C(id="E1", kind="timing_fit", phase="E", name="Timing fit + pause mapping",
       summary="Land each take in its slot with pauses where the source pauses.",
       stage="mix", slot="inline", tier="core", needs=["D1"], uses=["A4", "B2"],
       default_provider="timing_fit.atempo",
       params=[_num("max_stretch", "Regenerate beyond stretch (±)", 0.15)]),
    _C(id="E2", kind="bandwidth", phase="E", name="Bandwidth extension",
       summary="Extend 24 kHz takes to a 48 kHz delivery bandwidth.",
       stage="mix", slot="inline", tier="recommended", needs=["D1"], uses=["G1", "G2"]),
    _C(id="E3", kind="scene_match", phase="E", name="Acoustic scene matching",
       summary="Match reverb, EQ and distance so the voice sits in the scene.",
       stage="mix", slot="inline", tier="advanced", needs=["D1"], uses=["B2"]),
    _C(id="E4", kind="loudness", phase="E", name="Levelling + loudness",
       summary="Dialogue levelling, ducking and delivery loudness presets.",
       stage="mix", slot="inline", tier="core", needs=["D1"],
       default_provider="loudness.static",
       params=[_sel("delivery", "Delivery preset", "web -16",
                    ["web -16", "EBU R128 -23", "ATSC A/85 -24", "Netflix -27"]),
               _flag("export_stems", "Export stems", False)]),
    _C(id="E5", kind="provenance", phase="E", name="Watermark + provenance",
       summary="Provenance manifest and audio watermark on every render. Deferred.",
       stage="render", order=10, tier="deferred"),
    # ---- F · picture out ---------------------------------------------------------------------
    _C(id="F1", kind="lipsync", phase="F", name="Live-action lip sync",
       summary="Mouth resynthesis, per shot, only where a speaking face is visible.",
       stage="lipsync", slot="inline", tier="recommended", needs=["B2", "B4"], uses=["B1"],
       requires="lipsync", legacy_field="lipsync", default_provider="lipsync.none"),
    _C(id="F2", kind="mouth_restore", phase="F", name="Mouth-region restoration",
       summary="Match the resynthesized mouth to surrounding grain and sharpness.",
       stage="lipsync", order=20, tier="advanced", needs=["F1"], requires="lipsync"),
    _C(id="F3", kind="anim_mouth", phase="F", name="2D animation mouth retiming",
       summary="Research only: declines to edit animation; can export a viseme track.",
       stage="lipsync", order=30, tier="experimental", needs=["B4"], requires="lipsync"),
    _C(id="F4", kind="text_replace", phase="F", name="On-screen text replacement",
       summary="Translated overlay track / editing template for on-screen text.",
       stage="render", order=5, tier="advanced", needs=["B3", "C2"]),
    # ---- G · judges --------------------------------------------------------------------------
    _C(id="G1", kind="judge_intelligibility", phase="G", name="Intelligibility judge",
       summary="Re-transcribe every take; reject mumbled, truncated and hallucinated ones.",
       stage="synthesize", slot="inline", tier="recommended", needs=["D1"]),
    _C(id="G2", kind="judge_similarity", phase="G", name="Speaker similarity judge",
       summary="Pick the take that sounds most like the speaker; flag identity drift.",
       stage="synthesize", slot="inline", tier="recommended", needs=["D1"], uses=["A7"]),
    _C(id="G3", kind="judge_naturalness", phase="G", name="Naturalness prediction",
       summary="Rank candidate takes by predicted naturalness.",
       stage="synthesize", slot="inline", tier="advanced", needs=["D1"]),
    _C(id="G4", kind="judge_emotion", phase="G", name="Emotion consistency judge",
       summary="Flag dubs whose delivery is flatter than the source line.",
       stage="synthesize", slot="inline", tier="advanced", needs=["D1", "A8"]),
    _C(id="G5", kind="judge_sync", phase="G", name="Lip-sync scoring",
       summary="Per-shot sync offset and confidence; abstains on animation.",
       stage="review", order=10, tier="recommended", needs=["F1", "B2"], requires="lipsync"),
    _C(id="G6", kind="reviewer", phase="G", name="Multimodal review",
       summary="A model watches flagged segments in context and explains what is wrong.",
       stage="review", order=20, tier="advanced", needs=["D1"]),
]

BY_ID: dict[str, Capability] = {c.id: c for c in CAPABILITIES}
BY_KIND: dict[str, Capability] = {c.kind: c for c in CAPABILITIES}
KINDS: list[str] = [c.kind for c in CAPABILITIES]

# ---- presets -----------------------------------------------------------------------------------

_MINIMAL = ["A1", "A4", "A6", "C1", "C2", "D1", "D2", "E1", "E4"]
_BALANCED = [*_MINIMAL, "A2", "A3", "A7", "B2", "C4", "C5", "D3", "D7", "E2", "G1", "G2",
             "B1", "B4", "F1", "F2", "G5", "C6"]  # the requires= ones only count when toggled on
_MAX_EXCLUDED = {"E5", "D8", "F3"}

PRESETS: dict[str, list[str]] = {
    "minimal": _MINIMAL,
    "balanced": _BALANCED,
    "max": [c.id for c in CAPABILITIES if c.id not in _MAX_EXCLUDED],
}
PRESET_LABELS: dict[str, str] = {
    "minimal": "Today's pipeline plus the free wins. No extra models.",
    "balanced": "Adds everything that raises quality without a human in the loop.",
    "max": "Every capability that is ready. Slow; may need cloud keys.",
}

# Provider preference per kind for the local / cloud runtimes, best first. The first one whose
# available() passes wins; the builtin is the implicit last resort. Later milestones append here.
RUNTIME_DEFAULTS: dict[str, dict[str, list[str]]] = {
    "local": {
        "separation": ["separation.demucs"],
        "asr": ["asr.faster_whisper"],
        "diarization": ["diarization.pyannote"],
        "translation": ["translation.openai_compatible"],
        "tts": ["tts.f5_tts"],
        "lipsync": ["lipsync.latentsync", "lipsync.wav2lip"],
    },
    "cloud": {
        "separation": ["separation.lalalai"],
        "asr": ["asr.openai_whisper"],
        "diarization": ["diarization.pyannote_api"],
        "translation": ["translation.anthropic", "translation.openai", "translation.deepl"],
        "tts": ["tts.elevenlabs"],
        "lipsync": ["lipsync.replicate"],
    },
}


def _is_available(provider_id: str) -> bool:
    from . import config  # local import: config has no dependency on this module

    cls = REGISTRY.get(provider_id)
    if cls is None:
        return False
    try:
        return cls(config.provider_options(provider_id)).available()[0]
    except Exception:  # noqa: BLE001 - provider code is optional/untrusted
        return False


def pick_provider(cap: Capability, runtime: str) -> tuple[str, str]:
    """(provider_id, note) for a capability under a runtime. Cloud falls back to local, then to
    the builtin, and the note says so."""
    order = {"cloud": ["cloud", "local"], "local": ["local"]}.get(runtime, [])
    for rt in order:
        for pid in RUNTIME_DEFAULTS.get(rt, {}).get(cap.kind, []):
            if _is_available(pid):
                note = "" if rt == runtime else f"no {runtime} provider ready; using {rt}"
                return pid, note
    note = "" if runtime == "builtin" else f"no {runtime} provider ready; using built-in"
    return cap.builtin, note


def feature_on(cfg: PipelineConfig, cap: Capability) -> bool:
    if cap.requires == "lipsync":
        return cfg.lipsync_enabled
    if cap.requires == "subtitles":
        return cfg.subtitles_enabled
    return True


def apply_preset(
    cfg: PipelineConfig,
    preset: str,
    runtime: str,
    lipsync: bool | None = None,
    subtitles: bool | None = None,
) -> list[str]:
    """Simple mode: write concrete choices for every capability into `cfg`. Returns notes about
    fallbacks (e.g. cloud requested but no key configured)."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset '{preset}'; expected one of {sorted(PRESETS)}")
    if runtime not in ("builtin", "local", "cloud"):
        raise ValueError(f"unknown runtime '{runtime}'")
    cfg.preset = preset  # type: ignore[assignment]
    cfg.runtime = runtime  # type: ignore[assignment]
    if lipsync is not None:
        cfg.lipsync_enabled = lipsync
    if subtitles is not None:
        cfg.subtitles_enabled = subtitles

    wanted = set(PRESETS[preset])
    notes: list[str] = []
    cfg.capabilities = {}
    for cap in CAPABILITIES:
        # Lip sync itself follows its toggle in every preset; presets decide its helpers.
        enabled = (cap.id in wanted or cap.id == "F1") and feature_on(cfg, cap)
        provider_id, note = pick_provider(cap, runtime) if enabled else (cap.builtin, "")
        if cap.id == "F1" and not enabled:
            provider_id = "lipsync.none"
        if note:
            notes.append(f"{cap.id} {cap.name}: {note}")
        if cap.legacy_field:
            setattr(cfg, cap.legacy_field, ProviderChoice(provider_id=provider_id))
        cfg.capabilities[cap.id] = CapabilityChoice(enabled=enabled, provider_id=provider_id)
    return notes


class ResolvedCapability(BaseModel):
    id: str
    enabled: bool
    provider_id: str
    options: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""  # why it is off / was switched on


def resolve(cfg: PipelineConfig) -> dict[str, ResolvedCapability]:
    """What will actually run. Order of authority: explicit per-capability choice → preset
    membership → dependency closure (an enabled capability switches its `needs` on) → feature
    gates (lip sync / subtitles off wins over everything) → a provider of "<kind>.off" means off."""
    wanted = set(PRESETS.get(cfg.preset, _MINIMAL))
    out: dict[str, ResolvedCapability] = {}
    for cap in CAPABILITIES:
        choice = cfg.capabilities.get(cap.id)
        enabled = choice.enabled if choice is not None else cap.id in wanted
        if cap.id == "F1":  # driven by the lip-sync toggle + provider, not by preset membership
            enabled = True
        reason = ""
        if enabled and not feature_on(cfg, cap):
            enabled, reason = False, f"{cap.requires} is switched off"
        if cap.legacy_field:
            legacy: ProviderChoice = getattr(cfg, cap.legacy_field)
            provider_id, options = legacy.provider_id, dict(legacy.options)
        else:
            provider_id = (choice.provider_id if choice else "") or cap.builtin
            options = dict(choice.options) if choice else {}
        params = {f.key: f.default for f in cap.params}
        if choice:
            params.update(choice.params)
        out[cap.id] = ResolvedCapability(
            id=cap.id, enabled=enabled, provider_id=provider_id, options=options, params=params,
            reason=reason,
        )

    # dependency closure
    changed = True
    while changed:
        changed = False
        for cap in CAPABILITIES:
            if not out[cap.id].enabled:
                continue
            for dep in cap.needs:
                if not out[dep].enabled and feature_on(cfg, BY_ID[dep]):
                    out[dep].enabled = True
                    out[dep].reason = f"switched on: required by {cap.id}"
                    changed = True

    for cap in CAPABILITIES:
        r = out[cap.id]
        if cap.tier == "core" and not r.enabled:
            r.enabled, r.reason = True, "core capability — always on"
        if r.enabled and cap.id != "F1" and r.provider_id.endswith(".off"):
            r.enabled, r.reason = False, "no provider selected"
        elif cap.id == "F1" and r.provider_id == "lipsync.none":
            r.enabled, r.reason = False, "lip sync provider is 'none'"
    return out


def steps(stage: StageKey, slot: Slot) -> list[Capability]:
    return sorted(
        (c for c in CAPABILITIES if c.stage == stage and c.slot == slot), key=lambda c: c.order
    )
