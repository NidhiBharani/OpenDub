// Mirrors server/app/models.py exactly (snake_case). Change both together.

export type StageKey =
  | 'ingest' | 'separate' | 'analyze' | 'transcribe' | 'translate'
  | 'synthesize' | 'mix' | 'lipsync' | 'review' | 'render'

export const STAGE_ORDER: StageKey[] = [
  'ingest', 'separate', 'analyze', 'transcribe', 'translate', 'synthesize', 'mix', 'lipsync',
  'review', 'render',
]

export const STAGE_LABELS: Record<StageKey, string> = {
  ingest: 'Ingest', separate: 'Separate', analyze: 'Analyze', transcribe: 'Transcribe',
  translate: 'Translate', synthesize: 'Synthesize', mix: 'Mix', lipsync: 'Lip sync',
  review: 'Review', render: 'Render',
}

// One provider kind per capability (see server/app/capabilities.py). Providers register as
// "<kind>.<slug>"; the original six kinds double as PipelineConfig fields, the rest are
// configured through PipelineConfig.capabilities.
export type ProviderKind = string

export const LEGACY_KINDS = [
  'separation', 'asr', 'diarization', 'translation', 'tts', 'lipsync',
] as const
export type LegacyKind = (typeof LEGACY_KINDS)[number]

/** @deprecated alias of LEGACY_KINDS — the six kinds that live on PipelineConfig itself. */
export const PROVIDER_KINDS = LEGACY_KINDS

export type StageStatus =
  | 'pending' | 'queued' | 'running' | 'done' | 'dirty' | 'error' | 'skipped'
export type JobStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled'

/** Outcome of one capability step inside a stage, keyed by capability id (e.g. "A3"). */
export interface StepState {
  status: 'done' | 'skipped' | 'error'
  provider_id: string
  detail: string
}

export interface StageState {
  status: StageStatus
  detail: string
  updated_at: string | null
  steps?: Record<string, StepState>
}

export interface Take {
  id: string
  path: string
  duration: number
  provider_id: string
  rate_factor: number
  created_at: string
  /** target language the take was voiced in ('' on takes from before multi-language) */
  lang: string
}

/** One recognised word with timing; `text` keeps the recogniser's own spacing. */
export interface Word {
  start: number
  end: number
  text: string
  confidence: number | null
}

export interface Segment {
  id: string
  start: number
  end: number
  speaker_id: string
  source_text: string
  translated_text: string
  emotion: string
  notes: string
  words: Word[]
  takes: Take[]
  active_take_id: string | null
  translate_dirty: boolean
  synth_dirty: boolean
  /** Derived from Project.skip_ranges by the server: this line keeps the original audio and is
   *  neither translated nor voiced. */
  skipped: boolean
}

export interface Speaker {
  id: string
  name: string
  color: string
  reference_path: string | null
}

export interface ProviderChoice {
  provider_id: string
  options: Record<string, unknown>
}

/** Per-capability configuration, keyed by capability id in PipelineConfig.capabilities. For the
 *  six legacy kinds the provider lives in the PipelineConfig field instead. */
export interface CapabilityChoice {
  enabled: boolean
  provider_id: string
  options: Record<string, unknown>
  params: Record<string, unknown>
}

export type PipelineMode = 'simple' | 'advanced'
export type PresetName = 'minimal' | 'balanced' | 'max'
export type PresetSetting = PresetName | 'custom'
export type RuntimeName = 'builtin' | 'local' | 'cloud'
export type RuntimeSetting = RuntimeName | 'custom'

export interface PipelineConfig {
  mode: PipelineMode
  preset: PresetSetting
  runtime: RuntimeSetting
  lipsync_enabled: boolean
  subtitles_enabled: boolean
  capabilities: Record<string, CapabilityChoice>

  separation: ProviderChoice
  asr: ProviderChoice
  diarization: ProviderChoice
  translation: ProviderChoice
  tts: ProviderChoice
  lipsync: ProviderChoice
}

export interface MediaInfo {
  duration: number
  width: number
  height: number
  fps: number
  has_audio: boolean
}

/** A time span the user excludes from dubbing: the mix keeps the original audio (and lip sync
 *  keeps the original picture) between start and end. */
export interface TimeRange {
  id: string
  start: number
  end: number
  label: string
}

export interface Project {
  id: string
  name: string
  created_at: string
  source_filename: string
  source_lang: string
  target_lang: string
  media: MediaInfo | null
  pipeline: PipelineConfig
  stages: Partial<Record<StageKey, StageState>>
  speakers: Speaker[]
  segments: Segment[]
  skip_ranges: TimeRange[]
  /** Version the current state was last restored from / snapshotted to; null when unsaved. */
  active_version_id: string | null
}

// ---- anchors (suggested cut points) ----

export type AnchorKind = 'scene' | 'silence' | 'segment'

/** A suggested cut point on the timeline. `confidence` is 0..1 (scene-change score, silence
 *  length, or 1 for segment boundaries). */
export interface Anchor {
  t: number
  kind: AnchorKind
  confidence: number
  /** silence anchors: the gap's extent; scene anchors: 0-width */
  start: number
  end: number
}

export interface AnchorSet {
  version: number
  anchors: Anchor[]
}

/** A generated filmstrip: one contact sheet JPEG with `cols`×`rows` tiles of `tile_w`×`tile_h`
 *  px, one every `interval` seconds. `url` is served by the media route. */
export interface Filmstrip {
  interval: number
  cols: number
  rows: number
  tile_w: number
  tile_h: number
  count: number
  url: string
}

// ---- versions ----

export type VersionKind = 'auto' | 'manual' | 'pre_restore'

/** A snapshot of one dubbing attempt: the segments (with active takes), speakers, pipeline
 *  config, stage states and the produced outputs at that moment. Outputs are copied under
 *  versions/<id>/ and served through the media route. */
export interface Version {
  id: string
  label: string
  kind: VersionKind
  created_at: string
  source_lang: string
  target_lang: string
  preset: PresetSetting
  runtime: RuntimeSetting
  /** provider ids that ran, keyed by legacy kind */
  providers: Record<string, string>
  stages: Partial<Record<StageKey, StageState>>
  segment_count: number
  voiced_count: number
  skipped_ranges: number
  has_mix: boolean
  has_render: boolean
  /** paths (relative to the project dir) of the copied outputs, keyed 'dub_mix' | 'playback_dub' | 'dubbed' */
  outputs: Record<string, string>
  summary: string
  bytes: number
  job_id: string | null
  /** version the working state had been restored from when this one was taken (lineage) */
  parent_id: string | null
}

export interface ProjectSummary {
  id: string
  name: string
  created_at: string
  duration: number
  target_lang: string
  segment_count: number
  stages: Partial<Record<StageKey, StageState>>
}

export interface Job {
  id: string
  project_id: string
  kind: 'pipeline' | 'segment' | 'ingest' | 'restore'
  stages: StageKey[]
  segment_id: string | null
  status: JobStatus
  progress: number
  stage: StageKey | null
  message: string
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

// ---- providers / settings ----

export interface ConfigField {
  key: string
  label: string
  type: 'string' | 'secret' | 'number' | 'boolean' | 'select'
  default: unknown
  options: string[]
  placeholder: string
  help: string
}

export interface ProviderMeta {
  id: string
  kind: ProviderKind
  name: string
  description: string
  runtime: 'local' | 'cloud'
  fields: ConfigField[]
}

export interface ProviderInfo {
  meta: ProviderMeta
  available: boolean
  reason: string
  configured_options: Record<string, unknown>
}

// ---- capability map ----
// Mirrors server/app/capabilities.py + server/app/api/capabilities.py.

export type Phase = 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G'
export type Tier = 'core' | 'recommended' | 'advanced' | 'experimental' | 'deferred'
export type Slot = 'pre' | 'post' | 'inline'
/** Feature toggle gating a capability; '' means it always applies. */
export type FeatureGate = '' | 'lipsync' | 'subtitles'

export interface Capability {
  id: string // "A1" … "G6"
  kind: ProviderKind
  phase: Phase
  name: string
  summary: string
  stage: StageKey
  slot: Slot
  order: number
  tier: Tier
  needs: string[]
  uses: string[]
  requires: FeatureGate
  legacy_field: string // PipelineConfig field holding the provider choice (original six)
  default_provider: string
  params: ConfigField[]
}

export interface PresetInfo {
  id: string
  description: string
  capabilities: string[]
}

export interface CapabilityMap {
  phases: Record<string, string>
  capabilities: Capability[]
  presets: PresetInfo[]
}

/** What a capability resolves to for one project: what will actually run. */
export interface ResolvedCapability {
  id: string
  enabled: boolean
  provider_id: string
  options: Record<string, unknown>
  params: Record<string, unknown>
  reason: string // why it is off / was switched on
}

export const TIER_LABELS: Record<Tier, string> = {
  core: 'core', recommended: 'recommended', advanced: 'advanced',
  experimental: 'experimental', deferred: 'later',
}

/** The always-available builtin for a capability; "<kind>.off" means "does not run". */
export function capabilityBuiltin(cap: Capability): string {
  return cap.default_provider || `${cap.kind}.off`
}

export interface Waveform {
  version: number
  sample_rate: number // peak pairs per second
  peaks: number[] // flattened [min,max] pairs, values in -1..1
}

export const SECRET_MASK = '•••'

/** Languages offered in the UI (ISO 639-1). The server accepts any code. */
export const LANGUAGES: { code: string; name: string }[] = [
  { code: 'en', name: 'English' }, { code: 'ja', name: 'Japanese' }, { code: 'hi', name: 'Hindi' },
  { code: 'zh', name: 'Chinese' }, { code: 'ko', name: 'Korean' }, { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' }, { code: 'de', name: 'German' }, { code: 'pt', name: 'Portuguese' },
  { code: 'it', name: 'Italian' }, { code: 'ru', name: 'Russian' }, { code: 'ar', name: 'Arabic' },
  { code: 'id', name: 'Indonesian' }, { code: 'ta', name: 'Tamil' }, { code: 'bn', name: 'Bengali' },
]

export function languageName(code: string): string {
  return LANGUAGES.find((l) => l.code === code)?.name ?? code.toUpperCase()
}

export function formatTime(t: number, withMs = false): string {
  if (!isFinite(t) || t < 0) t = 0
  // Split integer milliseconds, not float seconds: `6.3 % 1` is 0.2999…, which floored to "06.299".
  const totalMs = Math.round(t * 1000)
  const h = Math.floor(totalMs / 3_600_000)
  const m = Math.floor((totalMs % 3_600_000) / 60_000)
  const s = Math.floor((totalMs % 60_000) / 1000)
  const base = h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`
  if (!withMs) return base
  return `${base}.${String(totalMs % 1000).padStart(3, '0')}`
}
