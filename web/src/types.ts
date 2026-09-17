// Mirrors server/app/models.py exactly (snake_case). Change both together.

export type StageKey =
  | 'ingest' | 'separate' | 'transcribe' | 'translate'
  | 'synthesize' | 'mix' | 'lipsync' | 'render'

export const STAGE_ORDER: StageKey[] = [
  'ingest', 'separate', 'transcribe', 'translate', 'synthesize', 'mix', 'lipsync', 'render',
]

export const STAGE_LABELS: Record<StageKey, string> = {
  ingest: 'Ingest', separate: 'Separate', transcribe: 'Transcribe', translate: 'Translate',
  synthesize: 'Synthesize', mix: 'Mix', lipsync: 'Lip sync', render: 'Render',
}

export type ProviderKind =
  | 'separation' | 'asr' | 'diarization' | 'translation' | 'tts' | 'lipsync'

export const PROVIDER_KINDS: ProviderKind[] = [
  'separation', 'asr', 'diarization', 'translation', 'tts', 'lipsync',
]

export type StageStatus =
  | 'pending' | 'queued' | 'running' | 'done' | 'dirty' | 'error' | 'skipped'
export type JobStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled'

export interface StageState {
  status: StageStatus
  detail: string
  updated_at: string | null
}

export interface Take {
  id: string
  path: string
  duration: number
  provider_id: string
  rate_factor: number
  created_at: string
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
  takes: Take[]
  active_take_id: string | null
  translate_dirty: boolean
  synth_dirty: boolean
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

export type PipelineConfig = Record<ProviderKind, ProviderChoice>

export interface MediaInfo {
  duration: number
  width: number
  height: number
  fps: number
  has_audio: boolean
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
  kind: 'pipeline' | 'segment' | 'ingest'
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

export interface Waveform {
  version: number
  sample_rate: number // peak pairs per second
  peaks: number[] // flattened [min,max] pairs, values in -1..1
}

export const SECRET_MASK = '•••'

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
