// Valid domain objects mirroring server/app/models.py, for use in component tests.
import type {
  Capability, CapabilityMap, Project, ProjectSummary, ProviderInfo, ResolvedCapability,
} from '../types'

let stageClock = 0
// Distinct timestamps per stage, mirroring how a real manifest records each stage completion.
const doneStage = (detail = '') => ({
  status: 'done' as const,
  detail,
  updated_at: `2026-07-08T09:00:${String(stageClock++).padStart(2, '0')}Z`,
})

export const demoSummary: ProjectSummary = {
  id: 'prj_1e2b94e5',
  name: 'Demo Episode',
  created_at: '2026-07-08T08:59:04Z',
  duration: 29.4,
  target_lang: 'en',
  segment_count: 5,
  stages: {
    ingest: doneStage('29.4s, 640x360 @ 24 fps'),
    separate: doneStage(),
    analyze: { status: 'skipped', detail: '', updated_at: null },
    transcribe: {
      ...doneStage('5 segments, 1 speaker(s)'),
      steps: {
        A7: { status: 'done', provider_id: 'speaker_embed.mfcc', detail: '1 voice profiled' },
        A8: { status: 'skipped', provider_id: 'delivery.off', detail: '' },
      },
    },
    translate: doneStage(),
    synthesize: doneStage(),
    mix: doneStage(),
    lipsync: { status: 'skipped', detail: '', updated_at: null },
    review: { status: 'pending', detail: '', updated_at: null },
    render: doneStage(),
  },
}

export const demoProject: Project = {
  id: 'prj_1e2b94e5',
  name: 'Demo Episode',
  created_at: '2026-07-08T08:59:04Z',
  source_filename: 'demo.mp4',
  source_lang: 'ja',
  target_lang: 'en',
  media: { duration: 29.4, width: 640, height: 360, fps: 24, has_audio: true },
  pipeline: {
    mode: 'simple',
    preset: 'minimal',
    runtime: 'builtin',
    lipsync_enabled: false,
    subtitles_enabled: false,
    capabilities: {},
    separation: { provider_id: 'separation.passthrough', options: {} },
    asr: { provider_id: 'asr.mock', options: {} },
    diarization: { provider_id: 'diarization.single_speaker', options: {} },
    translation: { provider_id: 'translation.mock', options: {} },
    tts: { provider_id: 'tts.mock', options: {} },
    lipsync: { provider_id: 'lipsync.none', options: {} },
  },
  stages: demoSummary.stages,
  speakers: [{ id: 'spk_d639006e', name: 'Speaker 1', color: '#e8604c', reference_path: 'speakers/spk_d639006e/reference.wav' }],
  segments: [
    {
      id: 'seg_019e618e', start: 1.0, end: 3.2, speaker_id: 'spk_d639006e',
      source_text: '[line 1]', translated_text: '(en) [line 1]', emotion: '', notes: '', words: [],
      takes: [], active_take_id: null, translate_dirty: false, synth_dirty: false, skipped: false,
    },
  ],
  skip_ranges: [],
  active_version_id: null,
}

export const mockProvider: ProviderInfo = {
  meta: {
    id: 'translation.mock', kind: 'translation', name: 'Mock translator',
    description: 'Always-available fallback', runtime: 'local', fields: [],
  },
  available: true,
  reason: 'ready',
  configured_options: {},
}

const provider = (id: string, kind: string, name: string): ProviderInfo => ({
  meta: { id, kind, name, description: `${name} provider`, runtime: 'local', fields: [] },
  available: true,
  reason: 'ready',
  configured_options: {},
})

/** A registry covering every kind used by `capabilityMap` below. */
export const demoProviders: ProviderInfo[] = [
  mockProvider,
  provider('separation.passthrough', 'separation', 'Passthrough'),
  provider('separation.demucs', 'separation', 'Demucs'),
  provider('region_detect.off', 'region_detect', 'Off'),
  provider('region_detect.energy', 'region_detect', 'Energy heuristic'),
  provider('lipsync.none', 'lipsync', 'None'),
  provider('lipsync.wav2lip', 'lipsync', 'Wav2Lip'),
]

const cap = (c: Partial<Capability> & Pick<Capability, 'id' | 'kind' | 'phase' | 'name' | 'stage'>): Capability => ({
  summary: `${c.name} summary`, slot: 'post', order: 0, tier: 'advanced', needs: [], uses: [],
  requires: '', legacy_field: '', default_provider: '', params: [], ...c,
})

/** A four-capability slice of the real map — one per interesting shape (core+legacy, non-legacy
 *  with params, the lip-sync special case, and a deferred one). */
export const capabilityMap: CapabilityMap = {
  phases: {
    A: 'Source analysis · audio', B: 'Source analysis · picture', C: 'Text transformation',
    D: 'Speech generation', E: 'Audio post-production', F: 'Picture generation',
    G: 'Automatic quality judges',
  },
  capabilities: [
    cap({
      id: 'A1', kind: 'separation', phase: 'A', name: 'Dialogue / M&E separation', stage: 'separate',
      slot: 'inline', tier: 'core', legacy_field: 'separation', default_provider: 'separation.passthrough',
    }),
    cap({
      id: 'A3', kind: 'region_detect', phase: 'A', name: 'Speech / music / singing regions',
      stage: 'analyze', tier: 'recommended', needs: ['A1'],
      params: [{
        key: 'song_action', label: 'Song regions', type: 'select', default: 'passthrough',
        options: ['passthrough', 'dub', 'flag'], placeholder: '', help: '',
      }],
    }),
    cap({
      id: 'F1', kind: 'lipsync', phase: 'F', name: 'Live-action lip sync', stage: 'lipsync',
      slot: 'inline', tier: 'recommended', requires: 'lipsync', legacy_field: 'lipsync',
      default_provider: 'lipsync.none',
    }),
    cap({
      id: 'E5', kind: 'provenance', phase: 'E', name: 'Watermark + provenance', stage: 'render',
      tier: 'deferred',
    }),
  ],
  presets: [
    { id: 'minimal', description: "Today's pipeline plus the free wins. No extra models.", capabilities: ['A1'] },
    { id: 'balanced', description: 'Adds everything that raises quality without a human in the loop.', capabilities: ['A1', 'A3'] },
    { id: 'max', description: 'Every capability that is ready. Slow; may need cloud keys.', capabilities: ['A1', 'A3', 'F1'] },
  ],
}

const resolvedCap = (
  id: string, enabled: boolean, providerId: string, extra: Partial<ResolvedCapability> = {},
): ResolvedCapability => ({ id, enabled, provider_id: providerId, options: {}, params: {}, reason: '', ...extra })

export const resolvedCapabilities: Record<string, ResolvedCapability> = {
  A1: resolvedCap('A1', true, 'separation.passthrough', { reason: 'core capability — always on' }),
  A3: resolvedCap('A3', false, 'region_detect.off', { params: { song_action: 'passthrough' } }),
  F1: resolvedCap('F1', false, 'lipsync.none', { reason: "lip sync provider is 'none'" }),
  E5: resolvedCap('E5', false, 'provenance.off'),
}
