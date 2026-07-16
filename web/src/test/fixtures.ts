// Valid domain objects mirroring server/app/models.py, for use in component tests.
import type { Project, ProjectSummary, ProviderInfo } from '../types'

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
    transcribe: doneStage('5 segments, 1 speaker(s)'),
    translate: doneStage(),
    synthesize: doneStage(),
    mix: doneStage(),
    lipsync: { status: 'skipped', detail: '', updated_at: null },
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
      source_text: '[line 1]', translated_text: '(en) [line 1]', emotion: '', notes: '',
      takes: [], active_take_id: null, translate_dirty: false, synth_dirty: false,
    },
  ],
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
