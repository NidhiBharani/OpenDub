// Typed REST client + SSE subscription. All server communication goes through this module.
import type {
  AnchorSet, CapabilityChoice, CapabilityMap, Filmstrip, Job, LegacyKind, PipelineMode,
  PresetName, Project, ProjectSummary, ProviderChoice, ProviderInfo, ResolvedCapability,
  RuntimeName, Segment, Speaker, StageKey, TimeRange, Version, Waveform,
} from '../types'

const BASE = '/api'

class ApiError extends Error {
  constructor(public status: number, detail: string) {
    super(detail)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* not json */ }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

const json = (body: unknown): RequestInit => ({
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

/** PATCH /api/projects/{pid} body. `pipeline` carries only the legacy kinds (sparse); every other
 *  capability is configured through `capabilities`, keyed by capability id. */
export interface ProjectPatch {
  name?: string
  source_lang?: string
  target_lang?: string
  pipeline?: Partial<Record<LegacyKind, ProviderChoice>>
  mode?: PipelineMode
  capabilities?: Record<string, CapabilityChoice>
}

/** POST /api/projects/{pid}/pipeline/preset body. */
export interface PresetBody {
  preset: PresetName
  runtime: RuntimeName
  lipsync?: boolean
  subtitles?: boolean
}

export const api = {
  // projects
  listProjects: () => request<ProjectSummary[]>('/projects'),
  getProject: (pid: string) => request<Project>(`/projects/${pid}`),
  deleteProject: (pid: string) => request<{ ok: boolean }>(`/projects/${pid}`, { method: 'DELETE' }),
  updateProject: (pid: string, patch: ProjectPatch) =>
    request<Project>(`/projects/${pid}`, { method: 'PATCH', ...json(patch) }),
  createProject: (file: File, name: string, sourceLang: string, targetLang: string, onProgress?: (frac: number) => void) =>
    uploadProject(file, name, sourceLang, targetLang, onProgress),

  // segments / speakers
  updateSegment: (pid: string, sid: string, patch: Partial<Pick<Segment,
    'source_text' | 'translated_text' | 'speaker_id' | 'start' | 'end' | 'emotion' | 'active_take_id' | 'notes'>>) =>
    request<Segment>(`/projects/${pid}/segments/${sid}`, { method: 'PATCH', ...json(patch) }),
  updateSpeaker: (pid: string, spid: string, patch: Partial<Pick<Speaker, 'name' | 'color'>>) =>
    request<Speaker>(`/projects/${pid}/speakers/${spid}`, { method: 'PATCH', ...json(patch) }),
  regenerateSegment: (pid: string, sid: string, stages: ('translate' | 'synthesize')[]) =>
    request<Job>(`/projects/${pid}/segments/${sid}/regenerate`, { method: 'POST', ...json({ stages }) }),
  /** Split one segment at time `at` (seconds, strictly inside it). Words and text are divided at
   *  the word boundary nearest to `at`. Returns the whole project (two segments replace one). */
  splitSegment: (pid: string, sid: string, at: number) =>
    request<Project>(`/projects/${pid}/segments/${sid}/split`, { method: 'POST', ...json({ at }) }),
  /** Merge a segment with the next one in time order (same or different speaker). */
  mergeSegmentWithNext: (pid: string, sid: string) =>
    request<Project>(`/projects/${pid}/segments/${sid}/merge-next`, { method: 'POST' }),
  createSegment: (pid: string, body: { start: number; end: number; speaker_id?: string; source_text?: string }) =>
    request<Project>(`/projects/${pid}/segments`, { method: 'POST', ...json(body) }),
  deleteSegment: (pid: string, sid: string) =>
    request<Project>(`/projects/${pid}/segments/${sid}`, { method: 'DELETE' }),
  /** Re-run speech recognition over just this segment's time range (a segment job). */
  transcribeSegment: (pid: string, sid: string) =>
    request<Job>(`/projects/${pid}/segments/${sid}/transcribe`, { method: 'POST' }),

  // skip ranges (keep the original in these spans) + suggested anchors
  setSkipRanges: (pid: string, ranges: TimeRange[]) =>
    request<Project>(`/projects/${pid}/skip-ranges`, { method: 'PUT', ...json({ ranges }) }),
  /** Suggested cut points (scene changes, silences, segment edges). Computed on first call and
   *  cached on disk; `refresh` recomputes. */
  getAnchors: (pid: string, refresh = false) =>
    request<AnchorSet>(`/projects/${pid}/anchors${refresh ? '?refresh=1' : ''}`),
  getFilmstrip: (pid: string) => request<Filmstrip>(`/projects/${pid}/filmstrip`),

  // versions
  listVersions: (pid: string) => request<Version[]>(`/projects/${pid}/versions`),
  createVersion: (pid: string, label?: string) =>
    request<Version>(`/projects/${pid}/versions`, { method: 'POST', ...json(label ? { label } : {}) }),
  renameVersion: (pid: string, vid: string, label: string) =>
    request<Version>(`/projects/${pid}/versions/${vid}`, { method: 'PATCH', ...json({ label }) }),
  deleteVersion: (pid: string, vid: string) =>
    request<{ ok: boolean }>(`/projects/${pid}/versions/${vid}`, { method: 'DELETE' }),
  /** Restore a version as the current state (refused while a job runs). The server snapshots the
   *  current state first as a "pre_restore" version. */
  restoreVersion: (pid: string, vid: string) =>
    request<Project>(`/projects/${pid}/versions/${vid}/restore`, { method: 'POST' }),

  // pipeline / jobs
  runPipeline: (pid: string, stages?: StageKey[]) =>
    request<Job>(`/projects/${pid}/pipeline/run`, { method: 'POST', ...json(stages ? { stages } : {}) }),
  cancelJob: (jobId: string) => request<Job>(`/jobs/${jobId}/cancel`, { method: 'POST' }),
  listJobs: (pid?: string) => request<Job[]>(`/jobs${pid ? `?project_id=${pid}` : ''}`),

  // capability map
  getCapabilities: () => request<CapabilityMap>('/capabilities'),
  getProjectCapabilities: (pid: string) =>
    request<Record<string, ResolvedCapability>>(`/projects/${pid}/capabilities`),
  applyPreset: (pid: string, body: PresetBody) =>
    request<{ project: Project; notes: string[] }>(
      `/projects/${pid}/pipeline/preset`, { method: 'POST', ...json(body) },
    ),

  // providers / settings
  listProviders: () => request<ProviderInfo[]>('/providers'),
  saveProviderOptions: (providerId: string, options: Record<string, unknown>) =>
    request<ProviderInfo>(`/settings/providers/${providerId}`, { method: 'PUT', ...json({ options }) }),
  checkProvider: (providerId: string) =>
    request<{ available: boolean; reason: string }>(`/providers/${providerId}/check`, { method: 'POST' }),

  // media
  waveform: (pid: string, asset: 'original' | 'vocals' | 'dub_mix') =>
    request<Waveform>(`/projects/${pid}/waveform/${asset}`),
  mediaUrl: (pid: string, relPath: string) => `${BASE}/media/${pid}/${relPath}`,
}

function uploadProject(
  file: File, name: string, sourceLang: string, targetLang: string,
  onProgress?: (frac: number) => void,
): Promise<Project> {
  // XHR for upload progress events (fetch has no standard upload progress).
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)
    form.append('name', name)
    form.append('source_lang', sourceLang)
    form.append('target_lang', targetLang)
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/projects`)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText))
      else {
        let detail = xhr.statusText
        try { detail = JSON.parse(xhr.responseText).detail ?? detail } catch { /* ignore */ }
        reject(new ApiError(xhr.status, detail))
      }
    }
    xhr.onerror = () => reject(new ApiError(0, 'network error'))
    xhr.send(form)
  })
}

export type ProjectEvent =
  | { type: 'job'; job: Job }
  | { type: 'project'; project: Project }

/** Subscribe to a project's SSE stream. Returns an unsubscribe function.
 *
 * `onConnect` fires on every successful (re)connect. EventSource auto-reconnects after a server
 * restart, but events emitted while disconnected are lost forever — callers must treat each
 * reconnect as "my state may be stale" and re-fetch, or the editor silently diverges (e.g. a mix
 * that finished during a backend restart never shows up, so the dub track never mounts). */
export function subscribeProjectEvents(
  pid: string,
  onEvent: (e: ProjectEvent) => void,
  onConnect?: () => void,
): () => void {
  const es = new EventSource(`${BASE}/projects/${pid}/events`)
  if (onConnect) es.onopen = () => onConnect()
  es.addEventListener('job', (e) => onEvent({ type: 'job', job: JSON.parse((e as MessageEvent).data) }))
  es.addEventListener('project', (e) => onEvent({ type: 'project', project: JSON.parse((e as MessageEvent).data) }))
  return () => es.close()
}

export { ApiError }
