// Typed REST client + SSE subscription. All server communication goes through this module.
import type {
  Job, Project, ProjectSummary, ProviderInfo, Segment, Speaker, StageKey, Waveform,
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

export const api = {
  // projects
  listProjects: () => request<ProjectSummary[]>('/projects'),
  getProject: (pid: string) => request<Project>(`/projects/${pid}`),
  deleteProject: (pid: string) => request<{ ok: boolean }>(`/projects/${pid}`, { method: 'DELETE' }),
  updateProject: (pid: string, patch: Partial<Pick<Project, 'name' | 'source_lang' | 'target_lang' | 'pipeline'>>) =>
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

  // pipeline / jobs
  runPipeline: (pid: string, stages?: StageKey[]) =>
    request<Job>(`/projects/${pid}/pipeline/run`, { method: 'POST', ...json(stages ? { stages } : {}) }),
  cancelJob: (jobId: string) => request<Job>(`/jobs/${jobId}/cancel`, { method: 'POST' }),
  listJobs: (pid?: string) => request<Job[]>(`/jobs${pid ? `?project_id=${pid}` : ''}`),

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
