// App-wide zustand store: data, playback/timeline state, and actions.
import { create } from 'zustand'
import { api, subscribeProjectEvents } from '../api/client'
import type { Job, Project, ProjectSummary, ProviderInfo, Segment, StageKey, Waveform } from '../types'

export type View = 'library' | 'editor' | 'settings'
export type AudioTrack = 'original' | 'dub'

export interface Toast {
  id: number
  kind: 'info' | 'error' | 'success'
  text: string
}

interface AppState {
  // navigation
  view: View
  setView: (v: View) => void

  // data
  projects: ProjectSummary[]
  project: Project | null
  providers: ProviderInfo[]
  jobs: Record<string, Job>
  waveforms: Partial<Record<'original' | 'vocals' | 'dub_mix', Waveform>>

  // editor state
  selection: string | null // segment id
  playhead: number // seconds; written continuously by the Player
  playing: boolean
  audioTrack: AudioTrack
  zoom: number // px per second (clamped 2..500)
  scrollX: number // timeline: seconds at left edge
  seekRequest: { t: number; nonce: number } | null // Player consumes; set via seek()

  // toasts
  toasts: Toast[]
  toast: (kind: Toast['kind'], text: string) => void
  dismissToast: (id: number) => void

  // actions — data
  loadProjects: () => Promise<void>
  loadProviders: () => Promise<void>
  openProject: (pid: string) => Promise<void>
  closeProject: () => void
  refreshProject: () => Promise<void>
  loadWaveforms: () => Promise<void>
  applyProject: (p: Project) => void
  applyJob: (j: Job) => void
  updateSegment: (sid: string, patch: Parameters<typeof api.updateSegment>[2]) => Promise<void>
  selectSegment: (sid: string | null, seek?: boolean) => void

  // actions — playback/timeline
  setPlayhead: (t: number) => void
  seek: (t: number) => void
  setPlaying: (p: boolean) => void
  setAudioTrack: (t: AudioTrack) => void
  setZoom: (z: number, anchorT?: number, anchorPx?: number) => void
  setScrollX: (s: number) => void
}

let unsubscribe: (() => void) | null = null
let toastSeq = 1
let seekSeq = 1

export const useStore = create<AppState>((set, get) => ({
  view: 'library',
  setView: (view) => set({ view }),

  projects: [],
  project: null,
  providers: [],
  jobs: {},
  waveforms: {},

  selection: null,
  playhead: 0,
  playing: false,
  audioTrack: 'original',
  zoom: 30,
  scrollX: 0,
  seekRequest: null,

  toasts: [],
  toast: (kind, text) => {
    const id = toastSeq++
    set((s) => ({ toasts: [...s.toasts, { id, kind, text }] }))
    setTimeout(() => get().dismissToast(id), kind === 'error' ? 6000 : 3500)
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  loadProjects: async () => set({ projects: await api.listProjects() }),
  loadProviders: async () => set({ providers: await api.listProviders() }),

  openProject: async (pid) => {
    const project = await api.getProject(pid)
    unsubscribe?.()
    unsubscribe = subscribeProjectEvents(pid, (e) => {
      if (e.type === 'job') get().applyJob(e.job)
      else if (e.type === 'project') get().applyProject(e.project)
    })
    const jobs = await api.listJobs(pid)
    set({
      project,
      view: 'editor',
      jobs: Object.fromEntries(jobs.map((j) => [j.id, j])),
      selection: null,
      playhead: 0,
      playing: false,
      audioTrack: 'original',
      zoom: 30,
      scrollX: 0,
      waveforms: {},
    })
    void get().loadWaveforms()
  },

  closeProject: () => {
    unsubscribe?.()
    unsubscribe = null
    set({ project: null, view: 'library', jobs: {}, selection: null, playing: false })
    void get().loadProjects()
  },

  refreshProject: async () => {
    const pid = get().project?.id
    if (pid) set({ project: await api.getProject(pid) })
  },

  loadWaveforms: async () => {
    const p = get().project
    if (!p) return
    for (const asset of ['original', 'vocals', 'dub_mix'] as const) {
      try {
        const wf = await api.waveform(p.id, asset)
        set((s) => ({ waveforms: { ...s.waveforms, [asset]: wf } }))
      } catch { /* asset not produced yet */ }
    }
  },

  applyProject: (p) => {
    if (get().project?.id === p.id) set({ project: p })
  },

  applyJob: (j) => {
    set((s) => ({ jobs: { ...s.jobs, [j.id]: j } }))
    if (j.status === 'done') {
      // stage outputs (waveforms, mixes) may have changed
      void get().loadWaveforms()
    }
  },

  updateSegment: async (sid, patch) => {
    const p = get().project
    if (!p) return
    const seg = await api.updateSegment(p.id, sid, patch)
    set({
      project: {
        ...p,
        segments: p.segments.map((s) => (s.id === sid ? seg : s)),
      },
    })
  },

  selectSegment: (sid, doSeek = false) => {
    set({ selection: sid })
    if (sid && doSeek) {
      const seg = get().project?.segments.find((s) => s.id === sid)
      if (seg) get().seek(seg.start)
    }
  },

  setPlayhead: (t) => set({ playhead: t }),
  seek: (t) => {
    const dur = get().project?.media?.duration ?? Infinity
    const clamped = Math.max(0, Math.min(t, dur))
    set({ seekRequest: { t: clamped, nonce: seekSeq++ }, playhead: clamped })
  },
  setPlaying: (playing) => set({ playing }),
  setAudioTrack: (audioTrack) => set({ audioTrack }),

  setZoom: (z, anchorT, anchorPx) => {
    const zoom = Math.max(2, Math.min(500, z))
    if (anchorT !== undefined && anchorPx !== undefined) {
      // keep the time under the cursor fixed while zooming
      set({ zoom, scrollX: Math.max(0, anchorT - anchorPx / zoom) })
    } else {
      set({ zoom })
    }
  },
  setScrollX: (s) => set({ scrollX: Math.max(0, s) }),
}))

/** Selector helpers */
export const useProject = () => useStore((s) => s.project)
export const useSelectedSegment = (): Segment | null =>
  useStore((s) => s.project?.segments.find((seg) => seg.id === s.selection) ?? null)
export const useActiveJob = (): Job | null =>
  useStore((s) => {
    const active = Object.values(s.jobs).filter((j) => j.status === 'queued' || j.status === 'running')
    return active.sort((a, b) => b.created_at.localeCompare(a.created_at))[0] ?? null
  })

export function stageStatus(project: Project | null, key: StageKey) {
  return project?.stages?.[key]?.status ?? 'pending'
}
