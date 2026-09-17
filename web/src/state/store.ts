// App-wide zustand store: data, playback/timeline state, and actions.
import { create } from 'zustand'
import { api, subscribeProjectEvents } from '../api/client'
import type { Job, Project, ProjectSummary, ProviderInfo, Segment, StageKey, Waveform } from '../types'

export type View = 'library' | 'editor' | 'settings'
export type AudioTrack = 'original' | 'dub'
export type ThemeName = 'dark' | 'light'

const THEME_KEY = 'opendub.theme'

function initialTheme(): ThemeName {
  try {
    const saved = localStorage.getItem(THEME_KEY)
    if (saved === 'light' || saved === 'dark') return saved
    if (window.matchMedia?.('(prefers-color-scheme: light)').matches) return 'light'
  } catch { /* storage blocked */ }
  return 'dark'
}

function applyTheme(theme: ThemeName): void {
  document.documentElement.dataset.theme = theme
}

export interface Toast {
  id: number
  kind: 'error' | 'success'
  text: string
}

interface AppState {
  // navigation
  view: View
  setView: (v: View) => void
  theme: ThemeName
  setTheme: (t: ThemeName) => void
  /** Full-screen live progress view over the editor; opens itself when a multi-stage run starts. */
  runViewOpen: boolean
  setRunViewOpen: (open: boolean) => void

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
  renameSpeaker: (spid: string, name: string) => Promise<void>
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

const theme0 = initialTheme()
applyTheme(theme0)

const isActive = (j: Job): boolean => j.status === 'queued' || j.status === 'running'
/** A run worth the full progress view: several stages, not a one-chip re-run or the upload ingest. */
const isFullRun = (j: Job): boolean => j.kind === 'pipeline' && j.stages.length > 1

export const useStore = create<AppState>((set, get) => ({
  view: 'library',
  setView: (view) => set({ view }),
  theme: theme0,
  setTheme: (theme) => {
    applyTheme(theme)
    try { localStorage.setItem(THEME_KEY, theme) } catch { /* storage blocked */ }
    set({ theme })
  },
  runViewOpen: false,
  setRunViewOpen: (runViewOpen) => set({ runViewOpen }),

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
    unsubscribe = subscribeProjectEvents(
      pid,
      (e) => {
        if (e.type === 'job') get().applyJob(e.job)
        else if (e.type === 'project') get().applyProject(e.project)
      },
      // Fires on every (re)connect: events during a backend restart are lost, so resync.
      () => {
        void get().refreshProject()
        void get().loadWaveforms()
      },
    )
    const jobs = await api.listJobs(pid)
    set({
      project,
      view: 'editor',
      jobs: Object.fromEntries(jobs.map((j) => [j.id, j])),
      runViewOpen: jobs.some((j) => isActive(j) && isFullRun(j)),
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
    set({ project: null, view: 'library', jobs: {}, selection: null, playing: false, runViewOpen: false })
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
    set((s) => ({
      jobs: { ...s.jobs, [j.id]: j },
      // a run we have not seen before puts its progress view up front
      runViewOpen: s.runViewOpen || (!s.jobs[j.id] && isActive(j) && isFullRun(j)),
    }))
    if (j.status === 'done') {
      // stage outputs (waveforms, mixes) may have changed
      void get().loadWaveforms()
    }
  },

  updateSegment: async (sid, patch) => {
    const pid = get().project?.id
    if (!pid) return
    const seg = await api.updateSegment(pid, sid, patch)
    // Merge into the *current* project (it may have been replaced by an SSE
    // event while the PATCH was in flight) — never a pre-request snapshot.
    set((s) =>
      s.project && s.project.id === pid
        ? {
            project: {
              ...s.project,
              segments: s.project.segments.map((x) => (x.id === sid ? seg : x)),
            },
          }
        : {},
    )
  },

  renameSpeaker: async (spid, name) => {
    const pid = get().project?.id
    if (!pid) return
    const speaker = await api.updateSpeaker(pid, spid, { name })
    set((s) =>
      s.project && s.project.id === pid
        ? { project: { ...s.project, speakers: s.project.speakers.map((x) => (x.id === spid ? speaker : x)) } }
        : {},
    )
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

/** Most recent multi-stage pipeline run (active or finished) — what the progress view shows. */
export const useLatestRun = (): Job | null =>
  useStore((s) => {
    const runs = Object.values(s.jobs).filter(isFullRun)
    return runs.sort((a, b) => b.created_at.localeCompare(a.created_at))[0] ?? null
  })

/** Speaker colour from the theme palette by cast order (server colours predate the themes). */
export function speakerColor(project: Project | null, speakerId: string | undefined): string {
  const idx = project?.speakers.findIndex((sp) => sp.id === speakerId) ?? -1
  return idx < 0 ? 'var(--text-faint)' : `var(--spk-${(idx % 8) + 1})`
}

export function stageStatus(project: Project | null, key: StageKey) {
  return project?.stages?.[key]?.status ?? 'pending'
}
