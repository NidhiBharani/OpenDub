// App-wide zustand store: data, playback/timeline state, and actions.
import { create } from 'zustand'
import { api, subscribeProjectEvents } from '../api/client'
import type { PresetBody } from '../api/client'
import type {
  AnchorSet, CapabilityChoice, CapabilityMap, Filmstrip, Job, LegacyKind, PipelineMode, Project,
  ProjectSummary, ProviderInfo, ResolvedCapability, Segment, StageKey, TimeRange, Version,
  Waveform,
} from '../types'

export type View = 'library' | 'editor' | 'settings'
export type AudioTrack = 'original' | 'dub'
export type ThemeName = 'dark' | 'light'
export type SidebarTab = 'script' | 'versions' | 'cast'
export type ClipHeight = 'S' | 'M' | 'L'
/** A time span selected on the timeline (I/O points); null when nothing is ranged. */
export interface Range { start: number; end: number }

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
  /** The 41 capabilities, their phases and the presets — static, loaded once. */
  capabilityMap: CapabilityMap | null
  /** What the current project's pipeline resolves to, keyed by capability id. */
  resolvedCapabilities: Record<string, ResolvedCapability>
  jobs: Record<string, Job>
  waveforms: Partial<Record<'original' | 'vocals' | 'dub_mix', Waveform>>
  /** Suggested cut points for the open project (null until loaded / unavailable). */
  anchors: AnchorSet | null
  filmstrip: Filmstrip | null
  versions: Version[]
  /** Version whose dubbed playback the viewer is previewing (read-only), or null for the live state. */
  previewVersionId: string | null

  // editor state
  sidebarTab: SidebarTab
  setSidebarTab: (t: SidebarTab) => void
  selection: string | null // segment id
  /** Selected skip range id (mutually exclusive with a segment selection in the inspector). */
  skipSelection: string | null
  range: Range | null
  clipHeight: ClipHeight
  setClipHeight: (h: ClipHeight) => void
  /** Anchors below this confidence are hidden on the scene strip (0..1). */
  anchorThreshold: number
  setAnchorThreshold: (v: number) => void
  snapping: boolean
  setSnapping: (on: boolean) => void
  /** Which row/field the script editor is editing; null when browsing. */
  editing: { segmentId: string; field: 'source' | 'translation' } | null
  setEditing: (e: { segmentId: string; field: 'source' | 'translation' } | null) => void
  playhead: number // seconds; written continuously by the Player
  playing: boolean
  /** playback rate; negative values are not supported by <video>, so J/K/L shuttle stays >= 0.25 */
  rate: number
  setRate: (r: number) => void
  loopRange: boolean
  setLoopRange: (on: boolean) => void
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
  loadCapabilityMap: () => Promise<void>
  loadResolvedCapabilities: () => Promise<void>
  openProject: (pid: string) => Promise<void>
  closeProject: () => void
  refreshProject: () => Promise<void>
  loadWaveforms: () => Promise<void>
  applyProject: (p: Project) => void
  applyJob: (j: Job) => void
  updateSegment: (sid: string, patch: Parameters<typeof api.updateSegment>[2]) => Promise<void>
  renameSpeaker: (spid: string, name: string) => Promise<void>
  selectSegment: (sid: string | null, seek?: boolean) => void
  selectSkipRange: (rid: string | null) => void
  splitSegment: (sid: string, at: number) => Promise<void>
  mergeSegmentWithNext: (sid: string) => Promise<void>
  deleteSegment: (sid: string) => Promise<void>

  // actions — skip ranges / anchors / range selection
  setRange: (r: Range | null) => void
  setRangeIn: (t: number) => void
  setRangeOut: (t: number) => void
  /** Persist a skip range for the current range selection (or an explicit span). */
  addSkipRange: (span?: Range, label?: string) => Promise<void>
  updateSkipRange: (rid: string, patch: Partial<Pick<TimeRange, 'start' | 'end' | 'label'>>) => Promise<void>
  removeSkipRange: (rid: string) => Promise<void>
  loadAnchors: (refresh?: boolean) => Promise<void>
  loadFilmstrip: () => Promise<void>

  // actions — versions
  loadVersions: () => Promise<void>
  snapshotVersion: (label?: string) => Promise<void>
  renameVersion: (vid: string, label: string) => Promise<void>
  deleteVersion: (vid: string) => Promise<void>
  restoreVersion: (vid: string) => Promise<void>
  setPreviewVersion: (vid: string | null) => void

  // actions — pipeline configuration
  /** Simple mode: write concrete choices for every capability from a preset + runtime. */
  applyPreset: (body: PresetBody) => Promise<void>
  /** Advanced mode: patch one capability's choice (merged over what it resolves to today). */
  setCapability: (capId: string, patch: Partial<CapabilityChoice>) => Promise<void>
  /** Advanced mode: pick the provider for one of the six legacy kinds. */
  setLegacyProvider: (kind: LegacyKind, providerId: string) => Promise<void>
  setPipelineMode: (mode: PipelineMode) => Promise<void>

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

/** Identity of the configuration the cached `resolvedCapabilities` were computed from. */
let resolvedKey = ''
const pipelineKey = (p: Project | null): string =>
  p ? `${p.id}:${JSON.stringify(p.pipeline)}` : ''

const theme0 = initialTheme()
applyTheme(theme0)

const isActive = (j: Job): boolean => j.status === 'queued' || j.status === 'running'
/** A run worth the full progress view: several stages, not a one-chip re-run or the upload ingest. */
const isFullRun = (j: Job): boolean => j.kind === 'pipeline' && j.stages.length > 1
/** How far along a job's lifecycle a status is; a job never moves backwards. */
const statusRank = (j: Job): number => (j.status === 'queued' ? 0 : j.status === 'running' ? 1 : 2)

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
  capabilityMap: null,
  resolvedCapabilities: {},
  jobs: {},
  waveforms: {},
  anchors: null,
  filmstrip: null,
  versions: [],
  previewVersionId: null,

  sidebarTab: 'script',
  setSidebarTab: (sidebarTab) => set({ sidebarTab }),
  selection: null,
  skipSelection: null,
  range: null,
  clipHeight: 'M',
  setClipHeight: (clipHeight) => set({ clipHeight }),
  anchorThreshold: 0.35,
  setAnchorThreshold: (v) => set({ anchorThreshold: Math.max(0, Math.min(1, v)) }),
  snapping: true,
  setSnapping: (snapping) => set({ snapping }),
  editing: null,
  setEditing: (editing) => set({ editing }),
  playhead: 0,
  playing: false,
  rate: 1,
  setRate: (r) => set({ rate: Math.max(0.25, Math.min(4, r)) }),
  loopRange: false,
  setLoopRange: (loopRange) => set({ loopRange }),
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

  // The capability map is static: load it once and keep it. Failures are silent — every other
  // view works without it, and Settings says so on its own.
  loadCapabilityMap: async () => {
    if (get().capabilityMap) return
    try {
      set({ capabilityMap: await api.getCapabilities() })
    } catch { /* offline or an older server */ }
  },

  loadResolvedCapabilities: async () => {
    const project = get().project
    if (!project) {
      resolvedKey = ''
      set({ resolvedCapabilities: {} })
      return
    }
    resolvedKey = pipelineKey(project)
    try {
      const resolved = await api.getProjectCapabilities(project.id)
      // The project may have been swapped out while the request was in flight.
      if (get().project?.id === project.id) set({ resolvedCapabilities: resolved })
    } catch { /* offline or an older server */ }
  },

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
      skipSelection: null,
      range: null,
      editing: null,
      sidebarTab: 'script',
      previewVersionId: null,
      playhead: 0,
      playing: false,
      audioTrack: 'original',
      zoom: 30,
      scrollX: 0,
      waveforms: {},
      anchors: null,
      filmstrip: null,
      versions: [],
      resolvedCapabilities: {},
    })
    void get().loadWaveforms()
    void get().loadResolvedCapabilities()
    void get().loadAnchors()
    void get().loadFilmstrip()
    void get().loadVersions()
  },

  closeProject: () => {
    unsubscribe?.()
    unsubscribe = null
    resolvedKey = ''
    set({
      project: null, view: 'library', jobs: {}, selection: null, skipSelection: null, range: null,
      editing: null, playing: false, runViewOpen: false, resolvedCapabilities: {}, anchors: null,
      filmstrip: null, versions: [], previewVersionId: null,
    })
    void get().loadProjects()
  },

  refreshProject: async () => {
    const pid = get().project?.id
    if (pid) get().applyProject(await api.getProject(pid))
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
    if (get().project?.id !== p.id) return
    set({ project: p })
    // Anything that touches the pipeline (here, another tab, a preset) changes what will run.
    if (pipelineKey(p) !== resolvedKey) void get().loadResolvedCapabilities()
  },

  applyJob: (j) => {
    // A fast job's SSE events (running → done) can land before the POST that started it
    // resolves; that response still says "queued" and must not overwrite the newer state.
    const known = get().jobs[j.id]
    if (known && statusRank(j) < statusRank(known)) return
    set((s) => ({
      jobs: { ...s.jobs, [j.id]: j },
      // a run we have not seen before puts its progress view up front
      runViewOpen: s.runViewOpen || (!s.jobs[j.id] && isActive(j) && isFullRun(j)),
    }))
    if (j.status === 'done') {
      // stage outputs (waveforms, mixes) may have changed
      void get().loadWaveforms()
      if (isFullRun(j) || j.kind === 'restore') void get().loadVersions()
      if (j.stages.includes('ingest') || j.stages.includes('transcribe')) void get().loadAnchors()
      if (j.stages.includes('ingest')) void get().loadFilmstrip()
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

  applyPreset: async (body) => {
    const project = get().project
    if (!project) return
    const { project: updated, notes } = await api.applyPreset(project.id, body)
    get().applyProject(updated)
    // Fallbacks the server had to make (no key, no GPU provider installed) are the whole point
    // of the notes — surface every one of them rather than silently running something else.
    for (const note of notes) get().toast('error', note)
  },

  setCapability: async (capId, patch) => {
    const project = get().project
    if (!project) return
    // The server replaces the whole CapabilityChoice, so send a full one: the stored choice if
    // there is one, otherwise what this capability currently resolves to.
    const stored = project.pipeline.capabilities?.[capId]
    const resolved = get().resolvedCapabilities[capId]
    const base: CapabilityChoice = stored ?? {
      enabled: resolved?.enabled ?? false,
      provider_id: resolved?.provider_id ?? '',
      options: resolved?.options ?? {},
      params: resolved?.params ?? {},
    }
    const updated = await api.updateProject(project.id, {
      capabilities: { [capId]: { ...base, ...patch } },
    })
    get().applyProject(updated)
  },

  setLegacyProvider: async (kind, providerId) => {
    const project = get().project
    if (!project) return
    // Sparse patch: the PATCH body's `pipeline` accepts legacy kinds only, so never spread the
    // whole PipelineConfig (mode/preset/capabilities are not provider kinds).
    const updated = await api.updateProject(project.id, {
      pipeline: { [kind]: { provider_id: providerId, options: {} } },
    })
    get().applyProject(updated)
  },

  setPipelineMode: async (mode) => {
    const project = get().project
    if (!project || project.pipeline.mode === mode) return
    get().applyProject(await api.updateProject(project.id, { mode }))
  },

  selectSegment: (sid, doSeek = false) => {
    set({ selection: sid, skipSelection: sid ? null : get().skipSelection })
    if (sid && doSeek) {
      const seg = get().project?.segments.find((s) => s.id === sid)
      if (seg) get().seek(seg.start)
    }
  },
  selectSkipRange: (rid) => set({ skipSelection: rid, selection: rid ? null : get().selection }),

  splitSegment: async (sid, at) => {
    const pid = get().project?.id
    if (!pid) return
    get().applyProject(await api.splitSegment(pid, sid, at))
  },
  mergeSegmentWithNext: async (sid) => {
    const pid = get().project?.id
    if (!pid) return
    get().applyProject(await api.mergeSegmentWithNext(pid, sid))
    set({ selection: sid })
  },
  deleteSegment: async (sid) => {
    const pid = get().project?.id
    if (!pid) return
    get().applyProject(await api.deleteSegment(pid, sid))
    if (get().selection === sid) set({ selection: null })
  },

  setRange: (range) => set({ range: range && range.end > range.start ? range : null }),
  setRangeIn: (t) => {
    const dur = get().project?.media?.duration ?? Infinity
    const cur = get().range
    const start = Math.max(0, Math.min(t, dur))
    const end = cur && cur.end > start ? cur.end : Math.min(dur, start + 1)
    set({ range: { start, end } })
  },
  setRangeOut: (t) => {
    const dur = get().project?.media?.duration ?? Infinity
    const cur = get().range
    const end = Math.max(0, Math.min(t, dur))
    const start = cur && cur.start < end ? cur.start : Math.max(0, end - 1)
    set({ range: { start, end } })
  },

  addSkipRange: async (span, label = '') => {
    const project = get().project
    const r = span ?? get().range
    if (!project || !r || r.end <= r.start) return
    const next: TimeRange = { id: `tmp_${Date.now().toString(36)}`, start: r.start, end: r.end, label }
    const updated = await api.setSkipRanges(project.id, [...project.skip_ranges, next])
    get().applyProject(updated)
    // the server assigns the real id: select the range that covers our span
    const saved = updated.skip_ranges.find((x) => Math.abs(x.start - r.start) < 1e-3 && Math.abs(x.end - r.end) < 1e-3)
    set({ range: null, skipSelection: saved?.id ?? null, selection: null })
  },
  updateSkipRange: async (rid, patch) => {
    const project = get().project
    if (!project) return
    const ranges = project.skip_ranges.map((x) => (x.id === rid ? { ...x, ...patch } : x))
    get().applyProject(await api.setSkipRanges(project.id, ranges))
  },
  removeSkipRange: async (rid) => {
    const project = get().project
    if (!project) return
    get().applyProject(await api.setSkipRanges(project.id, project.skip_ranges.filter((x) => x.id !== rid)))
    if (get().skipSelection === rid) set({ skipSelection: null })
  },
  loadAnchors: async (refresh = false) => {
    const p = get().project
    if (!p || !p.media) return
    try {
      const anchors = await api.getAnchors(p.id, refresh)
      if (get().project?.id === p.id) set({ anchors })
    } catch { /* not ingested yet, or an older server */ }
  },
  loadFilmstrip: async () => {
    const p = get().project
    if (!p || !p.media) return
    try {
      const filmstrip = await api.getFilmstrip(p.id)
      if (get().project?.id === p.id) set({ filmstrip })
    } catch { /* not ingested yet, or an older server */ }
  },

  loadVersions: async () => {
    const p = get().project
    if (!p) return
    try {
      const versions = await api.listVersions(p.id)
      if (get().project?.id === p.id) set({ versions })
    } catch { /* older server */ }
  },
  snapshotVersion: async (label) => {
    const p = get().project
    if (!p) return
    await api.createVersion(p.id, label)
    await get().loadVersions()
    await get().refreshProject()
  },
  renameVersion: async (vid, label) => {
    const p = get().project
    if (!p) return
    const v = await api.renameVersion(p.id, vid, label)
    set((s) => ({ versions: s.versions.map((x) => (x.id === vid ? v : x)) }))
  },
  deleteVersion: async (vid) => {
    const p = get().project
    if (!p) return
    await api.deleteVersion(p.id, vid)
    set((s) => ({
      versions: s.versions.filter((x) => x.id !== vid),
      previewVersionId: s.previewVersionId === vid ? null : s.previewVersionId,
    }))
  },
  restoreVersion: async (vid) => {
    const p = get().project
    if (!p) return
    const updated = await api.restoreVersion(p.id, vid)
    get().applyProject(updated)
    set({ previewVersionId: null, selection: null, skipSelection: null })
    void get().loadWaveforms()
    void get().loadVersions()
  },
  setPreviewVersion: (previewVersionId) => set({ previewVersionId, playing: false }),

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
