// Library view: a project browser (docs/plans/editor-redesign.md §8). Toolbar (New project, search,
// sort, grid/list), a grid of 16:9 cards or a dense table, and drop-to-upload over the whole body.
import type { ChangeEvent, DragEvent, KeyboardEvent, ReactNode } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import { Button, IconButton, Modal, ProgressBar, Segmented, Select, TextInput } from '../components/primitives'
import { Icon } from '../components/Icon'
import { NavLink, TopBar } from '../components/TopBar'
import { useStore } from '../state/store'
import { LANGUAGES, STAGE_LABELS, STAGE_ORDER, formatTime } from '../types'
import type { ProjectSummary, StageKey, StageStatus } from '../types'

type ViewMode = 'grid' | 'list'
type SortKey = 'modified' | 'name' | 'duration'

const VIEW_KEY = 'opendub.library.view'
const TARGET_KEY = 'opendub.library.target'
const SOURCE_KEY = 'opendub.library.source'

const SOURCE_LANG_OPTIONS = [
  { value: 'auto', label: 'Detect automatically' },
  ...LANGUAGES.map((l) => ({ value: l.code, label: l.name })),
]
const TARGET_LANG_OPTIONS = LANGUAGES.map((l) => ({ value: l.code, label: l.name }))

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'modified', label: 'Modified' }, { value: 'name', label: 'Name' }, { value: 'duration', label: 'Duration' },
]

const STAGE_COLOR: Record<StageStatus, string> = {
  pending: 'var(--border-strong)',
  queued: 'var(--running)',
  running: 'var(--running)',
  done: 'var(--ok)',
  dirty: 'var(--warn)',
  error: 'var(--err)',
  skipped: 'var(--border-strong)',
}

function readStorage(key: string, fallback: string): string {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}
function writeStorage(key: string, value: string) {
  try { localStorage.setItem(key, value) } catch { /* private mode / disabled storage */ }
}

function stripExt(filename: string): string {
  const idx = filename.lastIndexOf('.')
  return idx > 0 ? filename.slice(0, idx) : filename
}

function formatSize(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function isProjectActive(p: ProjectSummary): boolean {
  return STAGE_ORDER.some((key) => p.stages?.[key]?.status === 'running')
}

/** Last stage completion, falling back to creation. */
function modifiedAt(p: ProjectSummary): number {
  let t = Date.parse(p.created_at) || 0
  for (const key of STAGE_ORDER) {
    const u = p.stages?.[key]?.updated_at
    if (u) t = Math.max(t, Date.parse(u) || 0)
  }
  return t
}

function formatDate(ms: number): string {
  if (!ms) return '—'
  const d = new Date(ms)
  const now = new Date()
  const sameYear = d.getFullYear() === now.getFullYear()
  return d.toLocaleDateString(undefined, sameYear ? { month: 'short', day: 'numeric' } : { year: 'numeric', month: 'short', day: 'numeric' })
}

/** The server's summary carries source_lang; the shared type does not (yet). */
function langPair(p: ProjectSummary): string {
  const src = (p as { source_lang?: string }).source_lang
  return `${src ? src.toUpperCase() : '??'} → ${p.target_lang.toUpperCase()}`
}

/** One human line for where a project stands, worst news first. */
export function projectStatus(p: ProjectSummary): { text: string; color: string } {
  const st = (k: StageKey): StageStatus => p.stages?.[k]?.status ?? 'pending'
  const failed = STAGE_ORDER.find((k) => st(k) === 'error')
  if (failed) {
    const detail = p.stages?.[failed]?.detail
    return { text: `${STAGE_LABELS[failed]} failed${detail ? ` · ${detail}` : ''}`, color: 'var(--err-text)' }
  }
  const active = STAGE_ORDER.find((k) => st(k) === 'running') ?? STAGE_ORDER.find((k) => st(k) === 'queued')
  if (active) return { text: `${STAGE_LABELS[active]} in progress…`, color: 'var(--accent-text)' }
  const dirty = STAGE_ORDER.find((k) => st(k) === 'dirty')
  if (dirty) return { text: `Edited · re-run from ${STAGE_LABELS[dirty]}`, color: 'var(--warn-text)' }
  if (st('render') === 'done') return { text: 'Rendered · dubbed.mp4 ready', color: 'var(--ok)' }
  if (st('synthesize') === 'done') return { text: 'Voiced · mix and render to finish', color: 'var(--text-dim)' }
  if (st('translate') === 'done') return { text: 'Translated · review the script before voicing', color: 'var(--text-dim)' }
  if (st('transcribe') === 'done') return { text: 'Transcribed · ready to translate', color: 'var(--text-dim)' }
  if (st('ingest') === 'done') return { text: 'Ingested · press Run pipeline to begin', color: 'var(--text-dim)' }
  return { text: 'Waiting for ingest', color: 'var(--text-dim)' }
}

export function Library() {
  const projects = useStore((s) => s.projects)
  const loadProjects = useStore((s) => s.loadProjects)
  const setView = useStore((s) => s.setView)
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortKey>('modified')
  const [viewMode, setViewMode] = useState<ViewMode>(() => (readStorage(VIEW_KEY, 'grid') === 'list' ? 'list' : 'grid'))
  const [dragDepth, setDragDepth] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    void loadProjects()
  }, [loadProjects])

  // Poll while any visible project is mid-stage — cheap, self-cancelling.
  useEffect(() => {
    const id = setInterval(() => {
      if (useStore.getState().projects.some(isProjectActive)) void loadProjects()
    }, 5000)
    return () => clearInterval(id)
  }, [loadProjects])

  function chooseView(v: ViewMode) {
    setViewMode(v)
    writeStorage(VIEW_KEY, v)
  }

  const q = query.trim().toLowerCase()
  const shown = useMemo(() => {
    const list = q ? projects.filter((p) => p.name.toLowerCase().includes(q)) : [...projects]
    list.sort((a, b) => {
      if (sort === 'name') return a.name.localeCompare(b.name)
      if (sort === 'duration') return b.duration - a.duration
      return modifiedAt(b) - modifiedAt(a)
    })
    return list
  }, [projects, q, sort])
  const running = projects.filter(isProjectActive).length

  function handleInputChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) setPendingFile(file)
    e.target.value = ''
  }

  function hasFiles(e: DragEvent) {
    return Array.from(e.dataTransfer?.types ?? []).includes('Files')
  }
  function onDragEnter(e: DragEvent<HTMLDivElement>) {
    if (!hasFiles(e)) return
    e.preventDefault()
    setDragDepth((d) => d + 1)
  }
  function onDragOver(e: DragEvent<HTMLDivElement>) {
    if (!hasFiles(e)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }
  function onDragLeave(e: DragEvent<HTMLDivElement>) {
    if (!hasFiles(e)) return
    setDragDepth((d) => Math.max(0, d - 1))
  }
  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragDepth(0)
    const file = e.dataTransfer.files?.[0]
    if (file) setPendingFile(file)
  }
  const dragging = dragDepth > 0

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <TopBar>
        <NavLink label="Library" active onClick={() => setView('library')} />
        <NavLink label="Settings" onClick={() => setView('settings')} />
        <div style={{ flex: 1 }} />
        {running > 0 && (
          <span className="timecode" style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', background: 'var(--running)',
              animation: 'od-pulse 1200ms ease-in-out infinite',
            }} />
            {running} {running === 1 ? 'job' : 'jobs'} running
          </span>
        )}
      </TopBar>

      <div
        role="toolbar"
        aria-label="Library"
        style={{
          height: 36, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 8, padding: '0 12px',
          background: 'var(--bg-raised)', borderBottom: '1px solid var(--border)',
        }}
      >
        <Button variant="primary" onClick={() => inputRef.current?.click()} title="New project from a video file">
          <Icon name="upload" size={13} />
          New project
        </Button>
        <div style={{ position: 'relative', width: 220 }}>
          <Icon name="search" size={13} style={{
            position: 'absolute', left: 7, top: 6, color: 'var(--text-faint)', pointerEvents: 'none',
          }} />
          <TextInput
            value={query}
            onChange={setQuery}
            placeholder="Search projects"
            ariaLabel="Search projects"
            style={{ paddingLeft: 24 }}
          />
        </div>
        <Select
          value={sort}
          onChange={(v) => setSort(v as SortKey)}
          options={SORT_OPTIONS}
          ariaLabel="Sort by"
          style={{ width: 110 }}
        />
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
          {shown.length === projects.length
            ? `${projects.length} ${projects.length === 1 ? 'project' : 'projects'}`
            : `${shown.length} of ${projects.length}`}
        </span>
        <Segmented<ViewMode>
          value={viewMode}
          onChange={chooseView}
          ariaLabel="View"
          options={[
            { value: 'grid', label: <Icon name="grid" size={13} />, title: 'Grid' },
            { value: 'list', label: <Icon name="list" size={13} />, title: 'List' },
          ]}
        />
      </div>

      <div
        onDragEnter={onDragEnter}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        style={{
          flex: 1, minHeight: 0, overflowY: 'auto', padding: 16, position: 'relative',
          outline: dragging ? '2px dashed var(--accent)' : '2px dashed transparent', outlineOffset: -8,
          background: dragging ? 'var(--accent-dim)' : 'var(--bg)',
          transition: 'background var(--ease), outline-color var(--ease)',
        }}
      >
        {projects.length === 0 ? (
          <EmptyState dragging={dragging} />
        ) : shown.length === 0 ? (
          <div style={{ fontSize: 12, color: 'var(--text-dim)', textAlign: 'center', paddingTop: 48 }}>
            No project matches “{query}”.
          </div>
        ) : viewMode === 'grid' ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 16 }}>
            {shown.map((p) => <ProjectCard key={p.id} project={p} />)}
          </div>
        ) : (
          <ProjectTable projects={shown} />
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="video/*,.mkv"
        aria-label="Choose a video file"
        style={{ display: 'none' }}
        onChange={handleInputChange}
      />
      {pendingFile && <CreateProjectModal file={pendingFile} onClose={() => setPendingFile(null)} />}
    </div>
  )
}

function EmptyState({ dragging }: { dragging: boolean }) {
  return (
    <div style={{
      height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: 12, color: dragging ? 'var(--accent-text)' : 'var(--text-dim)', textAlign: 'center',
    }}>
      {dragging ? 'Drop to create a project' : 'No projects yet — drop a video or click New project.'}
    </div>
  )
}

/** Ten stage cells, 3px tall, coloured by status. */
function StageStrip({ project, height = 3 }: { project: ProjectSummary; height?: number }) {
  return (
    <div style={{ display: 'flex', gap: 2 }} aria-hidden>
      {STAGE_ORDER.map((key) => {
        const status: StageStatus = project.stages?.[key]?.status ?? 'pending'
        const pulse = status === 'running' || status === 'queued'
        return (
          <div
            key={key}
            title={`${STAGE_LABELS[key]}: ${status}`}
            style={{
              flex: 1, height, borderRadius: 1, background: STAGE_COLOR[status],
              animation: pulse ? 'od-pulse 1200ms ease-in-out infinite' : undefined,
            }}
          />
        )
      })}
    </div>
  )
}

function Poster({ project, width, height, radius = 4, children }: {
  project: ProjectSummary; width?: number | string; height?: number | string; radius?: number; children?: ReactNode
}) {
  const ingest = project.stages?.ingest
  const hasVideo = ingest?.status === 'done'
  return (
    <div style={{
      position: 'relative', width, height, aspectRatio: height ? undefined : '16 / 9', flexShrink: 0,
      background: '#000', borderRadius: radius, border: '1px solid var(--border)', overflow: 'hidden',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      {hasVideo ? (
        <video
          src={`${api.mediaUrl(project.id, 'playback.mp4')}?v=${encodeURIComponent(ingest?.updated_at ?? '')}#t=1`}
          preload="metadata"
          muted
          playsInline
          tabIndex={-1}
          style={{ width: '100%', height: '100%', objectFit: 'cover', pointerEvents: 'none', display: 'block' }}
        />
      ) : (
        <Icon name="film" size={14} style={{ color: 'var(--text-faint)' }} />
      )}
      {children}
    </div>
  )
}

function ProjectCard({ project }: { project: ProjectSummary }) {
  const openProject = useStore((s) => s.openProject)
  const [hover, setHover] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const status = projectStatus(project)

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={project.name}
      onClick={() => void openProject(project.id)}
      onKeyDown={(e) => { if (e.key === 'Enter' && e.target === e.currentTarget) void openProject(project.id) }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setHover(false) }}
      style={{
        position: 'relative', padding: 6, background: 'var(--bg-panel)', cursor: 'pointer',
        border: `1px solid ${hover ? 'var(--accent)' : 'var(--border)'}`, borderRadius: 'var(--r-md)',
        display: 'flex', flexDirection: 'column', gap: 6, transition: 'border-color var(--ease)',
      }}
    >
      <Poster project={project}>
        <span className="timecode" style={{
          position: 'absolute', right: 4, bottom: 4, padding: '0 4px', borderRadius: 3, lineHeight: '16px',
          background: 'rgb(0 0 0 / 0.7)', color: '#fff', fontSize: 10,
        }}>
          {formatTime(project.duration)}
        </span>
        {/* Clicks on the card open the project; keep the delete button's click to itself. */}
        <span onClick={(e) => e.stopPropagation()} style={{
          position: 'absolute', top: 4, right: 4,
          opacity: hover ? 1 : 0, pointerEvents: hover ? 'auto' : 'none', transition: 'opacity var(--ease)',
        }}>
          <IconButton
            name="trash"
            title="Delete project"
            size={22}
            iconSize={13}
            onClick={() => setConfirmDelete(true)}
            style={{ background: 'rgb(0 0 0 / 0.7)', color: '#fff' }}
          />
        </span>
      </Poster>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: '0 2px 2px' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 12, fontWeight: 600, minWidth: 0, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {project.name}
          </span>
          <span className="timecode" style={{ fontSize: 10, whiteSpace: 'nowrap' }}>{langPair(project)}</span>
        </div>
        <StageStrip project={project} />
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, fontSize: 11 }}>
          <span title={status.text} style={{ color: status.color, minWidth: 0, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {status.text}
          </span>
          <span style={{ color: 'var(--text-faint)', whiteSpace: 'nowrap' }}>
            {project.segment_count} {project.segment_count === 1 ? 'line' : 'lines'} · {formatDate(modifiedAt(project))}
          </span>
        </div>
      </div>
      {confirmDelete && <DeleteConfirmModal project={project} onClose={() => setConfirmDelete(false)} />}
    </div>
  )
}

const TH_STYLE = {
  height: 24, padding: '0 8px', textAlign: 'left' as const, fontSize: 11, fontWeight: 500,
  letterSpacing: '0.04em', textTransform: 'uppercase' as const, color: 'var(--text-dim)',
  background: 'var(--bg-raised)', borderBottom: '1px solid var(--border)', whiteSpace: 'nowrap' as const,
  position: 'sticky' as const, top: 0, zIndex: 1,
}
const TD_STYLE = {
  height: 28, padding: '0 8px', fontSize: 12, borderBottom: '1px solid var(--border-subtle)',
  whiteSpace: 'nowrap' as const, overflow: 'hidden', textOverflow: 'ellipsis',
}

function ProjectTable({ projects }: { projects: ProjectSummary[] }) {
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 'var(--r-md)', background: 'var(--bg-panel)', overflow: 'hidden' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
        <colgroup>
          <col style={{ width: 64 }} />
          <col />
          <col style={{ width: 90 }} />
          <col style={{ width: 80 }} />
          <col style={{ width: 64 }} />
          <col style={{ width: '30%' }} />
          <col style={{ width: 96 }} />
          <col style={{ width: 40 }} />
        </colgroup>
        <thead>
          <tr>
            <th style={TH_STYLE} aria-label="Poster" />
            <th style={TH_STYLE}>Name</th>
            <th style={TH_STYLE}>Languages</th>
            <th style={{ ...TH_STYLE, textAlign: 'right' }}>Duration</th>
            <th style={{ ...TH_STYLE, textAlign: 'right' }}>Lines</th>
            <th style={TH_STYLE}>Status</th>
            <th style={TH_STYLE}>Modified</th>
            <th style={TH_STYLE} aria-label="Actions" />
          </tr>
        </thead>
        <tbody>
          {projects.map((p) => <ProjectRow key={p.id} project={p} />)}
        </tbody>
      </table>
    </div>
  )
}

function ProjectRow({ project }: { project: ProjectSummary }) {
  const openProject = useStore((s) => s.openProject)
  const [hover, setHover] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const status = projectStatus(project)

  function onKeyDown(e: KeyboardEvent<HTMLTableRowElement>) {
    if (e.key === 'Enter' && e.target === e.currentTarget) void openProject(project.id)
  }

  return (
    <tr
      tabIndex={0}
      aria-label={project.name}
      onClick={() => void openProject(project.id)}
      onKeyDown={onKeyDown}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{ cursor: 'pointer', background: hover ? 'var(--bg-raised)' : 'transparent' }}
    >
      <td style={{ ...TD_STYLE, padding: '0 0 0 8px' }}>
        <Poster project={project} width={48} height={27} radius={2} />
      </td>
      <td style={{ ...TD_STYLE, fontWeight: 600 }}>{project.name}</td>
      <td style={TD_STYLE}><span className="timecode" style={{ fontSize: 11 }}>{langPair(project)}</span></td>
      <td style={{ ...TD_STYLE, textAlign: 'right' }}><span className="timecode">{formatTime(project.duration)}</span></td>
      <td style={{ ...TD_STYLE, textAlign: 'right' }}><span className="timecode">{project.segment_count}</span></td>
      <td style={{ ...TD_STYLE, fontSize: 11, color: status.color }} title={status.text}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 60, flexShrink: 0 }}><StageStrip project={project} /></div>
          <span style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>{status.text}</span>
        </div>
      </td>
      <td style={{ ...TD_STYLE, fontSize: 11, color: 'var(--text-dim)' }}>{formatDate(modifiedAt(project))}</td>
      <td style={{ ...TD_STYLE, padding: 0, textAlign: 'center' }} onClick={(e) => e.stopPropagation()}>
        <IconButton
          name="trash"
          title="Delete project"
          size={22}
          iconSize={13}
          onClick={() => setConfirmDelete(true)}
          style={{ color: hover ? 'var(--text-dim)' : 'var(--text-faint)' }}
        />
        {confirmDelete && <DeleteConfirmModal project={project} onClose={() => setConfirmDelete(false)} />}
      </td>
    </tr>
  )
}

function DeleteConfirmModal({ project, onClose }: { project: ProjectSummary; onClose: () => void }) {
  const loadProjects = useStore((s) => s.loadProjects)
  const toast = useStore((s) => s.toast)
  const [deleting, setDeleting] = useState(false)

  async function handleDelete() {
    setDeleting(true)
    try {
      await api.deleteProject(project.id)
      toast('success', `Deleted "${project.name}"`)
      onClose()
      await loadProjects()
    } catch (e) {
      setDeleting(false)
      toast('error', e instanceof Error ? e.message : 'Failed to delete project')
    }
  }

  return (
    <Modal title="Delete project" onClose={deleting ? () => {} : onClose} width={380}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', lineHeight: 1.5, marginBottom: 16 }}>
        Delete “{project.name}” — this removes all media. Cannot be undone.
      </div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
        <Button variant="ghost" onClick={onClose} disabled={deleting}>
          Cancel
        </Button>
        <Button variant="danger" onClick={() => void handleDelete()} disabled={deleting}>
          {deleting ? 'Deleting…' : 'Delete'}
        </Button>
      </div>
    </Modal>
  )
}

function FormRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 10, minHeight: 26, marginBottom: 8 }}>
      <span style={{ width: 104, flexShrink: 0, fontSize: 11, color: 'var(--text-dim)', textAlign: 'right' }}>{label}</span>
      <span style={{ flex: 1, minWidth: 0 }}>{children}</span>
    </label>
  )
}

function CreateProjectModal({ file, onClose }: { file: File; onClose: () => void }) {
  const openProject = useStore((s) => s.openProject)
  const toast = useStore((s) => s.toast)
  const [name, setName] = useState(stripExt(file.name))
  const [sourceLang, setSourceLang] = useState(() => readStorage(SOURCE_KEY, 'ja'))
  const [targetLang, setTargetLang] = useState(() => readStorage(TARGET_KEY, 'en'))
  const [creating, setCreating] = useState(false)
  const [progress, setProgress] = useState(0)

  async function handleCreate() {
    const trimmed = name.trim() || stripExt(file.name)
    setCreating(true)
    setProgress(0)
    writeStorage(SOURCE_KEY, sourceLang)
    writeStorage(TARGET_KEY, targetLang)
    try {
      const project = await api.createProject(file, trimmed, sourceLang, targetLang, setProgress)
      onClose()
      await openProject(project.id)
    } catch (e) {
      setCreating(false)
      toast('error', e instanceof Error ? e.message : 'Failed to create project')
    }
  }

  return (
    <Modal title="New project" onClose={creating ? () => {} : onClose} width={420}>
      <FormRow label="Name">
        <TextInput value={name} onChange={setName} disabled={creating} ariaLabel="Name" autoFocus />
      </FormRow>
      <FormRow label="Source language">
        <Select value={sourceLang} onChange={setSourceLang} disabled={creating} options={SOURCE_LANG_OPTIONS} ariaLabel="Source language" />
      </FormRow>
      <FormRow label="Target language">
        <Select value={targetLang} onChange={setTargetLang} disabled={creating} options={TARGET_LANG_OPTIONS} ariaLabel="Target language" />
      </FormRow>
      <FormRow label="File">
        <span style={{ fontSize: 11, color: 'var(--text-dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'block' }}>
          {file.name} · {formatSize(file.size)}
        </span>
      </FormRow>
      {creating && (
        <div style={{ margin: '4px 0 12px' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>
            Uploading… {Math.round(progress * 100)}%
          </div>
          <ProgressBar value={progress} />
        </div>
      )}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 8 }}>
        <Button variant="ghost" onClick={onClose} disabled={creating}>
          Cancel
        </Button>
        <Button variant="primary" onClick={() => void handleCreate()} disabled={creating || !name.trim()}>
          {creating ? 'Creating…' : 'Create project'}
        </Button>
      </div>
    </Modal>
  )
}
