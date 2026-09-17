// Library view: project gallery + drag-drop upload. First impression of the product.
import type { ChangeEvent, DragEvent } from 'react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { Button, Field, Modal, ProgressBar, Select, TextInput } from '../components/primitives'
import { Icon } from '../components/Icon'
import { NavLink, TopBar } from '../components/TopBar'
import { useStore } from '../state/store'
import { formatTime, STAGE_LABELS, STAGE_ORDER } from '../types'
import type { ProjectSummary, StageKey, StageStatus } from '../types'

const SOURCE_LANG_OPTIONS = [
  { value: 'auto', label: 'Detect automatically' },
  { value: 'ja', label: 'Japanese' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ko', label: 'Korean' },
  { value: 'en', label: 'English' },
]

const TARGET_LANG_OPTIONS = [{ value: 'en', label: 'English' }]

const STAGE_VISUALS: Record<StageStatus, { background: string; pulse?: boolean }> = {
  pending: { background: 'var(--border-strong)' },
  queued: { background: 'var(--running)', pulse: true },
  running: { background: 'var(--running)', pulse: true },
  done: { background: 'var(--ok)' },
  dirty: { background: 'var(--warn)' },
  error: { background: 'var(--err)' },
  skipped: { background: 'var(--border-strong)' },
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

export function Library() {
  const projects = useStore((s) => s.projects)
  const loadProjects = useStore((s) => s.loadProjects)
  const setView = useStore((s) => s.setView)
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [sourceLang, setSourceLang] = useState('ja')
  const [query, setQuery] = useState('')

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

  const q = query.trim().toLowerCase()
  const shown = q ? projects.filter((p) => p.name.toLowerCase().includes(q)) : projects
  const running = projects.filter(isProjectActive).length

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <TopBar padX={48}>
        <NavLink label="Library" active onClick={() => setView('library')} />
        <NavLink label="Settings" onClick={() => setView('settings')} />
        <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>voice-preserving dubbing studio</span>
        <div style={{ flex: 1 }} />
        {running > 0 && (
          <span className="timecode" style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%', background: 'var(--accent)',
              animation: 'od-pulse 1200ms ease-in-out infinite',
            }} />
            {running} {running === 1 ? 'job' : 'jobs'} running
          </span>
        )}
      </TopBar>
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '40px 48px 48px' }}>
        <div style={{ maxWidth: 1344, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 28 }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 24, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <h1 className="serif" style={{ margin: 0, fontSize: 60, lineHeight: 1 }}>
                Every line, <span style={{ fontStyle: 'italic', color: 'var(--accent)' }}>their</span> voice.
              </h1>
              <p style={{ margin: 0, fontSize: 14, color: 'var(--text-dim)' }}>
                {projects.length} {projects.length === 1 ? 'project' : 'projects'} on this machine · media stays in{' '}
                <span style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>data/</span>
              </p>
            </div>
            <div style={{ flex: 1 }} />
            <input
              type="search"
              aria-label="Search projects"
              placeholder="Search projects"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{
                width: 260, height: 44, padding: '0 14px', borderRadius: 10, fontSize: 13,
                border: '1px solid var(--border-strong)', background: 'var(--bg-raised)',
              }}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 24 }}>
            <DropZone onFile={setPendingFile} sourceLang={sourceLang} setSourceLang={setSourceLang} />
            {shown.map((p) => (
              <ProjectCard key={p.id} project={p} />
            ))}
          </div>

          {projects.length === 0 ? <EmptyState /> : shown.length === 0 ? (
            <div style={{ fontSize: 13, color: 'var(--text-dim)' }}>No project matches “{query}”.</div>
          ) : <StageLegend />}
        </div>
      </div>
      {pendingFile && (
        <CreateProjectModal file={pendingFile} initialSourceLang={sourceLang} onClose={() => setPendingFile(null)} />
      )}
    </div>
  )
}

const CARD_H = 296

function DropZone({ onFile, sourceLang, setSourceLang }: {
  onFile: (f: File) => void; sourceLang: string; setSourceLang: (v: string) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onFile(file)
  }

  function handleInputChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) onFile(file)
    e.target.value = ''
  }

  return (
    <div
      onDragEnter={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      style={{
        minHeight: CARD_H, padding: 24, borderRadius: 'var(--r-lg)',
        border: `1.5px dashed ${dragOver ? 'var(--accent)' : 'var(--border-dashed)'}`,
        background: dragOver ? 'var(--accent-dim)' : 'transparent',
        display: 'flex', flexDirection: 'column', gap: 14,
        transition: 'border-color var(--ease), background var(--ease)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'var(--accent)' }}>
        <Icon name="upload" size={28} />
        <div className="serif" style={{ fontSize: 28, lineHeight: 1.05, color: 'var(--text)' }}>
          Drop a video here to start a dub
        </div>
      </div>
      <div style={{ fontSize: 13, color: 'var(--text-dim)' }}>
        mp4, mkv, mov or webm. It is ingested right away; nothing else runs until you press Run pipeline.
      </div>
      <div style={{ flex: 1 }} />
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10 }}>
        <label style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 5 }}>
          <span className="eyebrow">From</span>
          <Select value={sourceLang} onChange={setSourceLang} options={SOURCE_LANG_OPTIONS} style={{ minHeight: 44 }} />
        </label>
        <Icon name="arrow" style={{ color: 'var(--text-dim)', marginBottom: 14 }} />
        <label style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 5 }}>
          <span className="eyebrow">To</span>
          <Select value="en" onChange={() => {}} options={TARGET_LANG_OPTIONS} style={{ minHeight: 44 }} />
        </label>
      </div>
      <Button variant="primary" onClick={() => inputRef.current?.click()} style={{ minHeight: 44 }}>
        Choose a file
      </Button>
      <input
        ref={inputRef}
        type="file"
        accept="video/*,.mkv"
        style={{ display: 'none' }}
        onChange={handleInputChange}
      />
    </div>
  )
}

function EmptyState() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 8 }}>
      <div className="serif" style={{ fontSize: 26 }}>No projects yet</div>
      <div style={{ fontSize: 13, color: 'var(--text-dim)' }}>
        Your dubs will line up here. Every stage has a built-in fallback, so the first one runs without any model installed.
      </div>
    </div>
  )
}

function StageLegend() {
  const items: [string, string][] = [
    ['var(--ok)', 'done'], ['var(--accent)', 'running'], ['var(--warn)', 'needs re-run'],
    ['var(--err)', 'error'], ['var(--border-strong)', 'pending or skipped'],
  ]
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 18, fontSize: 12, color: 'var(--text-dim)', flexWrap: 'wrap' }}>
      <span>Stage strip:</span>
      {items.map(([color, label]) => (
        <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 14, height: 4, borderRadius: 2, background: color }} />{label}
        </span>
      ))}
    </div>
  )
}

function StageStrip({ project }: { project: ProjectSummary }) {
  return (
    <div style={{ display: 'flex', gap: 3 }}>
      {STAGE_ORDER.map((key) => {
        const status: StageStatus = project.stages?.[key]?.status ?? 'pending'
        const visuals = STAGE_VISUALS[status]
        return (
          <div
            key={key}
            title={`${STAGE_LABELS[key]}: ${status}`}
            style={{
              flex: 1, height: 4, borderRadius: 2, background: visuals.background,
              animation: visuals.pulse ? 'od-pulse 1200ms ease-in-out infinite' : undefined,
            }}
          />
        )
      })}
    </div>
  )
}

/** One human line for where a project stands, worst news first. */
function projectStatus(p: ProjectSummary): { text: string; color: string } {
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

const POSTER_BG = 'repeating-linear-gradient(135deg, #0a0617 0 14px, #130c28 14px 28px)'

function ProjectCard({ project }: { project: ProjectSummary }) {
  const openProject = useStore((s) => s.openProject)
  const [hover, setHover] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const status = projectStatus(project)
  const ingest = project.stages?.ingest
  const hasVideo = ingest?.status === 'done'

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => void openProject(project.id)}
      onKeyDown={(e) => { if (e.key === 'Enter' && e.target === e.currentTarget) void openProject(project.id) }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        position: 'relative', minHeight: CARD_H, background: 'var(--bg-raised)', overflow: 'hidden',
        border: `1px solid ${hover ? 'var(--accent)' : 'var(--border-strong)'}`,
        borderRadius: 'var(--r-lg)', cursor: 'pointer', display: 'flex', flexDirection: 'column',
        transform: hover ? 'translateY(-2px)' : 'none',
        transition: 'border-color var(--ease), transform var(--ease)',
      }}
    >
      <div style={{
        position: 'relative', height: 168, flexShrink: 0, background: POSTER_BG,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: 'var(--mono)', fontSize: 11, color: '#7a6fa3',
      }}>
        {hasVideo ? (
          <video
            src={`${api.mediaUrl(project.id, 'playback.mp4')}?v=${encodeURIComponent(ingest?.updated_at ?? '')}#t=1`}
            preload="metadata"
            muted
            playsInline
            tabIndex={-1}
            style={{ width: '100%', height: '100%', objectFit: 'cover', pointerEvents: 'none' }}
          />
        ) : 'no preview yet'}
        <span className="timecode" style={{
          position: 'absolute', right: 10, bottom: 10, padding: '2px 7px', borderRadius: 5,
          background: 'rgb(12 7 26 / 0.8)', color: '#f3eeff', fontSize: 11,
        }}>
          {formatTime(project.duration)}
        </span>
      </div>
      <div style={{ flex: 1, padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{
            fontSize: 15, fontWeight: 500, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {project.name}
          </span>
          <div style={{ flex: 1 }} />
          <span className="timecode" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>
            → {project.target_lang.toUpperCase()} · {project.segment_count} {project.segment_count === 1 ? 'line' : 'lines'}
          </span>
        </div>
        <StageStrip project={project} />
        <div style={{
          fontSize: 12, color: status.color, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }} title={status.text}>
          {status.text}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
          {new Date(project.created_at).toLocaleDateString()}
        </div>
      </div>
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); setConfirmDelete(true) }}
        title="Delete project"
        aria-label="Delete project"
        style={{
          position: 'absolute', top: 10, right: 10, width: 32, height: 32,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          borderRadius: 'var(--r-md)', border: 'none', background: 'rgb(12 7 26 / 0.8)',
          color: '#f3eeff', cursor: 'pointer', padding: 0,
          opacity: hover ? 1 : 0, pointerEvents: hover ? 'auto' : 'none',
          transition: 'opacity var(--ease)',
        }}
      >
        <Icon name="trash" />
      </button>
      {confirmDelete && <DeleteConfirmModal project={project} onClose={() => setConfirmDelete(false)} />}
    </div>
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
      <div style={{ fontSize: 13, color: 'var(--text-dim)', lineHeight: 1.5, marginBottom: 20 }}>
        Delete project — this removes all media. Cannot be undone.
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

function CreateProjectModal({ file, initialSourceLang, onClose }: {
  file: File; initialSourceLang: string; onClose: () => void
}) {
  const openProject = useStore((s) => s.openProject)
  const toast = useStore((s) => s.toast)
  const [name, setName] = useState(stripExt(file.name))
  const [sourceLang, setSourceLang] = useState(initialSourceLang)
  const [creating, setCreating] = useState(false)
  const [progress, setProgress] = useState(0)

  async function handleCreate() {
    const trimmed = name.trim() || stripExt(file.name)
    setCreating(true)
    setProgress(0)
    try {
      const project = await api.createProject(file, trimmed, sourceLang, 'en', setProgress)
      onClose()
      await openProject(project.id)
    } catch (e) {
      setCreating(false)
      toast('error', e instanceof Error ? e.message : 'Failed to create project')
    }
  }

  return (
    <Modal title="New project" onClose={creating ? () => {} : onClose}>
      <Field label="Name">
        <TextInput value={name} onChange={setName} disabled={creating} />
      </Field>
      <Field label="Source language">
        <Select value={sourceLang} onChange={setSourceLang} disabled={creating} options={SOURCE_LANG_OPTIONS} />
      </Field>
      <Field label="Target language" help="More target languages coming">
        <Select
          value="en"
          onChange={() => {}}
          disabled
          options={TARGET_LANG_OPTIONS}
          style={{ cursor: 'not-allowed', opacity: 0.6 }}
        />
      </Field>
      <div style={{ fontSize: 12, color: 'var(--text-faint)', margin: '4px 0 18px' }}>
        {file.name} · {formatSize(file.size)}
      </div>
      {creating && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
            Uploading… {Math.round(progress * 100)}%
          </div>
          <ProgressBar value={progress} />
        </div>
      )}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
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
