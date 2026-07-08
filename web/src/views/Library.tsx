// Library view: project gallery + drag-drop upload. First impression of the product.
import type { ChangeEvent, DragEvent } from 'react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { Button, Field, Modal, ProgressBar, Select, TextInput } from '../components/primitives'
import type { View } from '../state/store'
import { useStore } from '../state/store'
import { formatTime, STAGE_LABELS, STAGE_ORDER } from '../types'
import type { ProjectSummary, StageStatus } from '../types'

const SOURCE_LANG_OPTIONS = [
  { value: 'auto', label: 'Detect automatically' },
  { value: 'ja', label: 'Japanese' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ko', label: 'Korean' },
  { value: 'en', label: 'English' },
]

const TARGET_LANG_OPTIONS = [{ value: 'en', label: 'English' }]

const STAGE_VISUALS: Record<StageStatus, { background: string; border?: string; pulse?: boolean }> = {
  pending: { background: 'var(--border)' },
  queued: { background: 'var(--running)', pulse: true },
  running: { background: 'var(--running)', pulse: true },
  done: { background: 'var(--ok)' },
  dirty: { background: 'var(--warn)' },
  error: { background: 'var(--err)' },
  skipped: { background: 'transparent', border: '1px solid var(--border)' },
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

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <TopBar setView={setView} />
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: 32 }}>
        <div style={{ maxWidth: 1080, margin: '0 auto' }}>
          <DropZone onFile={setPendingFile} />
          {projects.length === 0 ? (
            <EmptyState />
          ) : (
            <div
              style={{
                display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
                gap: 16, marginTop: 28,
              }}
            >
              {projects.map((p) => (
                <ProjectCard key={p.id} project={p} />
              ))}
            </div>
          )}
        </div>
      </div>
      {pendingFile && <CreateProjectModal file={pendingFile} onClose={() => setPendingFile(null)} />}
    </div>
  )
}

function TopBar({ setView }: { setView: (v: View) => void }) {
  return (
    <div
      style={{
        height: 56, flexShrink: 0, padding: '0 24px', borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center',
      }}
    >
      <span style={{ fontSize: 15, fontWeight: 650, letterSpacing: '-0.01em' }}>
        Open<span style={{ color: 'var(--accent)' }}>Dub</span>
      </span>
      <span style={{ fontSize: 12, color: 'var(--text-faint)', marginLeft: 10 }}>
        voice-preserving dubbing studio
      </span>
      <div style={{ flex: 1 }} />
      <Button variant="ghost" onClick={() => setView('settings')}>
        Settings
      </Button>
    </div>
  )
}

function DropZone({ onFile }: { onFile: (f: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [hover, setHover] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const active = hover || dragOver

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
      onClick={() => inputRef.current?.click()}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onDragEnter={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click() }}
      style={{
        width: '100%', height: 120, borderRadius: 'var(--r-lg)',
        border: `1.5px dashed ${active ? 'var(--accent)' : 'var(--border-strong)'}`,
        background: active ? 'var(--accent-dim)' : 'transparent',
        display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center',
        padding: '0 24px', cursor: 'pointer', transition: 'border-color var(--ease), background var(--ease)',
      }}
    >
      <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>
        Drop a video here or click to browse — mp4, mkv, mov, webm
      </span>
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
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', marginTop: 80 }}>
      <div style={{ fontSize: 40, opacity: 0.5, marginBottom: 10 }}>🎬</div>
      <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>No projects yet</div>
      <div style={{ fontSize: 13, color: 'var(--text-dim)' }}>Drop a video above to start dubbing</div>
    </div>
  )
}

function StageStrip({ project }: { project: ProjectSummary }) {
  return (
    <div style={{ display: 'flex', gap: 3, marginTop: 12 }}>
      {STAGE_ORDER.map((key) => {
        const status: StageStatus = project.stages?.[key]?.status ?? 'pending'
        const visuals = STAGE_VISUALS[status]
        return (
          <div
            key={key}
            title={`${STAGE_LABELS[key]}: ${status}`}
            style={{
              flex: 1, height: 6, borderRadius: 3, background: visuals.background,
              border: visuals.border, animation: visuals.pulse ? 'od-pulse 1200ms ease-in-out infinite' : undefined,
            }}
          />
        )
      })}
    </div>
  )
}

function ProjectCard({ project }: { project: ProjectSummary }) {
  const openProject = useStore((s) => s.openProject)
  const [hover, setHover] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)

  return (
    <div
      onClick={() => void openProject(project.id)}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        position: 'relative', background: 'var(--bg-raised)',
        border: `1px solid ${hover ? 'var(--border-strong)' : 'var(--border)'}`,
        borderRadius: 'var(--r-lg)', padding: 16, cursor: 'pointer',
        transform: hover ? 'translateY(-1px)' : 'none',
        transition: 'border-color var(--ease), transform var(--ease)',
      }}
    >
      <div
        style={{
          fontSize: 14, fontWeight: 600, paddingRight: 20,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}
      >
        {project.name}
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4 }}>
        <span className="timecode">{formatTime(project.duration)}</span>
        {` · ${project.segment_count} segments · → EN`}
      </div>
      <StageStrip project={project} />
      <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 10 }}>
        {new Date(project.created_at).toLocaleDateString()}
      </div>
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); setConfirmDelete(true) }}
        title="Delete project"
        style={{
          position: 'absolute', top: 8, right: 8, width: 22, height: 22, lineHeight: '20px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          borderRadius: 'var(--r-sm)', border: '1px solid transparent', background: 'transparent',
          color: 'var(--text-dim)', cursor: 'pointer', fontSize: 12, padding: 0,
          opacity: hover ? 1 : 0, pointerEvents: hover ? 'auto' : 'none',
          transition: 'opacity var(--ease), background var(--ease)',
        }}
      >
        ✕
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

function CreateProjectModal({ file, onClose }: { file: File; onClose: () => void }) {
  const openProject = useStore((s) => s.openProject)
  const toast = useStore((s) => s.toast)
  const [name, setName] = useState(stripExt(file.name))
  const [sourceLang, setSourceLang] = useState('ja')
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
