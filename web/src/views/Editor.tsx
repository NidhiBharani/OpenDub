// Editor view — NLE shell (docs/plans/editor-redesign.md §3):
//   title bar (project, languages, run controls, export)
//   status strip (pipeline stages; nothing runs on a bare click)
//   Sidebar (Script | Versions | Cast) | Viewer | Inspector
//   resize handle
//   Timeline (toolbar + track headers + canvas)
// While a full run is up front, RunView replaces the workspace.
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { IconButton, MOD, Select, TextInput } from '../components/primitives'
import { TopBar } from '../components/TopBar'
import { Inspector } from '../editor/Inspector'
import { RunControls, StatusStrip } from '../editor/PipelineBar'
import { Player } from '../editor/Player'
import { RunView } from '../editor/RunView'
import { Sidebar } from '../editor/Sidebar'
import { Timeline } from '../editor/Timeline'
import { useEditorKeyboard } from '../editor/useKeyboard'
import { stageStatus, useProject, useStore } from '../state/store'
import { formatTime, LANGUAGES } from '../types'
import type { Project } from '../types'

const SIDEBAR_KEY = 'opendub.layout.sidebar'
const INSPECTOR_KEY = 'opendub.layout.inspector'
const TIMELINE_KEY = 'opendub.layout.timeline'

function readPx(key: string, fallback: number): number {
  try {
    const v = Number(localStorage.getItem(key))
    return Number.isFinite(v) && v > 0 ? v : fallback
  } catch { return fallback }
}

/** Horizontal or vertical drag-to-resize handle. `onDelta` receives pixels moved. */
function ResizeHandle({ axis, onDelta, onEnd, label }: {
  axis: 'x' | 'y'; onDelta: (px: number) => void; onEnd?: () => void; label: string
}) {
  const last = useRef(0)
  return (
    <div
      role="separator"
      aria-label={label}
      aria-orientation={axis === 'x' ? 'vertical' : 'horizontal'}
      onPointerDown={(e) => {
        e.preventDefault()
        last.current = axis === 'x' ? e.clientX : e.clientY
        e.currentTarget.setPointerCapture(e.pointerId)
      }}
      onPointerMove={(e) => {
        if (!e.currentTarget.hasPointerCapture(e.pointerId)) return
        const now = axis === 'x' ? e.clientX : e.clientY
        onDelta(now - last.current)
        last.current = now
      }}
      onPointerUp={(e) => {
        e.currentTarget.releasePointerCapture(e.pointerId)
        onEnd?.()
      }}
      style={{
        flexShrink: 0, background: 'var(--border)', userSelect: 'none',
        cursor: axis === 'x' ? 'col-resize' : 'row-resize',
        width: axis === 'x' ? 1 : '100%', height: axis === 'y' ? 1 : '100%',
        // widen the hit area without widening the visible line
        outline: '3px solid transparent', outlineOffset: -1, position: 'relative', zIndex: 2,
      }}
    />
  )
}

export function Editor() {
  useEditorKeyboard()

  const project = useProject()
  const closeProject = useStore((s) => s.closeProject)
  const setView = useStore((s) => s.setView)
  const runViewOpen = useStore((s) => s.runViewOpen)

  const [sidebarW, setSidebarW] = useState(() => readPx(SIDEBAR_KEY, 360))
  const [inspectorW, setInspectorW] = useState(() => readPx(INSPECTOR_KEY, 300))
  const [timelineH, setTimelineH] = useState(() => readPx(TIMELINE_KEY, 280))

  const persist = useCallback((key: string, v: number) => {
    try { localStorage.setItem(key, String(Math.round(v))) } catch { /* storage blocked */ }
  }, [])

  if (!project) return null

  const renderDone = stageStatus(project, 'render') === 'done'
  const media = project.media

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <TopBar onBrand={closeProject} padX={12}>
        <div style={{ width: 1, height: 18, background: 'var(--border)', flexShrink: 0 }} />
        <ProjectName project={project} />
        <LanguagePair project={project} />
        {media && (
          <span className="timecode" style={{ whiteSpace: 'nowrap' }}>
            {formatTime(media.duration)} · {media.width}×{media.height} · {media.fps.toFixed(3).replace(/\.?0+$/, '')} fps
          </span>
        )}
        <div style={{ flex: 1 }} />
        <RunControls />
        <a
          href={renderDone ? api.mediaUrl(project.id, 'render/dubbed.mp4') : undefined}
          download
          aria-disabled={!renderDone}
          title={renderDone ? 'Export dubbed.mp4' : 'Export (run Render first)'}
          aria-label="Export dubbed.mp4"
          style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 26, height: 26,
            borderRadius: 'var(--r-md)', textDecoration: 'none', flexShrink: 0,
            border: '1px solid var(--border-strong)', color: 'var(--text)', background: 'var(--bg-overlay)',
            opacity: renderDone ? 1 : 0.4, pointerEvents: renderDone ? 'auto' : 'none',
          }}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M8 2.5V11M4.5 7.5L8 11l3.5-3.5M2.5 13.5h11" />
          </svg>
        </a>
        <IconButton name="settings" title="Pipeline settings" size={26} onClick={() => setView('settings')} />
      </TopBar>
      <StatusStrip />

      {runViewOpen ? <RunView /> : (
        <>
          <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
            <section aria-label="Sidebar" style={{
              width: sidebarW, flexShrink: 0, minWidth: 0, minHeight: 0, background: 'var(--bg-panel)',
              display: 'flex', flexDirection: 'column',
            }}>
              <Sidebar />
            </section>
            <ResizeHandle
              axis="x" label="Resize sidebar"
              onDelta={(dx) => setSidebarW((w) => Math.max(260, Math.min(640, w + dx)))}
              onEnd={() => persist(SIDEBAR_KEY, sidebarW)}
            />
            <section aria-label="Viewer" style={{ flex: 1, minWidth: 0, minHeight: 0, background: 'var(--bg-sunk)' }}>
              <Player />
            </section>
            <ResizeHandle
              axis="x" label="Resize inspector"
              onDelta={(dx) => setInspectorW((w) => Math.max(240, Math.min(520, w - dx)))}
              onEnd={() => persist(INSPECTOR_KEY, inspectorW)}
            />
            <section aria-label="Inspector" style={{
              width: inspectorW, flexShrink: 0, minWidth: 0, minHeight: 0, background: 'var(--bg-panel)',
            }}>
              <Inspector />
            </section>
          </div>
          <ResizeHandle
            axis="y" label="Resize timeline"
            onDelta={(dy) => setTimelineH((h) => Math.max(160, Math.min(window.innerHeight * 0.7, h - dy)))}
            onEnd={() => persist(TIMELINE_KEY, timelineH)}
          />
          <section aria-label="Timeline" style={{ height: timelineH, flexShrink: 0, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <Timeline />
          </section>
        </>
      )}
    </div>
  )
}

function LanguagePair({ project }: { project: Project }) {
  const refreshProject = useStore((s) => s.refreshProject)
  const toast = useStore((s) => s.toast)
  const options = LANGUAGES.map((l) => ({ value: l.code, label: l.code.toUpperCase() }))
  const known = (code: string) => options.some((o) => o.value === code)
  const src = known(project.source_lang) ? options : [{ value: project.source_lang, label: project.source_lang.toUpperCase() }, ...options]
  const tgt = known(project.target_lang) ? options : [{ value: project.target_lang, label: project.target_lang.toUpperCase() }, ...options]

  async function change(field: 'source_lang' | 'target_lang', value: string) {
    if (value === project[field]) return
    try {
      await api.updateProject(project.id, { [field]: value })
      await refreshProject()
      if (field === 'target_lang') toast('success', `Target language is now ${value.toUpperCase()} — run Translate to start a new dub`)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to change language')
    }
  }

  const sel = { width: 'auto', height: 22, fontSize: 11, padding: '0 4px', fontFamily: 'var(--mono)' } as const
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }} title={`Source → target language (${MOD}? for help)`}>
      <Select ariaLabel="Source language" value={project.source_lang} onChange={(v) => void change('source_lang', v)} options={src} style={sel} />
      <span style={{ color: 'var(--text-faint)', fontSize: 11 }}>→</span>
      <Select ariaLabel="Target language" value={project.target_lang} onChange={(v) => void change('target_lang', v)} options={tgt} style={sel} />
    </div>
  )
}

function ProjectName({ project }: { project: Project }) {
  const refreshProject = useStore((s) => s.refreshProject)
  const toast = useStore((s) => s.toast)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(project.name)

  useEffect(() => {
    if (!editing) setDraft(project.name)
  }, [project.name, editing])

  async function commit() {
    setEditing(false)
    const trimmed = draft.trim()
    if (!trimmed || trimmed === project.name) {
      setDraft(project.name)
      return
    }
    try {
      await api.updateProject(project.id, { name: trimmed })
      await refreshProject()
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to rename project')
      setDraft(project.name)
    }
  }

  if (editing) {
    return (
      <TextInput
        autoFocus
        ariaLabel="Project name"
        value={draft}
        onChange={setDraft}
        onBlur={() => void commit()}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
          else if (e.key === 'Escape') { setDraft(project.name); setEditing(false) }
        }}
        style={{ width: 240, height: 22, fontWeight: 600, fontSize: 13 }}
      />
    )
  }

  return (
    <button
      type="button"
      onDoubleClick={() => setEditing(true)}
      title="Double-click to rename"
      style={{
        fontSize: 13, fontWeight: 600, cursor: 'text', padding: '2px 4px', border: 'none', background: 'transparent',
        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 320, color: 'var(--text)',
      }}
    >
      {project.name}
    </button>
  )
}
