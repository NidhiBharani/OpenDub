// Editor view: top bar (project, pipeline stepper, run) over Script | Player | Inspector, with the
// Timeline along the bottom. While a full run is up front, RunView replaces the workspace.
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { Icon } from '../components/Icon'
import { NavLink, TopBar } from '../components/TopBar'
import { Inspector } from '../editor/Inspector'
import { PipelineStepper, RunControls } from '../editor/PipelineBar'
import { Player } from '../editor/Player'
import { RunView } from '../editor/RunView'
import { TranscriptList } from '../editor/TranscriptList'
import { Timeline } from '../editor/Timeline'
import { useEditorKeyboard } from '../editor/useKeyboard'
import { stageStatus, useProject, useStore } from '../state/store'
import { formatTime } from '../types'
import type { Project } from '../types'

export function Editor() {
  useEditorKeyboard()

  const project = useProject()
  const closeProject = useStore((s) => s.closeProject)
  const setView = useStore((s) => s.setView)
  const runViewOpen = useStore((s) => s.runViewOpen)

  if (!project) return null

  const renderDone = stageStatus(project, 'render') === 'done'
  const media = project.media

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <TopBar onBrand={closeProject}>
        <div style={{ width: 1, height: 24, background: 'var(--border-strong)', flexShrink: 0 }} />
        <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0, flexShrink: 1 }}>
          <ProjectName project={project} />
          <div className="timecode" style={{ fontSize: 11, padding: '0 6px', whiteSpace: 'nowrap' }}>
            {project.source_lang.toUpperCase()} → {project.target_lang.toUpperCase()}
            {media && ` · ${formatTime(media.duration)} · ${media.width}×${media.height}`}
          </div>
        </div>

        <PipelineStepper />

        <RunControls />
        {renderDone && (
          <a
            href={api.mediaUrl(project.id, 'render/dubbed.mp4')}
            download
            style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 36, height: 36,
              borderRadius: 'var(--r-md)', textDecoration: 'none', flexShrink: 0,
              border: '1px solid var(--border-strong)', color: 'var(--text)',
            }}
            title="Export dubbed.mp4"
            aria-label="Export dubbed.mp4"
          >
            <Icon name="download" />
          </a>
        )}
        <NavLink label="Settings" onClick={() => setView('settings')} />
      </TopBar>

      {runViewOpen ? <RunView /> : (
        <>
          <div style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: '320px minmax(0, 1fr) 360px' }}>
            <section aria-label="Script" style={{
              minHeight: 0, borderRight: '1px solid var(--border-strong)', background: 'var(--bg-raised)',
            }}>
              <TranscriptList />
            </section>
            <section aria-label="Player" style={{ minWidth: 0, minHeight: 0 }}>
              <Player />
            </section>
            <section aria-label="Line inspector" style={{
              minHeight: 0, borderLeft: '1px solid var(--border-strong)', background: 'var(--bg-raised)',
            }}>
              <Inspector />
            </section>
          </div>
          <div style={{ height: Math.min(290, 140 + 38 * project.speakers.length), flexShrink: 0, borderTop: '1px solid var(--border-strong)', minHeight: 0 }}>
            <Timeline />
          </div>
        </>
      )}
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
      <input
        autoFocus
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => void commit()}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
          else if (e.key === 'Escape') { setDraft(project.name); setEditing(false) }
        }}
        style={{
          background: 'var(--bg)', border: '1px solid var(--accent)', borderRadius: 'var(--r-sm)',
          color: 'var(--text)', fontWeight: 500, fontSize: 13, padding: '1px 5px', width: 260,
        }}
      />
    )
  }

  return (
    <div
      onDoubleClick={() => setEditing(true)}
      title="Double-click to rename"
      style={{
        fontSize: 13, fontWeight: 500, cursor: 'text', padding: '2px 6px',
        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 320,
      }}
    >
      {project.name}
    </div>
  )
}
