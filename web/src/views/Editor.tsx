// Editor view: header bar + Player / PipelineBar / Timeline main column + RightPanel sidebar.
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { Button } from '../components/primitives'
import { PipelineBar } from '../editor/PipelineBar'
import { Player } from '../editor/Player'
import { RightPanel } from '../editor/RightPanel'
import { Timeline } from '../editor/Timeline'
import { useEditorKeyboard } from '../editor/useKeyboard'
import { stageStatus, useProject, useStore } from '../state/store'
import type { Project } from '../types'

export function Editor() {
  useEditorKeyboard()

  const project = useProject()
  const closeProject = useStore((s) => s.closeProject)
  const setView = useStore((s) => s.setView)
  const audioTrack = useStore((s) => s.audioTrack)
  const setAudioTrack = useStore((s) => s.setAudioTrack)

  if (!project) return null

  const mixDone = stageStatus(project, 'mix') === 'done'
  const renderDone = stageStatus(project, 'render') === 'done'

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <header
        style={{
          height: 44, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10,
          padding: '0 12px', borderBottom: '1px solid var(--border)',
        }}
      >
        <Button variant="ghost" onClick={closeProject} title="Back to library">‹</Button>
        <ProjectName project={project} />
        <div style={{ flex: 1 }} />

        <div
          style={{
            display: 'flex', border: '1px solid var(--border-strong)',
            borderRadius: 'var(--r-md)', overflow: 'hidden',
          }}
        >
          <button
            type="button"
            onClick={() => setAudioTrack('original')}
            style={{
              border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 500,
              padding: '5px 12px',
              background: audioTrack === 'original' ? 'var(--bg-overlay)' : 'transparent',
              color: audioTrack === 'original' ? 'var(--text)' : 'var(--text-dim)',
            }}
          >
            Original
          </button>
          <button
            type="button"
            disabled={!mixDone}
            title={mixDone ? undefined : 'Run the Mix stage to preview the dub audio'}
            onClick={() => setAudioTrack('dub')}
            style={{
              border: 'none', borderLeft: '1px solid var(--border-strong)',
              cursor: mixDone ? 'pointer' : 'not-allowed', fontSize: 12, fontWeight: 500,
              padding: '5px 12px', opacity: mixDone ? 1 : 0.5,
              background: audioTrack === 'dub' ? 'var(--bg-overlay)' : 'transparent',
              color: audioTrack === 'dub' ? 'var(--text)' : 'var(--text-dim)',
            }}
          >
            Dub
          </button>
        </div>

        {renderDone && (
          <a
            href={api.mediaUrl(project.id, 'render/dubbed.mp4')}
            download
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 12px',
              borderRadius: 'var(--r-md)', fontSize: 13, fontWeight: 500, textDecoration: 'none',
              background: 'var(--bg-overlay)', border: '1px solid var(--border-strong)', color: 'var(--text)',
            }}
          >
            ⭳ Download
          </a>
        )}

        <Button variant="ghost" onClick={() => setView('settings')} title="Settings">⚙</Button>
      </header>

      <div style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 340px' }}>
        <div style={{ minWidth: 0, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          <div style={{ flex: 1, minHeight: 0 }}>
            <Player />
          </div>
          <PipelineBar />
          <div style={{ height: 230, flexShrink: 0, borderTop: '1px solid var(--border)', minHeight: 0 }}>
            <Timeline />
          </div>
        </div>
        <div style={{ borderLeft: '1px solid var(--border)', minHeight: 0, overflow: 'auto' }}>
          <RightPanel />
        </div>
      </div>
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
          color: 'var(--text)', fontWeight: 600, fontSize: 13, padding: '3px 6px', width: 260,
        }}
      />
    )
  }

  return (
    <div
      onDoubleClick={() => setEditing(true)}
      title="Double-click to rename"
      style={{
        fontSize: 13, fontWeight: 600, cursor: 'text', padding: '3px 6px',
        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 320,
      }}
    >
      {project.name}
    </div>
  )
}
