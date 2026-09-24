// Cast panel: one 28px row per speaker — colour dot, name (double-click to rename), line count,
// total speech time, and a play button for the reference clip. Clicking a row selects the
// speaker's first line.
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import { IconButton } from '../components/primitives'
import { speakerColor, useProject, useStore } from '../state/store'
import type { Speaker } from '../types'
import { playTake, useCurrentlyPlayingTake } from './Inspector'

const secs = (t: number): string => (t >= 60 ? `${Math.floor(t / 60)}m ${Math.round(t % 60)}s` : `${Math.round(t)}s`)

export function CastPanel() {
  const project = useProject()
  const selectSegment = useStore((s) => s.selectSegment)
  const selectedSpeaker = useStore((s) => s.project?.segments.find((sg) => sg.id === s.selection)?.speaker_id ?? null)

  const sorted = useMemo(
    () => [...(project?.segments ?? [])].sort((a, b) => a.start - b.start),
    [project?.segments],
  )
  if (!project) return null

  if (project.speakers.length === 0) {
    return (
      <div style={{ padding: '24px 16px', fontSize: 12, lineHeight: 1.5, color: 'var(--text-dim)', textAlign: 'center' }}>
        Speakers appear after Transcribe tells the voices apart.
      </div>
    )
  }

  return (
    <ul role="list" aria-label="Cast" style={{ listStyle: 'none', margin: 0, padding: 0, overflowY: 'auto', flex: 1, minHeight: 0 }}>
      {project.speakers.map((sp) => {
        const mine = sorted.filter((sg) => sg.speaker_id === sp.id)
        const total = mine.reduce((acc, sg) => acc + Math.max(0, sg.end - sg.start), 0)
        return (
          <CastRow
            key={sp.id}
            speaker={sp}
            color={speakerColor(project, sp.id)}
            lines={mine.length}
            seconds={total}
            selected={selectedSpeaker === sp.id}
            referenceUrl={sp.reference_path ? api.mediaUrl(project.id, sp.reference_path) : null}
            onSelect={() => { if (mine[0]) selectSegment(mine[0].id, true) }}
          />
        )
      })}
    </ul>
  )
}

function CastRow({ speaker, color, lines, seconds, selected, referenceUrl, onSelect }: {
  speaker: Speaker; color: string; lines: number; seconds: number; selected: boolean
  referenceUrl: string | null; onSelect: () => void
}) {
  const renameSpeaker = useStore((s) => s.renameSpeaker)
  const toast = useStore((s) => s.toast)
  const playingId = useCurrentlyPlayingTake()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(speaker.name)
  const [hover, setHover] = useState(false)
  useEffect(() => { if (!editing) setDraft(speaker.name) }, [speaker.name, editing])
  const refId = `ref:${speaker.id}`
  const playingRef = playingId === refId

  function commit() {
    setEditing(false)
    const name = draft.trim()
    if (!name || name === speaker.name) { setDraft(speaker.name); return }
    void renameSpeaker(speaker.id, name).catch((e: unknown) => {
      toast('error', e instanceof Error ? e.message : 'Failed to rename speaker')
      setDraft(speaker.name)
    })
  }

  return (
    <li
      aria-label={speaker.name}
      aria-selected={selected}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={() => { if (!editing) onSelect() }}
      onDoubleClick={() => setEditing(true)}
      style={{
        height: 28, display: 'flex', alignItems: 'center', gap: 8, padding: '0 6px 0 12px', cursor: 'default',
        borderBottom: '1px solid var(--border-subtle)', fontSize: 12,
        background: selected ? 'var(--accent-dim)' : hover ? 'var(--bg-raised)' : 'transparent',
        boxShadow: selected ? 'inset 2px 0 0 var(--accent)' : 'none',
      }}
    >
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
      {editing ? (
        <input
          type="text"
          autoFocus
          aria-label="Speaker name"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { e.preventDefault(); e.currentTarget.blur() }
            else if (e.key === 'Escape') { e.preventDefault(); setDraft(speaker.name); setEditing(false) }
          }}
          style={{
            flex: 1, minWidth: 0, height: 22, padding: '0 6px', fontSize: 12, fontWeight: 500,
            border: '1px solid var(--accent)', borderRadius: 'var(--r-md)', background: 'var(--bg-field)',
          }}
        />
      ) : (
        <span
          title="Double-click to rename"
          style={{ flex: 1, minWidth: 0, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
        >
          {speaker.name}
        </span>
      )}
      <span className="timecode" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>
        {lines} {lines === 1 ? 'line' : 'lines'} · {secs(seconds)}
      </span>
      {referenceUrl ? (
        <IconButton
          name={playingRef ? 'stop' : 'play'}
          title={playingRef ? 'Stop the reference clip' : 'Play the reference clip'}
          active={playingRef}
          onClick={() => playTake(referenceUrl, refId)}
          style={{ color }}
        />
      ) : (
        <span style={{ width: 24, flexShrink: 0 }} />
      )}
    </li>
  )
}
