// Synced transcript list: click to select + seek, follows the playhead while
// playing, shows per-segment dirty state and a project-wide progress summary.
import { useEffect, useMemo, useRef } from 'react'
import { useProject, useStore } from '../state/store'
import { formatTime } from '../types'
import type { Segment, Speaker } from '../types'

export function TranscriptList() {
  const project = useProject()
  const selection = useStore((s) => s.selection)
  const playhead = useStore((s) => s.playhead)
  const playing = useStore((s) => s.playing)
  const selectSegment = useStore((s) => s.selectSegment)
  const rowRefs = useRef(new Map<string, HTMLDivElement>())

  const segments = project?.segments ?? []
  const sorted = useMemo(() => [...segments].sort((a, b) => a.start - b.start), [segments])

  const currentId = useMemo(() => {
    const seg = sorted.find((s) => s.start <= playhead && playhead < s.end)
    return seg?.id ?? null
  }, [sorted, playhead])

  // Follow the playhead while playing: scroll the current row into view when
  // it changes.
  useEffect(() => {
    if (!playing || !currentId) return
    rowRefs.current.get(currentId)?.scrollIntoView({ block: 'nearest' })
  }, [currentId, playing])

  if (!project) return null

  const translated = segments.filter((s) => s.translated_text.trim() !== '').length
  const voiced = segments.filter((s) => s.active_take_id != null).length

  return (
    <div style={{ height: '100%', overflowY: 'auto' }}>
      <div style={{
        position: 'sticky', top: 0, zIndex: 1, background: 'var(--bg-raised)',
        padding: '8px 12px', borderBottom: '1px solid var(--border)',
        fontSize: 13, color: 'var(--text-dim)',
      }}>
        {segments.length} segments · {translated} translated · {voiced} voiced
      </div>
      {sorted.map((seg) => (
        <Row
          key={seg.id}
          segment={seg}
          speaker={project.speakers.find((sp) => sp.id === seg.speaker_id)}
          selected={selection === seg.id}
          current={currentId === seg.id}
          onClick={() => selectSegment(seg.id, true)}
          registerRef={(el) => {
            if (el) rowRefs.current.set(seg.id, el)
            else rowRefs.current.delete(seg.id)
          }}
        />
      ))}
    </div>
  )
}

function Row({ segment, speaker, selected, current, onClick, registerRef }: {
  segment: Segment
  speaker: Speaker | undefined
  selected: boolean
  current: boolean
  onClick: () => void
  registerRef: (el: HTMLDivElement | null) => void
}) {
  const dirty = segment.translate_dirty || segment.synth_dirty
  const background = selected ? 'var(--accent-dim)' : current ? 'var(--bg-overlay)' : 'transparent'

  return (
    <div
      ref={registerRef}
      onClick={onClick}
      style={{
        display: 'flex', gap: 10, padding: '8px 12px', borderBottom: '1px solid var(--border)',
        cursor: 'pointer', background,
        boxShadow: current ? 'inset 2px 0 0 var(--accent)' : undefined,
      }}
    >
      <div style={{
        width: 3, alignSelf: 'stretch', borderRadius: 2, flexShrink: 0,
        background: speaker?.color ?? 'var(--text-faint)',
      }} />
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="timecode">{formatTime(segment.start)}</span>
          <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{speaker?.name ?? 'Unknown'}</span>
          {dirty && (
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--warn)', flexShrink: 0 }} />
          )}
        </div>
        <div style={{
          fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {segment.source_text}
        </div>
        <div style={{
          fontSize: 13, color: 'var(--text-dim)', fontStyle: segment.translated_text ? 'normal' : 'italic',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {segment.translated_text || '— untranslated —'}
        </div>
      </div>
    </div>
  )
}
