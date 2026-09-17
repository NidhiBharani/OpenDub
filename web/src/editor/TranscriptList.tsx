// Synced transcript list: click to select + seek, follows the playhead while
// playing, shows per-segment dirty state and a project-wide progress summary.
import { useEffect, useMemo, useRef } from 'react'
import { speakerColor, useProject, useStore } from '../state/store'
import { formatTime } from '../types'
import type { Segment, Speaker } from '../types'

export function TranscriptList() {
  const project = useProject()
  const selection = useStore((s) => s.selection)
  const playhead = useStore((s) => s.playhead)
  const playing = useStore((s) => s.playing)
  const selectSegment = useStore((s) => s.selectSegment)
  const rowRefs = useRef(new Map<string, HTMLButtonElement>())

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
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{
        height: 44, flexShrink: 0, padding: '0 16px', borderBottom: '1px solid var(--border-strong)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
      }}>
        <span className="serif" style={{ fontSize: 20 }}>Script</span>
        <span className="timecode" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>
          {segments.length} lines · {translated} translated · {voiced} voiced
        </span>
      </div>
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
        {sorted.length === 0 && (
          <div style={{ padding: 16, fontSize: 13, color: 'var(--text-dim)' }}>
            The script appears here once the Transcribe stage has run.
          </div>
        )}
        {sorted.map((seg) => (
          <Row
            key={seg.id}
            segment={seg}
            speaker={project.speakers.find((sp) => sp.id === seg.speaker_id)}
            color={speakerColor(project, seg.speaker_id)}
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
    </div>
  )
}

function Row({ segment, speaker, color, selected, current, onClick, registerRef }: {
  segment: Segment
  speaker: Speaker | undefined
  color: string
  selected: boolean
  current: boolean
  onClick: () => void
  registerRef: (el: HTMLButtonElement | null) => void
}) {
  const flag = segment.translate_dirty ? 'needs re-translate' : segment.synth_dirty ? 'needs re-voice' : ''

  return (
    <button
      type="button"
      ref={registerRef}
      onClick={onClick}
      aria-pressed={selected}
      style={{
        width: '100%', textAlign: 'left', border: 'none', borderBottom: '1px solid var(--border)',
        background: selected ? 'var(--bg-overlay)' : 'transparent', cursor: 'pointer',
        padding: '10px 16px', display: 'flex', flexDirection: 'column', gap: 4,
      }}
    >
      <span style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, width: '100%' }}>
        <span className="timecode" style={{ fontSize: 11, color: current ? 'var(--accent)' : 'var(--text-dim)' }}>
          {formatTime(segment.start, true).slice(0, -2)}
        </span>
        <span style={{ width: 7, height: 7, borderRadius: '50%', background: color, flexShrink: 0 }} />
        <span style={{ color, fontWeight: 500 }}>{speaker?.name ?? 'Unknown'}</span>
        <span style={{ flex: 1 }} />
        {flag && <span style={{ color: 'var(--warn-text)' }}>{flag}</span>}
      </span>
      <span style={{
        fontSize: 13, lineHeight: 1.4, color: segment.translated_text ? 'var(--text)' : 'var(--text-faint)',
        fontStyle: segment.translated_text ? 'normal' : 'italic',
      }}>
        {segment.translated_text || '— untranslated —'}
      </span>
      <span style={{ fontSize: 12, lineHeight: 1.4, color: 'var(--text-dim)' }}>{segment.source_text}</span>
    </button>
  )
}
