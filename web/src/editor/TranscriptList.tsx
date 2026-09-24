// Script panel (docs/plans/editor-redesign.md §4): sentence-by-sentence correction.
//   header: search · filter (All / Stale / Warnings / Excluded) · counts · view options
//   rows (44px, grow while editing): speaker chip · start timecode · status glyphs · hover toolbar
//                                    source line (word spans with confidence) · translation line
//   inline editing: double-click a line or press Enter on the selected row; Enter commits and moves
//   to the next row, Tab walks source → translation → next row, Esc reverts.
// The file keeps its historical name; `TranscriptList` is an alias of `ScriptPanel`.
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent } from 'react'
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import {
  Badge, Button, IconButton, MenuItem, Modal, MOD, Popover, Segmented, SpeakerChip, StatusDot, TextInput,
} from '../components/primitives'
import { speakerColor, stageStatus, useProject, useStore } from '../state/store'
import { formatTime } from '../types'
import type { Job, Segment, Speaker, Word } from '../types'

type Filter = 'all' | 'stale' | 'warnings' | 'excluded'
type Field = 'source' | 'translation'

const LOW_CONFIDENCE = 0.6

const isActiveJob = (j: Job): boolean => j.status === 'queued' || j.status === 'running'

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable
}

const isStale = (s: Segment): boolean => s.translate_dirty || s.synth_dirty

/** Seconds the active take runs past its slot, and the overrun as a fraction of the slot. */
function overrun(s: Segment): { over: number; ratio: number } {
  const take = s.takes.find((t) => t.id === s.active_take_id)
  const slot = s.end - s.start
  if (!take || slot <= 0) return { over: 0, ratio: 0 }
  const over = take.duration - slot
  return { over, ratio: over / slot }
}

const errorText = (err: unknown, fallback: string): string => (err instanceof Error && err.message ? err.message : fallback)

// ---- panel -------------------------------------------------------------------------------------

export function ScriptPanel() {
  const project = useProject()
  const selection = useStore((s) => s.selection)
  const playhead = useStore((s) => s.playhead)
  const playing = useStore((s) => s.playing)
  const editing = useStore((s) => s.editing)
  const jobs = useStore((s) => s.jobs)
  const selectSegment = useStore((s) => s.selectSegment)
  const setEditing = useStore((s) => s.setEditing)

  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [showConfidence, setShowConfidence] = useState(true)
  const rowRefs = useRef(new Map<string, HTMLDivElement>())

  const segments = project?.segments ?? []
  const sorted = useMemo(() => [...segments].sort((a, b) => a.start - b.start), [segments])
  const synthDone = stageStatus(project, 'synthesize') === 'done'

  const hasWarning = useCallback((s: Segment): boolean => {
    if (s.skipped) return false
    if (synthDone && s.takes.length === 0) return true
    return overrun(s).ratio > 0.1
  }, [synthDone])

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    return sorted.filter((s) => {
      if (filter === 'stale' && !isStale(s)) return false
      if (filter === 'warnings' && !hasWarning(s)) return false
      if (filter === 'excluded' && !s.skipped) return false
      if (q && !s.source_text.toLowerCase().includes(q) && !s.translated_text.toLowerCase().includes(q)) return false
      return true
    })
  }, [sorted, filter, query, hasWarning])

  const currentId = useMemo(
    () => sorted.find((s) => s.start <= playhead && playhead < s.end)?.id ?? null,
    [sorted, playhead],
  )
  const busyIds = useMemo(() => {
    const ids = new Set<string>()
    for (const j of Object.values(jobs)) if (isActiveJob(j) && j.segment_id) ids.add(j.segment_id)
    return ids
  }, [jobs])

  // Follow the playhead while playing, and keep the selection in view when it changes elsewhere
  // (timeline click, ↑/↓).
  useEffect(() => {
    if (!playing || !currentId) return
    rowRefs.current.get(currentId)?.scrollIntoView({ block: 'nearest' })
  }, [currentId, playing])
  useEffect(() => {
    if (selection) rowRefs.current.get(selection)?.scrollIntoView({ block: 'nearest' })
  }, [selection])

  // Enter on the selected row starts editing its translation (the global map leaves Enter to us).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Enter' || e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return
      if (e.defaultPrevented || isEditableTarget(e.target)) return
      const s = useStore.getState()
      if (s.editing || !s.selection || !s.project?.segments.some((x) => x.id === s.selection)) return
      e.preventDefault()
      s.setEditing({ segmentId: s.selection, field: 'translation' })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const registerRef = useCallback((id: string, el: HTMLDivElement | null) => {
    if (el) rowRefs.current.set(id, el)
    else rowRefs.current.delete(id)
  }, [])

  /** Move the editor to `field` of the row `delta` rows away (in the visible order); exits at the ends. */
  const moveEditing = useCallback((fromId: string, delta: number, field: Field) => {
    const idx = visible.findIndex((s) => s.id === fromId)
    const next = idx < 0 ? undefined : visible[idx + delta]
    if (!next) {
      setEditing(null)
      return
    }
    selectSegment(next.id, false)
    setEditing({ segmentId: next.id, field })
  }, [visible, selectSegment, setEditing])

  if (!project) return null

  const staleCount = sorted.filter(isStale).length
  const excludedCount = sorted.filter((s) => s.skipped).length
  const counts = [
    visible.length === sorted.length ? `${sorted.length} lines` : `${visible.length} / ${sorted.length} lines`,
    staleCount > 0 ? `${staleCount} stale` : null,
    excludedCount > 0 ? `${excludedCount} excluded` : null,
  ].filter(Boolean).join(' · ')

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div className="panel-header" style={{ paddingLeft: 8 }}>
        <TextInput
          ariaLabel="Search lines"
          placeholder="Search source or translation"
          value={query}
          onChange={setQuery}
          style={{ height: 20, fontSize: 11, padding: '0 6px' }}
        />
        <IconButton
          name="eye"
          title={showConfidence ? 'Hide low-confidence words' : 'Show low-confidence words'}
          active={showConfidence}
          onClick={() => setShowConfidence((v) => !v)}
          size={22}
          iconSize={13}
        />
      </div>
      <div className="panel-header" style={{ paddingLeft: 8, background: 'var(--bg-panel)' }}>
        <Segmented<Filter>
          size="sm"
          ariaLabel="Filter lines"
          value={filter}
          onChange={setFilter}
          options={[
            { value: 'all', label: 'All' },
            { value: 'stale', label: 'Stale', title: 'Lines whose translation or voice is out of date' },
            { value: 'warnings', label: 'Warnings', title: 'Lines whose take overruns its slot or is missing' },
            { value: 'excluded', label: 'Excluded', title: 'Lines that keep the original audio' },
          ]}
        />
        <span style={{ flex: 1 }} />
        <span className="timecode" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {counts}
        </span>
      </div>

      <div role="listbox" aria-label="Script lines" style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
        {sorted.length === 0 && (
          <div style={{ padding: 12, fontSize: 12, color: 'var(--text-dim)' }}>
            Lines appear here once Transcribe has run.
          </div>
        )}
        {sorted.length > 0 && visible.length === 0 && (
          <div style={{ padding: 12, fontSize: 12, color: 'var(--text-dim)' }}>No lines match.</div>
        )}
        {visible.map((seg, i) => (
          <Row
            key={seg.id}
            pid={project.id}
            segment={seg}
            speaker={project.speakers.find((sp) => sp.id === seg.speaker_id)}
            color={speakerColor(project, seg.speaker_id)}
            selected={selection === seg.id}
            current={currentId === seg.id}
            editingField={editing?.segmentId === seg.id ? editing.field : null}
            busy={busyIds.has(seg.id)}
            synthDone={synthDone}
            showConfidence={showConfidence}
            isLast={i === visible.length - 1}
            moveEditing={moveEditing}
            registerRef={registerRef}
          />
        ))}
      </div>
    </div>
  )
}

/** @deprecated alias kept for the Sidebar stub; import `ScriptPanel`. */
export const TranscriptList = ScriptPanel

// ---- row ---------------------------------------------------------------------------------------

interface RowProps {
  pid: string
  segment: Segment
  speaker: Speaker | undefined
  color: string
  selected: boolean
  current: boolean
  editingField: Field | null
  busy: boolean
  synthDone: boolean
  showConfidence: boolean
  isLast: boolean
  moveEditing: (fromId: string, delta: number, field: Field) => void
  registerRef: (id: string, el: HTMLDivElement | null) => void
}

const Row = memo(function Row({
  pid, segment, speaker, color, selected, current, editingField, busy, synthDone, showConfidence, isLast,
  moveEditing, registerRef,
}: RowProps) {
  const [hover, setHover] = useState(false)
  const [focusWithin, setFocusWithin] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const closeMenu = useCallback(() => setMenuOpen(false), [])
  const setRowRef = useCallback((el: HTMLDivElement | null) => registerRef(segment.id, el), [registerRef, segment.id])

  const store = useStore.getState // actions are stable; read them at call time
  const skipped = !!segment.skipped
  const stale = isStale(segment)
  const staleWhy = segment.translate_dirty
    ? 'Stale: the source changed — re-translate to update this line'
    : 'Stale: the translation changed — re-voice to update the dub'
  const missingTake = synthDone && !skipped && segment.takes.length === 0
  const { over, ratio } = overrun(segment)
  const overrunLevel = skipped ? null : ratio > 0.3 ? 'err' : ratio > 0.1 ? 'warn' : null

  const select = () => store().selectSegment(segment.id, true)
  const startEdit = (field: Field) => {
    store().selectSegment(segment.id, false)
    store().setEditing({ segmentId: segment.id, field })
  }

  const regenerate = async (stages: ('translate' | 'synthesize')[]) => {
    try {
      store().applyJob(await api.regenerateSegment(pid, segment.id, stages))
    } catch (err) {
      store().toast('error', errorText(err, 'Failed to start regeneration.'))
    }
  }
  const retranscribe = async () => {
    setMenuOpen(false)
    try {
      store().applyJob(await api.transcribeSegment(pid, segment.id))
    } catch (err) {
      store().toast('error', errorText(err, 'Failed to start transcription.'))
    }
  }
  const run = (label: string, fn: () => Promise<void>) => {
    setMenuOpen(false)
    void fn().catch((err: unknown) => store().toast('error', errorText(err, `${label} failed.`)))
  }

  const onSplit = () => {
    const t = store().playhead
    if (t > segment.start && t < segment.end) run('Split', () => store().splitSegment(segment.id, t))
  }
  const splitEnabled = current // `current` is start <= playhead < end; the click re-checks strictly
  const regenTitle = (base: string) =>
    skipped ? `${base} — unavailable: this line keeps the original audio` : busy ? `${base} — working…` : base
  const toolbarVisible = hover || focusWithin || menuOpen

  const onRowKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget) return
    if (e.key === 'Enter') {
      e.preventDefault()
      startEdit('translation')
    }
  }

  const rowStyle: CSSProperties = {
    position: 'relative',
    minHeight: 44,
    padding: '5px 6px 6px 8px',
    borderBottom: '1px solid var(--border-subtle)',
    borderLeft: `2px solid ${selected ? 'var(--accent)' : 'transparent'}`,
    background: current ? 'var(--accent-dim)' : selected ? 'var(--bg-overlay)' : hover ? 'var(--bg-raised)' : 'transparent',
    cursor: 'default',
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    opacity: skipped ? 0.7 : 1,
  }

  return (
    <div
      ref={setRowRef}
      role="option"
      aria-selected={selected}
      aria-label={`Line ${formatTime(segment.start)} ${speaker?.name ?? ''}`}
      tabIndex={0}
      onClick={select}
      onKeyDown={onRowKeyDown}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocusCapture={() => setFocusWithin(true)}
      onBlurCapture={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setFocusWithin(false)
      }}
      style={rowStyle}
    >
      {/* line 1: speaker · timecode · status · toolbar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, height: 18, minWidth: 0 }}>
        <SpeakerChip name={speaker?.name ?? 'Unknown'} color={color} size="sm" />
        <span
          className="timecode"
          style={{ color: current ? 'var(--accent-text)' : undefined, flexShrink: 0 }}
        >
          {formatTime(segment.start, true).slice(0, -2)}
        </span>
        {stale && <span title={staleWhy} style={{ display: 'inline-flex' }}><StatusDot color="var(--warn)" /></span>}
        {missingTake && (
          <span title="No take: synthesis produced nothing for this line" style={{ display: 'inline-flex' }}>
            <StatusDot color="var(--err)" />
          </span>
        )}
        {skipped && (
          <Badge
            color="var(--text-dim)"
            bg="repeating-linear-gradient(135deg, var(--border-strong) 0 1px, var(--bg-overlay) 1px 4px)"
            title="Keeps the original audio (inside a skip range)"
          >
            KEPT ORIGINAL
          </Badge>
        )}
        {overrunLevel && (
          <Badge
            color={overrunLevel === 'err' ? 'var(--err-text)' : 'var(--warn-text)'}
            bg={overrunLevel === 'err' ? 'var(--err-dim)' : 'var(--warn-dim)'}
            title={`The active take runs ${over.toFixed(2)}s past its ${(segment.end - segment.start).toFixed(2)}s slot`}
          >
            +{over.toFixed(1)}s
          </Badge>
        )}
        <span style={{ flex: 1 }} />
        <div
          role="toolbar"
          aria-label="Line actions"
          onClick={(e) => e.stopPropagation()}
          onDoubleClick={(e) => e.stopPropagation()}
          style={{
            display: 'flex', alignItems: 'center', gap: 0, position: 'relative',
            opacity: toolbarVisible ? 1 : 0, pointerEvents: toolbarVisible ? 'auto' : 'none',
            transition: 'opacity var(--ease)',
          }}
        >
          <IconButton
            name="play" title="Play from here" iconSize={12}
            onClick={() => { store().seek(segment.start); store().setPlaying(true) }}
          />
          <IconButton
            name="translate" title={regenTitle('Re-translate and re-voice')} iconSize={13}
            disabled={skipped || busy} onClick={() => void regenerate(['translate', 'synthesize'])}
          />
          <IconButton
            name="mic" title={regenTitle('Re-voice')} iconSize={13}
            disabled={skipped || busy} onClick={() => void regenerate(['synthesize'])}
          />
          <IconButton
            name="split" iconSize={13}
            title={splitEnabled ? `Split at playhead (${MOD}B)` : `Split at playhead (${MOD}B) — move the playhead inside this line`}
            disabled={!splitEnabled} onClick={onSplit}
          />
          <IconButton name="more" title="More actions" iconSize={13} active={menuOpen} onClick={() => setMenuOpen((v) => !v)} />
          <Popover open={menuOpen} onClose={closeMenu} align="right" width={210}>
            <MenuItem
              icon="merge" label="Merge with next" disabled={isLast}
              onClick={() => run('Merge', () => store().mergeSegmentWithNext(segment.id))}
            />
            <MenuItem icon="refresh" label="Re-transcribe this line" disabled={busy} onClick={() => void retranscribe()} />
            <MenuItem
              icon="ban" label="Keep original here" disabled={skipped}
              onClick={() => run('Keep original', () => store().addSkipRange({ start: segment.start, end: segment.end }, ''))}
            />
            <div style={{ height: 1, background: 'var(--border-subtle)', margin: '4px 0' }} />
            <MenuItem icon="trash" label="Delete line" danger onClick={() => { setMenuOpen(false); setConfirmDelete(true) }} />
          </Popover>
        </div>
      </div>

      {/* line 0: source */}
      {editingField === 'source' ? (
        <LineEditor
          field="source"
          segment={segment}
          onMove={(delta, field) => moveEditing(segment.id, delta, field)}
          style={{ fontStyle: 'italic', fontSize: 11, color: 'var(--text-dim)' }}
        />
      ) : (
        <div
          onDoubleClick={(e) => { e.stopPropagation(); startEdit('source') }}
          title="Double-click to edit the source"
          style={{ fontSize: 11, fontStyle: 'italic', color: 'var(--text-dim)', lineHeight: '15px', minHeight: 15, wordBreak: 'break-word' }}
        >
          {segment.words.length > 0
            ? <Words words={segment.words} showConfidence={showConfidence} segmentId={segment.id} />
            : (segment.source_text || <span style={{ color: 'var(--text-faint)' }}>— no source —</span>)}
        </div>
      )}

      {/* line 2: translation */}
      {editingField === 'translation' ? (
        <LineEditor
          field="translation"
          segment={segment}
          onMove={(delta, field) => moveEditing(segment.id, delta, field)}
          style={{ fontSize: 12, color: 'var(--text)' }}
        />
      ) : (
        <div
          onDoubleClick={(e) => { e.stopPropagation(); startEdit('translation') }}
          title="Double-click to edit the translation"
          style={{
            fontSize: 12, lineHeight: '16px', minHeight: 16, wordBreak: 'break-word',
            color: segment.translated_text ? 'var(--text)' : 'var(--text-faint)',
            fontStyle: segment.translated_text ? 'normal' : 'italic',
          }}
        >
          {segment.translated_text || '— untranslated —'}
        </div>
      )}

      {confirmDelete && (
        <Modal title="Delete line" onClose={() => setConfirmDelete(false)} width={380}>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 14 }}>
            Remove this line, its translation and its takes. The original audio stays in the mix. Cannot be undone.
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <Button onClick={() => setConfirmDelete(false)}>Cancel</Button>
            <Button
              variant="danger"
              autoFocus
              onClick={() => { setConfirmDelete(false); run('Delete', () => store().deleteSegment(segment.id)) }}
            >
              Delete
            </Button>
          </div>
        </Modal>
      )}
    </div>
  )
})

// ---- word spans ---------------------------------------------------------------------------------

function Words({ words, showConfidence, segmentId }: { words: Word[]; showConfidence: boolean; segmentId: string }) {
  return (
    <>
      {words.map((w, i) => {
        const low = showConfidence && w.confidence !== null && w.confidence < LOW_CONFIDENCE
        return (
          <span
            key={i}
            role="button"
            tabIndex={-1}
            title={low ? `confidence ${w.confidence!.toFixed(2)} · click to seek` : `${formatTime(w.start, true)} · click to seek`}
            onClick={(e) => {
              e.stopPropagation()
              const s = useStore.getState()
              s.selectSegment(segmentId, false)
              s.seek(w.start)
            }}
            style={{
              cursor: 'pointer', borderRadius: 2,
              borderBottom: low ? '1px dotted var(--warn)' : '1px dotted transparent',
              color: low ? 'var(--text)' : undefined,
            }}
          >
            {w.text}
          </span>
        )
      })}
    </>
  )
}

// ---- inline editor ------------------------------------------------------------------------------

function LineEditor({ field, segment, onMove, style }: {
  field: Field
  segment: Segment
  onMove: (delta: number, field: Field) => void
  style?: CSSProperties
}) {
  const original = field === 'source' ? segment.source_text : segment.translated_text
  const [draft, setDraft] = useState(original)
  const ref = useRef<HTMLTextAreaElement>(null)
  const done = useRef(false) // set once the edit has been committed or reverted

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [draft])
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.focus()
    el.setSelectionRange(el.value.length, el.value.length)
  }, [])

  const commit = () => {
    if (done.current) return
    done.current = true
    if (draft === original) return
    const patch = field === 'source' ? { source_text: draft } : { translated_text: draft }
    void useStore.getState().updateSegment(segment.id, patch).catch((err: unknown) => {
      useStore.getState().toast('error', errorText(err, 'Failed to save the line.'))
    })
  }
  const exit = () => {
    const s = useStore.getState()
    if (s.editing?.segmentId === segment.id && s.editing.field === field) s.setEditing(null)
  }

  const onKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    e.stopPropagation()
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      commit()
      onMove(1, field)
    } else if (e.key === 'Tab') {
      e.preventDefault()
      commit()
      if (e.shiftKey) {
        if (field === 'translation') onMove(0, 'source')
        else onMove(-1, 'translation')
      } else if (field === 'source') onMove(0, 'translation')
      else onMove(1, 'source')
    } else if (e.key === 'Escape') {
      e.preventDefault()
      done.current = true
      exit()
    }
  }

  return (
    <textarea
      ref={ref}
      aria-label={field === 'source' ? 'Edit source' : 'Edit translation'}
      value={draft}
      rows={1}
      onChange={(e) => setDraft(e.target.value)}
      onKeyDown={onKeyDown}
      onBlur={() => { commit(); exit() }}
      onClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
      style={{
        display: 'block', width: '100%', resize: 'none', overflow: 'hidden',
        background: 'var(--bg-field)', border: '1px solid var(--accent)', borderRadius: 'var(--r-sm)',
        padding: '1px 4px', margin: '0 -5px', lineHeight: field === 'source' ? '15px' : '16px',
        fontFamily: 'inherit', boxSizing: 'content-box', ...style,
      }}
    />
  )
}
