// Inspector (docs/plans/editor-redesign.md §3): compact label/value rows for the selected line.
//   header 28px: "Line 04" · timecode range · duration
//   tabs: Line (properties + takes + regenerate footer) | Speaker | Info
//   a selected skip range shows its own sheet ("Keep original"); nothing selected shows a hint.
import type { CSSProperties, KeyboardEvent, ReactNode } from 'react'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import {
  Badge, Button, IconButton, PropertyRow, Segmented, Select, StatusDot, TextInput,
} from '../components/primitives'
import { speakerColor, stageStatus, useProject, useSelectedSegment, useStore } from '../state/store'
import { formatTime } from '../types'
import type { Job, Project, Segment, Speaker, Take, TimeRange } from '../types'

type SegmentPatch = Parameters<typeof api.updateSegment>[2]
type Tab = 'line' | 'speaker' | 'info'

const LABEL_W = 68

// ---- module-level shared take-preview audio element -------------------------------------------
// One <audio> shared across the whole app (the run view uses it too) so starting a new preview
// always stops whatever was playing before, regardless of re-renders.

let sharedAudio: HTMLAudioElement | null = null
let playingTakeId: string | null = null
const playbackListeners = new Set<() => void>()

function notifyPlayback() {
  playbackListeners.forEach((l) => l())
}

function getSharedAudio(): HTMLAudioElement {
  if (!sharedAudio) {
    sharedAudio = new Audio()
    sharedAudio.addEventListener('ended', () => { playingTakeId = null; notifyPlayback() })
    sharedAudio.addEventListener('error', () => { playingTakeId = null; notifyPlayback() })
  }
  return sharedAudio
}

export function playTake(url: string, id: string) {
  const audio = getSharedAudio()
  if (playingTakeId === id) {
    audio.pause()
    playingTakeId = null
    notifyPlayback()
    return
  }
  audio.pause()
  audio.src = url
  audio.currentTime = 0
  playingTakeId = id
  notifyPlayback()
  void audio.play().catch(() => {
    playingTakeId = null
    notifyPlayback()
  })
}

export function useCurrentlyPlayingTake(): string | null {
  const [id, setId] = useState<string | null>(playingTakeId)
  useEffect(() => {
    const listener = () => setId(playingTakeId)
    playbackListeners.add(listener)
    return () => { playbackListeners.delete(listener) }
  }, [])
  return id
}

// ---- helpers ------------------------------------------------------------------------------------

const errorText = (err: unknown, fallback: string): string => (err instanceof Error && err.message ? err.message : fallback)
const shortProvider = (id: string): string => id.split('.').pop() || id
const isActive = (j: Job): boolean => j.status === 'queued' || j.status === 'running'

/** Local text draft that follows its server value: reset when the key changes, and re-seeded when
 *  the server value changes externally (e.g. after Re-translate) — but never while the user has
 *  unsaved local edits (draft !== last-seen base). */
function useSyncedDraft(key: string, serverValue: string): [string, (v: string) => void] {
  const [state, setState] = useState({ key, base: serverValue, draft: serverValue })
  if (state.key !== key) {
    setState({ key, base: serverValue, draft: serverValue })
  } else if (state.base !== serverValue) {
    setState({ key, base: serverValue, draft: state.draft === state.base ? serverValue : state.draft })
  }
  const setDraft = (draft: string) => setState((s) => ({ ...s, draft }))
  return [state.draft, setDraft]
}

const fieldStyle: CSSProperties = { height: 24, fontSize: 12, padding: '0 6px' }

/** Auto-growing textarea in the field style; commits on blur and ⌘/Ctrl+Enter. */
function GrowingTextarea({ value, onChange, onCommit, ariaLabel, minRows = 1, style }: {
  value: string; onChange: (v: string) => void; onCommit: () => void; ariaLabel: string; minRows?: number
  style?: CSSProperties
}) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [value])
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      e.currentTarget.blur()
    }
  }
  return (
    <textarea
      ref={ref}
      aria-label={ariaLabel}
      value={value}
      rows={minRows}
      onChange={(e) => onChange(e.target.value)}
      onBlur={onCommit}
      onKeyDown={onKeyDown}
      style={{
        display: 'block', width: '100%', resize: 'none', overflow: 'hidden', minHeight: 24,
        background: 'var(--bg-field)', border: '1px solid var(--border-strong)', borderRadius: 'var(--r-md)',
        padding: '4px 6px', fontSize: 12, lineHeight: '16px', fontFamily: 'inherit', ...style,
      }}
    />
  )
}

function Header({ title, right }: { title: ReactNode; right?: ReactNode }) {
  return (
    <div className="panel-header" style={{ justifyContent: 'space-between' }}>
      <span style={{ fontSize: 12, fontWeight: 600, whiteSpace: 'nowrap' }}>{title}</span>
      {right && <span className="timecode" style={{ whiteSpace: 'nowrap' }}>{right}</span>}
    </div>
  )
}

function SectionLabel({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
      margin: '10px 0 4px', paddingTop: 8, borderTop: '1px solid var(--border-subtle)',
    }}>
      <span className="eyebrow">{children}</span>
      {right}
    </div>
  )
}

const Value = ({ children, mono, dim }: { children: ReactNode; mono?: boolean; dim?: boolean }) => (
  <span
    className={mono ? 'timecode' : undefined}
    style={{ fontSize: mono ? 11 : 12, color: dim ? 'var(--text-dim)' : mono ? 'var(--text)' : undefined, wordBreak: 'break-all' }}
  >
    {children}
  </span>
)

const bodyStyle: CSSProperties = {
  flex: 1, minHeight: 0, overflowY: 'auto', padding: '8px 10px 12px 8px',
  display: 'flex', flexDirection: 'column', gap: 2,
}

// ---- main ---------------------------------------------------------------------------------------

export function Inspector() {
  const project = useProject()
  const segment = useSelectedSegment()
  const skipSelection = useStore((s) => s.skipSelection)
  const skipRange = project?.skip_ranges?.find((r) => r.id === skipSelection) ?? null

  if (!project) return null
  if (segment) return <SegmentSheet key={segment.id} project={project} segment={segment} />
  if (skipRange) return <SkipRangeSheet key={skipRange.id} project={project} range={skipRange} />
  return (
    <div style={{
      height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: 'var(--text-dim)', fontSize: 12, padding: 24, textAlign: 'center', lineHeight: 1.5,
    }}>
      Select a line in the script or on the timeline to inspect it.
    </div>
  )
}

// ---- segment sheet -----------------------------------------------------------------------------

function SegmentSheet({ project, segment }: { project: Project; segment: Segment }) {
  const [tab, setTab] = useState<Tab>('line')
  const jobs = useStore((s) => s.jobs)
  const toast = useStore((s) => s.toast)
  const updateSegment = useStore((s) => s.updateSegment)
  const applyJob = useStore((s) => s.applyJob)

  const sorted = useMemo(() => [...project.segments].sort((a, b) => a.start - b.start), [project.segments])
  const lineNo = sorted.findIndex((s) => s.id === segment.id) + 1
  const duration = Math.max(0, segment.end - segment.start)
  const busy = Object.values(jobs).some((j) => isActive(j) && j.segment_id === segment.id)
  const speaker = project.speakers.find((sp) => sp.id === segment.speaker_id)

  const commit = (patch: SegmentPatch) => {
    void updateSegment(segment.id, patch).catch((err: unknown) => {
      toast('error', errorText(err, 'Failed to save changes.'))
    })
  }
  const regenerate = async (stages: ('translate' | 'synthesize')[]) => {
    try {
      applyJob(await api.regenerateSegment(project.id, segment.id, stages))
    } catch (err) {
      toast('error', errorText(err, 'Failed to start regeneration.'))
    }
  }

  const regenTitle = (base: string) =>
    segment.skipped ? `${base} — unavailable: this line keeps the original audio` : busy ? `${base} — working…` : base

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <Header
        title={`Line ${String(lineNo).padStart(2, '0')}`}
        right={`${formatTime(segment.start, true).slice(0, -2)} – ${formatTime(segment.end, true).slice(0, -2)} · ${duration.toFixed(1)}s`}
      />
      <div className="panel-header" style={{ paddingLeft: 8, background: 'var(--bg-panel)' }}>
        <Segmented<Tab>
          size="sm"
          ariaLabel="Inspector tabs"
          value={tab}
          onChange={setTab}
          options={[{ value: 'line', label: 'Line' }, { value: 'speaker', label: 'Speaker' }, { value: 'info', label: 'Info' }]}
        />
      </div>

      {tab === 'line' && (
        <>
          <div style={bodyStyle}>
            <LineTab project={project} segment={segment} commit={commit} />
          </div>
          <div style={{
            height: 40, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 6, padding: '0 8px',
            borderTop: '1px solid var(--border)',
          }}>
            <Button
              variant="default" size="sm" style={{ flex: 1 }}
              disabled={busy || !!segment.skipped}
              title={regenTitle('Translate again, then re-voice')}
              onClick={() => void regenerate(['translate', 'synthesize'])}
            >
              Re-translate
            </Button>
            <Button
              variant="primary" size="sm" style={{ flex: 1 }}
              disabled={busy || !!segment.skipped}
              title={regenTitle('Re-voice with the current translation')}
              onClick={() => void regenerate(['synthesize'])}
            >
              {busy ? 'Working…' : 'Re-voice'}
            </Button>
          </div>
        </>
      )}
      {tab === 'speaker' && (
        <div style={bodyStyle}>
          {speaker
            ? <SpeakerTab project={project} speaker={speaker} />
            : <div style={{ fontSize: 12, color: 'var(--text-dim)', padding: 4 }}>This line has no speaker.</div>}
        </div>
      )}
      {tab === 'info' && (
        <div style={bodyStyle}>
          <InfoTab project={project} segment={segment} />
        </div>
      )}
    </div>
  )
}

// ---- Line tab ------------------------------------------------------------------------------------

function LineTab({ project, segment, commit }: { project: Project; segment: Segment; commit: (p: SegmentPatch) => void }) {
  const toast = useStore((s) => s.toast)
  const [emotion, setEmotion] = useSyncedDraft(`${segment.id}:emotion`, segment.emotion)
  const [source, setSource] = useSyncedDraft(`${segment.id}:source`, segment.source_text)
  const [translation, setTranslation] = useSyncedDraft(`${segment.id}:translation`, segment.translated_text)
  const [start, setStart] = useSyncedDraft(`${segment.id}:start`, segment.start.toFixed(2))
  const [end, setEnd] = useSyncedDraft(`${segment.id}:end`, segment.end.toFixed(2))

  const duration = Math.max(0, segment.end - segment.start)
  const chars = translation.length
  const budget = Math.round(duration * 15)
  const ratio = budget > 0 ? chars / budget : (chars > 0 ? Infinity : 0)
  const meterColor = ratio > 1.15 ? 'var(--err)' : ratio > 1 ? 'var(--warn)' : 'var(--ok)'
  const verdict = ratio > 1 ? `over by ${chars - budget} · will be sped up` : 'fits the slot'

  const commitTiming = () => {
    const s = Number(start)
    const e = Number(end)
    const dur = project.media?.duration ?? Infinity
    if (!Number.isFinite(s) || !Number.isFinite(e) || s < 0 || e <= s || e > dur) {
      toast('error', `Timing must satisfy 0 ≤ start < end ≤ ${formatTime(dur, true)}.`)
      setStart(segment.start.toFixed(2))
      setEnd(segment.end.toFixed(2))
      return
    }
    const patch: SegmentPatch = {}
    if (Math.abs(s - segment.start) > 1e-6) patch.start = s
    if (Math.abs(e - segment.end) > 1e-6) patch.end = e
    if (Object.keys(patch).length > 0) commit(patch)
  }
  const onTimingKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') e.currentTarget.blur()
  }

  const synthDone = stageStatus(project, 'synthesize') === 'done'
  const missingTake = synthDone && !segment.skipped && segment.takes.length === 0

  return (
    <>
      <PropertyRow label="Speaker" labelWidth={LABEL_W}>
        <Select
          ariaLabel="Speaker"
          value={segment.speaker_id}
          onChange={(v) => commit({ speaker_id: v })}
          options={project.speakers.map((sp) => ({ value: sp.id, label: sp.name }))}
          style={{ ...fieldStyle, color: speakerColor(project, segment.speaker_id), fontWeight: 600 }}
        />
      </PropertyRow>
      <PropertyRow label="Emotion" labelWidth={LABEL_W}>
        <TextInput
          ariaLabel="Emotion hint"
          value={emotion}
          onChange={setEmotion}
          onBlur={() => { if (emotion !== segment.emotion) commit({ emotion }) }}
          placeholder="e.g. angry, whispering"
          style={fieldStyle}
        />
      </PropertyRow>
      <PropertyRow label="Source" labelWidth={LABEL_W} align="start">
        <GrowingTextarea
          ariaLabel="Source text"
          value={source}
          onChange={setSource}
          onCommit={() => { if (source !== segment.source_text) commit({ source_text: source }) }}
          style={{ marginTop: 1 }}
        />
      </PropertyRow>
      <PropertyRow label="Translation" labelWidth={LABEL_W} align="start">
        <GrowingTextarea
          ariaLabel="Translation"
          value={translation}
          onChange={setTranslation}
          onCommit={() => { if (translation !== segment.translated_text) commit({ translated_text: translation }) }}
          minRows={2}
          style={{ marginTop: 1 }}
        />
        <div style={{ height: 3, borderRadius: 2, background: 'var(--border-subtle)', overflow: 'hidden', marginTop: 5 }}>
          <div style={{
            height: '100%', background: meterColor, transition: 'width var(--ease)',
            width: `${Math.min(100, Math.round(ratio * 100))}%`,
          }} />
        </div>
        <div className="timecode" style={{ marginTop: 3, color: ratio > 1 ? meterColor : undefined }}>
          {chars} / {budget} chars · {verdict}
        </div>
      </PropertyRow>
      <PropertyRow label="Timing" labelWidth={LABEL_W}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <TextInput
            ariaLabel="Start (seconds)" value={start} onChange={setStart} onBlur={commitTiming} onKeyDown={onTimingKey}
            style={{ ...fieldStyle, fontFamily: 'var(--mono)', fontSize: 11 }}
          />
          <span style={{ color: 'var(--text-faint)', fontSize: 11 }}>→</span>
          <TextInput
            ariaLabel="End (seconds)" value={end} onChange={setEnd} onBlur={commitTiming} onKeyDown={onTimingKey}
            style={{ ...fieldStyle, fontFamily: 'var(--mono)', fontSize: 11 }}
          />
        </div>
      </PropertyRow>
      <PropertyRow label="Status" labelWidth={LABEL_W}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', fontSize: 11, color: 'var(--text-dim)' }}>
          {segment.skipped && (
            <Badge
              color="var(--text-dim)"
              bg="repeating-linear-gradient(135deg, var(--border-strong) 0 1px, var(--bg-overlay) 1px 4px)"
              title="Keeps the original audio (inside a skip range)"
            >
              KEPT ORIGINAL
            </Badge>
          )}
          {segment.translate_dirty && <><StatusDot color="var(--warn)" /><span>needs re-translate</span></>}
          {!segment.translate_dirty && segment.synth_dirty && <><StatusDot color="var(--warn)" /><span>needs re-voice</span></>}
          {missingTake && <><StatusDot color="var(--err)" /><span>no take</span></>}
          {!segment.skipped && !segment.translate_dirty && !segment.synth_dirty && !missingTake && (
            <><StatusDot color={segment.active_take_id ? 'var(--ok)' : 'var(--text-faint)'} /><span>{segment.active_take_id ? 'voiced' : 'not voiced yet'}</span></>
          )}
          <span style={{ color: 'var(--text-faint)' }}>· {segment.takes.length} {segment.takes.length === 1 ? 'take' : 'takes'}</span>
        </div>
      </PropertyRow>

      <TakesList project={project} segment={segment} commit={commit} />
    </>
  )
}

// ---- takes ---------------------------------------------------------------------------------------

function TakesList({ project, segment, commit }: { project: Project; segment: Segment; commit: (p: SegmentPatch) => void }) {
  const playingId = useCurrentlyPlayingTake()
  const ordered = useMemo(() => {
    const asc = [...segment.takes].sort((a, b) => a.created_at.localeCompare(b.created_at))
    return asc.map((take, i) => ({ take, n: i + 1 })).reverse()
  }, [segment.takes])

  return (
    <div>
      <SectionLabel>Takes · {segment.takes.length}</SectionLabel>
      {ordered.length === 0 ? (
        <div style={{ fontSize: 12, color: 'var(--text-faint)', padding: '2px 0' }}>No takes yet — Re-voice to create one.</div>
      ) : (
        <div role="radiogroup" aria-label="Takes" style={{ display: 'flex', flexDirection: 'column' }}>
          {ordered.map(({ take, n }) => (
            <TakeRow
              key={take.id}
              take={take}
              n={n}
              pid={project.id}
              active={segment.active_take_id === take.id}
              playing={playingId === take.id}
              onSelect={() => commit({ active_take_id: take.id })}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function TakeRow({ take, n, pid, active, playing, onSelect }: {
  take: Take; n: number; pid: string; active: boolean; playing: boolean; onSelect: () => void
}) {
  const rateOff = Math.abs(take.rate_factor - 1) > 1e-3
  return (
    <label style={{
      display: 'flex', alignItems: 'center', gap: 6, height: 26, padding: '0 2px 0 4px',
      borderRadius: 'var(--r-sm)', cursor: 'pointer', fontSize: 12,
      background: active ? 'var(--bg-overlay)' : 'transparent',
    }}>
      <input
        type="radio"
        name={`active-take-${take.id.slice(0, 4)}`}
        aria-label={`Use take ${n}`}
        checked={active}
        onChange={onSelect}
        style={{ accentColor: 'var(--accent)', flexShrink: 0, margin: 0, width: 12, height: 12 }}
      />
      <span style={{ fontWeight: active ? 600 : 500, flexShrink: 0 }}>Take {n}</span>
      <span style={{
        color: 'var(--text-dim)', fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0,
      }} title={take.provider_id}>
        {shortProvider(take.provider_id)}
      </span>
      {take.lang && <Badge>{take.lang.toUpperCase()}</Badge>}
      <span style={{ flex: 1 }} />
      <span
        className="timecode"
        style={{ whiteSpace: 'nowrap', flexShrink: 0, color: 'var(--text)' }}
        title={rateOff ? `time-stretched ×${take.rate_factor.toFixed(2)} to fit the slot` : `${take.duration.toFixed(2)}s`}
      >
        {take.duration.toFixed(1)}s
        {rateOff && <span style={{ color: 'var(--warn-text)' }}> ×{take.rate_factor.toFixed(2)}</span>}
      </span>
      <IconButton
        name={playing ? 'stop' : 'play'}
        title={playing ? `Stop take ${n}` : `Play take ${n}`}
        iconSize={11}
        size={22}
        onClick={() => playTake(api.mediaUrl(pid, take.path), take.id)}
      />
    </label>
  )
}

// ---- Speaker tab ---------------------------------------------------------------------------------

function SpeakerTab({ project, speaker }: { project: Project; speaker: Speaker }) {
  const renameSpeaker = useStore((s) => s.renameSpeaker)
  const toast = useStore((s) => s.toast)
  const [name, setName] = useSyncedDraft(`${speaker.id}:name`, speaker.name)
  const playingId = useCurrentlyPlayingTake()
  const color = speakerColor(project, speaker.id)
  const idx = project.speakers.findIndex((sp) => sp.id === speaker.id)
  const lines = project.segments.filter((s) => s.speaker_id === speaker.id).length
  const refId = `ref:${speaker.id}`

  const commitName = () => {
    const trimmed = name.trim()
    if (!trimmed || trimmed === speaker.name) {
      setName(speaker.name)
      return
    }
    void renameSpeaker(speaker.id, trimmed).catch((err: unknown) => {
      toast('error', errorText(err, 'Failed to rename the speaker.'))
      setName(speaker.name)
    })
  }

  return (
    <>
      <PropertyRow label="Name" labelWidth={LABEL_W}>
        <TextInput
          ariaLabel="Speaker name"
          value={name}
          onChange={setName}
          onBlur={commitName}
          onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
          style={fieldStyle}
        />
      </PropertyRow>
      <PropertyRow label="Colour" labelWidth={LABEL_W}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span aria-label="Speaker colour" style={{ width: 12, height: 12, borderRadius: 3, background: color, flexShrink: 0 }} />
          <Value mono dim>palette {idx < 0 ? '—' : (idx % 8) + 1} · by cast order</Value>
        </div>
      </PropertyRow>
      <PropertyRow label="Reference" labelWidth={LABEL_W}>
        {speaker.reference_path ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <IconButton
              name={playingId === refId ? 'stop' : 'play'}
              title={playingId === refId ? 'Stop reference clip' : 'Play reference clip'}
              iconSize={11} size={22}
              onClick={() => playTake(api.mediaUrl(project.id, speaker.reference_path!), refId)}
            />
            <Value mono dim>{speaker.reference_path.split('/').pop()}</Value>
          </div>
        ) : <Value dim>none</Value>}
      </PropertyRow>
      <PropertyRow label="Lines" labelWidth={LABEL_W}>
        <Value>{lines} of {project.segments.length}</Value>
      </PropertyRow>
      <PropertyRow label="Id" labelWidth={LABEL_W}>
        <Value mono dim>{speaker.id}</Value>
      </PropertyRow>
    </>
  )
}

// ---- Info tab ------------------------------------------------------------------------------------

function InfoTab({ project, segment }: { project: Project; segment: Segment }) {
  const scored = segment.words.filter((w) => w.confidence !== null)
  const mean = scored.length > 0 ? scored.reduce((a, w) => a + (w.confidence ?? 0), 0) / scored.length : null
  const low = scored.filter((w) => (w.confidence ?? 1) < 0.6).length
  const dirty = [segment.translate_dirty ? 'translate' : null, segment.synth_dirty ? 'synth' : null].filter(Boolean).join(', ')
  return (
    <>
      <PropertyRow label="Segment" labelWidth={LABEL_W}><Value mono>{segment.id}</Value></PropertyRow>
      <PropertyRow label="Languages" labelWidth={LABEL_W}>
        <Value mono>{project.source_lang.toUpperCase()} → {project.target_lang.toUpperCase()}</Value>
      </PropertyRow>
      <PropertyRow label="Span" labelWidth={LABEL_W}>
        <Value mono>{formatTime(segment.start, true)} – {formatTime(segment.end, true)}</Value>
      </PropertyRow>
      <PropertyRow label="Words" labelWidth={LABEL_W}>
        <Value>{segment.words.length}{low > 0 ? <span style={{ color: 'var(--text-dim)' }}> · {low} low confidence</span> : null}</Value>
      </PropertyRow>
      <PropertyRow label="Confidence" labelWidth={LABEL_W}>
        <Value mono>{mean === null ? '—' : mean.toFixed(2)}</Value>
      </PropertyRow>
      <PropertyRow label="Takes" labelWidth={LABEL_W}><Value>{segment.takes.length}</Value></PropertyRow>
      <PropertyRow label="Active take" labelWidth={LABEL_W}><Value mono>{segment.active_take_id ?? '—'}</Value></PropertyRow>
      <PropertyRow label="Dirty" labelWidth={LABEL_W}><Value dim={!dirty}>{dirty || 'none'}</Value></PropertyRow>
      <PropertyRow label="Excluded" labelWidth={LABEL_W}><Value dim={!segment.skipped}>{segment.skipped ? 'yes — keeps original' : 'no'}</Value></PropertyRow>
      {segment.notes && <PropertyRow label="Notes" labelWidth={LABEL_W} align="start"><Value dim>{segment.notes}</Value></PropertyRow>}
    </>
  )
}

// ---- skip range sheet --------------------------------------------------------------------------

function SkipRangeSheet({ project, range }: { project: Project; range: TimeRange }) {
  const updateSkipRange = useStore((s) => s.updateSkipRange)
  const removeSkipRange = useStore((s) => s.removeSkipRange)
  const toast = useStore((s) => s.toast)
  const [label, setLabel] = useSyncedDraft(`${range.id}:label`, range.label)
  const [start, setStart] = useSyncedDraft(`${range.id}:start`, range.start.toFixed(2))
  const [end, setEnd] = useSyncedDraft(`${range.id}:end`, range.end.toFixed(2))

  const duration = Math.max(0, range.end - range.start)
  const lines = project.segments.filter((s) => s.start < range.end && s.end > range.start).length

  const save = (patch: Partial<Pick<TimeRange, 'start' | 'end' | 'label'>>) => {
    void updateSkipRange(range.id, patch).catch((err: unknown) => {
      toast('error', errorText(err, 'Failed to update the range.'))
    })
  }
  const commitTiming = () => {
    const s = Number(start)
    const e = Number(end)
    const dur = project.media?.duration ?? Infinity
    if (!Number.isFinite(s) || !Number.isFinite(e) || s < 0 || e <= s || e > dur) {
      toast('error', `Range must satisfy 0 ≤ start < end ≤ ${formatTime(dur, true)}.`)
      setStart(range.start.toFixed(2))
      setEnd(range.end.toFixed(2))
      return
    }
    const patch: Partial<Pick<TimeRange, 'start' | 'end'>> = {}
    if (Math.abs(s - range.start) > 1e-6) patch.start = s
    if (Math.abs(e - range.end) > 1e-6) patch.end = e
    if (Object.keys(patch).length > 0) save(patch)
  }
  const blurOnEnter = (e: KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter') e.currentTarget.blur() }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <Header
        title="Keep original"
        right={`${formatTime(range.start, true).slice(0, -2)} – ${formatTime(range.end, true).slice(0, -2)}`}
      />
      <div style={bodyStyle}>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', padding: '2px 0 6px', lineHeight: 1.5 }}>
          The mix keeps the original audio here; lines inside are neither translated nor voiced.
        </div>
        <PropertyRow label="Label" labelWidth={LABEL_W}>
          <TextInput
            ariaLabel="Range label"
            value={label}
            onChange={setLabel}
            onBlur={() => { if (label !== range.label) save({ label }) }}
            onKeyDown={blurOnEnter}
            placeholder="e.g. opening song"
            style={fieldStyle}
          />
        </PropertyRow>
        <PropertyRow label="Start" labelWidth={LABEL_W}>
          <TextInput
            ariaLabel="Range start (seconds)" value={start} onChange={setStart} onBlur={commitTiming} onKeyDown={blurOnEnter}
            style={{ ...fieldStyle, fontFamily: 'var(--mono)', fontSize: 11 }}
          />
        </PropertyRow>
        <PropertyRow label="End" labelWidth={LABEL_W}>
          <TextInput
            ariaLabel="Range end (seconds)" value={end} onChange={setEnd} onBlur={commitTiming} onKeyDown={blurOnEnter}
            style={{ ...fieldStyle, fontFamily: 'var(--mono)', fontSize: 11 }}
          />
        </PropertyRow>
        <PropertyRow label="Duration" labelWidth={LABEL_W}><Value mono>{duration.toFixed(2)}s</Value></PropertyRow>
        <PropertyRow label="Lines" labelWidth={LABEL_W}>
          <Value>{lines} inside{lines > 0 ? <span style={{ color: 'var(--text-dim)' }}> · kept original</span> : null}</Value>
        </PropertyRow>
      </div>
      <div style={{
        height: 40, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 6, padding: '0 8px',
        borderTop: '1px solid var(--border)',
      }}>
        <Button
          variant="danger" size="sm" style={{ flex: 1 }}
          title="Remove this range; its lines are dubbed again"
          onClick={() => void removeSkipRange(range.id).catch((err: unknown) => toast('error', errorText(err, 'Failed to remove the range.')))}
        >
          Remove
        </Button>
      </div>
    </div>
  )
}
