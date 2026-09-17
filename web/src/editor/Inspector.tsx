// Segment Inspector: header (timecode/speaker), source text, translation
// (with char/s budget), emotion hint, takes list, and regenerate actions.
import type { CSSProperties, KeyboardEvent, ReactNode } from 'react'
import { useEffect, useMemo, useState } from 'react'
import { Icon } from '../components/Icon'
import { Button, Select, TextInput } from '../components/primitives'
import { api, ApiError } from '../api/client'
import { speakerColor, useProject, useSelectedSegment, useStore } from '../state/store'
import type { Toast } from '../state/store'
import { formatTime } from '../types'
import type { Job, Project, Segment, Take } from '../types'

type SegmentPatch = Parameters<typeof api.updateSegment>[2]

// ---- module-level shared take-preview audio element ----------------------
// A single <audio> shared across the whole app so that starting a new take
// preview always stops whatever was playing before, regardless of re-renders.

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

// ---- shared styles ---------------------------------------------------------

const textareaStyle: CSSProperties = {
  width: '100%',
  minHeight: 56,
  resize: 'vertical',
  background: 'var(--bg)',
  border: '1px solid var(--border-strong)',
  borderRadius: 'var(--r-md)',
  padding: '9px 11px',
  fontSize: 13,
  fontFamily: 'inherit',
  lineHeight: 1.5,
}

function commitOnCmdEnter(e: KeyboardEvent<HTMLTextAreaElement>) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault()
    e.currentTarget.blur() // triggers onBlur → save
  }
}

/** Local text draft that follows its server value: reset when the segment changes,
 *  and re-seeded when the server value changes externally (e.g. after Re-translate) —
 *  but never while the user has unsaved local edits (draft !== last-seen base). */
function useSyncedDraft(segmentId: string, serverValue: string): [string, (v: string) => void] {
  const [state, setState] = useState({ segmentId, base: serverValue, draft: serverValue })
  if (state.segmentId !== segmentId) {
    // segment switched: always reset
    setState({ segmentId, base: serverValue, draft: serverValue })
  } else if (state.base !== serverValue) {
    // server value changed under us: adopt it unless the user has typed something
    setState({
      segmentId,
      base: serverValue,
      draft: state.draft === state.base ? serverValue : state.draft,
    })
  }
  const setDraft = (draft: string) => setState((s) => ({ ...s, draft }))
  return [state.draft, setDraft]
}

function Section({ label, extra, children }: { label: string; extra?: ReactNode; children: ReactNode }) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6, gap: 8 }}>
        <span className="eyebrow">{label}</span>
        {extra}
      </div>
      {children}
    </div>
  )
}

// ---- main component ---------------------------------------------------------

export function Inspector() {
  const segment = useSelectedSegment()
  const project = useProject()
  const jobs = useStore((s) => s.jobs)
  const toast = useStore((s) => s.toast)
  const updateSegment = useStore((s) => s.updateSegment)
  const applyJob = useStore((s) => s.applyJob)

  if (!segment || !project) {
    return (
      <div style={{
        height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--text-dim)', fontSize: 13, padding: 24, textAlign: 'center',
      }}>
        Select a line in the script or on the timeline to edit it.
      </div>
    )
  }

  const commit = (patch: SegmentPatch) => {
    void updateSegment(segment.id, patch).catch((err: unknown) => {
      toast('error', err instanceof ApiError ? err.message : 'Failed to save changes.')
    })
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div style={{
        flex: 1, minHeight: 0, overflowY: 'auto', padding: '16px 18px',
        display: 'flex', flexDirection: 'column', gap: 16,
      }}>
        <HeaderSection segment={segment} project={project} commit={commit} />
        <SourceTextSection segment={segment} sourceLang={project.source_lang.toUpperCase()} commit={commit} />
        <TranslationSection segment={segment} targetLang={project.target_lang.toUpperCase()} commit={commit} />
        <EmotionSection segment={segment} commit={commit} />
        <TakesSection segment={segment} project={project} commit={commit} />
      </div>
      <ActionsRow project={project} segment={segment} jobs={jobs} toast={toast} applyJob={applyJob} />
    </div>
  )
}

// ---- header: timecode range, duration badge, speaker select ---------------

function HeaderSection({ segment, project, commit }: {
  segment: Segment; project: Project; commit: (p: SegmentPatch) => void
}) {
  const duration = Math.max(0, segment.end - segment.start)
  const color = speakerColor(project, segment.speaker_id)
  const lineNo = [...project.segments].sort((a, b) => a.start - b.start).findIndex((sg) => sg.id === segment.id) + 1
  const stale = segment.translate_dirty
    ? 'Source changed — re-translate to update this line.'
    : segment.synth_dirty ? 'Translation changed — re-voice to update the dub.' : ''

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
        <span className="serif" style={{ fontSize: 26, lineHeight: 1 }}>Line {String(lineNo).padStart(2, '0')}</span>
        <div style={{ flex: 1 }} />
        <span className="timecode" style={{ fontSize: 11 }}>
          {formatTime(segment.start, true).slice(0, -2)} – {formatTime(segment.end, true).slice(0, -2)} · {duration.toFixed(1)}s
        </span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ width: 9, height: 9, borderRadius: '50%', flexShrink: 0, background: color }} />
        <Select
          value={segment.speaker_id}
          onChange={(v) => commit({ speaker_id: v })}
          options={project.speakers.map((sp) => ({ value: sp.id, label: sp.name }))}
          style={{ color, fontWeight: 500 }}
        />
      </div>
      {stale && (
        <div role="status" style={{
          display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', borderRadius: 'var(--r-md)',
          background: 'var(--warn-dim)', color: 'var(--warn-text)', fontSize: 12,
        }}>
          <Icon name="warn" size={14} style={{ color: 'var(--warn)' }} />
          {stale}
        </div>
      )}
    </div>
  )
}

// ---- source text ------------------------------------------------------------

function SourceTextSection({ segment, sourceLang, commit }: {
  segment: Segment; sourceLang: string; commit: (p: SegmentPatch) => void
}) {
  const [draft, setDraft] = useSyncedDraft(segment.id, segment.source_text)

  const save = () => {
    if (draft !== segment.source_text) commit({ source_text: draft })
  }

  return (
    <Section label={`Source · ${sourceLang}`}>
      <textarea
        aria-label="Source text"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={save}
        onKeyDown={commitOnCmdEnter}
        rows={2}
        style={textareaStyle}
      />
    </Section>
  )
}

// ---- translation (with chars-per-second budget) ----------------------------

function TranslationSection({ segment, targetLang, commit }: {
  segment: Segment; targetLang: string; commit: (p: SegmentPatch) => void
}) {
  const [draft, setDraft] = useSyncedDraft(segment.id, segment.translated_text)

  const save = () => {
    if (draft !== segment.translated_text) commit({ translated_text: draft })
  }

  const duration = Math.max(0, segment.end - segment.start)
  const chars = draft.length
  const budget = Math.round(duration * 15)
  const ratio = budget > 0 ? chars / budget : (chars > 0 ? Infinity : 0)
  const color = ratio > 1.15 ? 'var(--err)' : ratio > 1 ? 'var(--warn)' : 'var(--ok)'
  const verdict = ratio > 1
    ? `over by ${chars - budget} — the take will be sped up`
    : 'fits the slot'

  return (
    <Section label={`Translation · ${targetLang}`}>
      <textarea
        aria-label="Translation"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={save}
        onKeyDown={commitOnCmdEnter}
        rows={3}
        style={{ ...textareaStyle, fontSize: 14 }}
      />
      <div style={{ height: 4, borderRadius: 2, background: 'var(--border-strong)', overflow: 'hidden', marginTop: 6 }}>
        <div style={{
          height: '100%', borderRadius: 2, background: color, transition: 'width var(--ease)',
          width: `${Math.min(100, Math.round(ratio * 100))}%`,
        }} />
      </div>
      <div className="timecode" style={{ fontSize: 11, color, marginTop: 6 }}>
        {chars} / {budget} chars · {verdict}
      </div>
    </Section>
  )
}

// ---- emotion hint ------------------------------------------------------------

function EmotionSection({ segment, commit }: { segment: Segment; commit: (p: SegmentPatch) => void }) {
  const [draft, setDraft] = useSyncedDraft(segment.id, segment.emotion)

  const save = () => {
    if (draft !== segment.emotion) commit({ emotion: draft })
  }

  return (
    <Section label="Emotion hint">
      <TextInput value={draft} onChange={setDraft} onBlur={save} placeholder="e.g. angry, whispering" />
    </Section>
  )
}

// ---- takes -------------------------------------------------------------------

function TakesSection({ segment, project, commit }: {
  segment: Segment; project: Project; commit: (p: SegmentPatch) => void
}) {
  const playingId = useCurrentlyPlayingTake()

  const ordered = useMemo(() => {
    const asc = [...segment.takes].sort((a, b) => a.created_at.localeCompare(b.created_at))
    return asc.map((take, i) => ({ take, n: i + 1 })).reverse()
  }, [segment.takes])

  return (
    <Section label={`Takes (${segment.takes.length})`}>
      {ordered.length === 0 ? (
        <div style={{ fontSize: 12, color: 'var(--text-faint)' }}>No takes yet.</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column' }}>
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
    </Section>
  )
}

function TakeRow({ take, n, pid, active, playing, onSelect }: {
  take: Take; n: number; pid: string; active: boolean; playing: boolean; onSelect: () => void
}) {
  const rateOff = Math.abs(take.rate_factor - 1) > 1e-3
  return (
    <label style={{
      display: 'flex', alignItems: 'center', gap: 10, minHeight: 44, padding: '0 6px 0 12px', marginBottom: 6,
      borderRadius: 'var(--r-md)', cursor: 'pointer', fontSize: 12,
      border: `1px solid ${active ? 'var(--ok)' : 'var(--border-strong)'}`,
      background: active ? 'var(--ok-dim)' : 'transparent',
    }}>
      <input
        type="radio"
        name={`active-take-${take.id.slice(0, 4)}`}
        checked={active}
        onChange={onSelect}
        style={{ accentColor: 'var(--ok)', flexShrink: 0 }}
      />
      <span style={{ fontWeight: 500, flexShrink: 0 }}>Take {n}</span>
      <span style={{
        color: 'var(--text-dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0,
      }}>
        {take.provider_id}
      </span>
      <div style={{ flex: 1 }} />
      <span className="timecode" style={{ fontSize: 11, whiteSpace: 'nowrap', flexShrink: 0 }} title={rateOff ? `time-stretched ×${take.rate_factor.toFixed(2)} to fit slot` : undefined}>
        {take.duration.toFixed(1)}s{rateOff && <span style={{ color: 'var(--warn-text)' }}> · ×{take.rate_factor.toFixed(2)}</span>}
      </span>
      {active && <span style={{ fontSize: 11, color: 'var(--ok)', whiteSpace: 'nowrap', flexShrink: 0 }}>in use</span>}
      <Button
        variant="ghost"
        title={playing ? 'Stop' : 'Play take'}
        onClick={() => playTake(api.mediaUrl(pid, take.path), take.id)}
        style={{ padding: '0 8px', color: 'var(--text)' }}
      >
        <Icon name={playing ? 'stop' : 'play'} size={12} />
      </Button>
    </label>
  )
}

// ---- sticky actions row -------------------------------------------------------

function ActionsRow({ project, segment, jobs, toast, applyJob }: {
  project: Project
  segment: Segment
  jobs: Record<string, Job>
  toast: (kind: Toast['kind'], text: string) => void
  applyJob: (j: Job) => void
}) {
  const hasActiveJob = Object.values(jobs).some(
    (j) => (j.status === 'queued' || j.status === 'running') && j.segment_id === segment.id,
  )

  const run = async (stages: ('translate' | 'synthesize')[]) => {
    try {
      const job = await api.regenerateSegment(project.id, segment.id, stages)
      applyJob(job)
    } catch (err) {
      toast('error', err instanceof ApiError ? err.message : 'Failed to start regeneration.')
    }
  }

  return (
    <div style={{
      flexShrink: 0, display: 'flex', alignItems: 'center', gap: 8,
      padding: '12px 18px', borderTop: '1px solid var(--border-strong)',
    }}>
      <Button
        variant="default"
        disabled={hasActiveJob}
        title="translate again, then re-voice"
        onClick={() => void run(['translate', 'synthesize'])}
        style={{ flex: 1, minHeight: 44 }}
      >
        Re-translate
      </Button>
      <Button
        variant="ok"
        disabled={hasActiveJob}
        title="re-voice with the current translation"
        onClick={() => void run(['synthesize'])}
        style={{ flex: 1, minHeight: 44 }}
      >
        {hasActiveJob ? 'Working…' : 'Re-voice line'}
      </Button>
    </div>
  )
}
