// Segment Inspector: header (timecode/speaker), source text, translation
// (with char/s budget), emotion hint, takes list, and regenerate actions.
import type { CSSProperties, KeyboardEvent, ReactNode } from 'react'
import { useEffect, useMemo, useState } from 'react'
import { Badge, Button, Select, TextInput } from '../components/primitives'
import { api, ApiError } from '../api/client'
import { useProject, useSelectedSegment, useStore } from '../state/store'
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

function playTake(url: string, id: string) {
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

function useCurrentlyPlayingTake(): string | null {
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
  minHeight: 60,
  resize: 'vertical',
  background: 'var(--bg)',
  border: '1px solid var(--border-strong)',
  borderRadius: 'var(--r-md)',
  padding: '8px 10px',
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
        <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-dim)' }}>{label}</span>
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
        color: 'var(--text-faint)', fontSize: 13,
      }}>
        Select a segment…
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
        flex: 1, minHeight: 0, overflowY: 'auto', padding: 14,
        display: 'flex', flexDirection: 'column', gap: 18,
      }}>
        <HeaderSection segment={segment} project={project} commit={commit} />
        <SourceTextSection segment={segment} commit={commit} />
        <TranslationSection segment={segment} commit={commit} />
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
  const speaker = project.speakers.find((sp) => sp.id === segment.speaker_id)
  const duration = Math.max(0, segment.end - segment.start)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span className="timecode">{formatTime(segment.start, true)} → {formatTime(segment.end, true)}</span>
        <Badge>{duration.toFixed(2)}s</Badge>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          width: 10, height: 10, borderRadius: '50%', flexShrink: 0,
          background: speaker?.color ?? 'var(--text-faint)',
        }} />
        <Select
          value={segment.speaker_id}
          onChange={(v) => commit({ speaker_id: v })}
          options={project.speakers.map((sp) => ({ value: sp.id, label: sp.name }))}
        />
      </div>
    </div>
  )
}

// ---- source text ------------------------------------------------------------

function SourceTextSection({ segment, commit }: { segment: Segment; commit: (p: SegmentPatch) => void }) {
  const [draft, setDraft] = useSyncedDraft(segment.id, segment.source_text)

  const save = () => {
    if (draft !== segment.source_text) commit({ source_text: draft })
  }

  return (
    <Section label="Source text">
      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={save}
        onKeyDown={commitOnCmdEnter}
        rows={3}
        style={textareaStyle}
      />
    </Section>
  )
}

// ---- translation (with chars-per-second budget) ----------------------------

function TranslationSection({ segment, commit }: { segment: Segment; commit: (p: SegmentPatch) => void }) {
  const [draft, setDraft] = useSyncedDraft(segment.id, segment.translated_text)

  const save = () => {
    if (draft !== segment.translated_text) commit({ translated_text: draft })
  }

  const duration = Math.max(0, segment.end - segment.start)
  const chars = draft.length
  const budget = duration * 15
  const ratio = budget > 0 ? chars / budget : (chars > 0 ? Infinity : 0)
  const color = ratio > 1.15 ? 'var(--err)' : ratio > 1 ? 'var(--warn)' : 'var(--ok)'

  return (
    <Section
      label="Translation"
      extra={
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {segment.translate_dirty && <Badge color="var(--warn)">outdated</Badge>}
          <span style={{ fontSize: 11, color, fontVariantNumeric: 'tabular-nums' }}>
            {chars} ch / ~{Math.round(budget)} max
          </span>
        </div>
      }
    >
      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={save}
        onKeyDown={commitOnCmdEnter}
        rows={4}
        style={textareaStyle}
      />
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
    <Section label="Emotion">
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
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8, padding: '6px 0',
      borderBottom: '1px solid var(--border)',
    }}>
      <input
        type="radio"
        name={`active-take-${take.id.slice(0, 4)}`}
        checked={active}
        onChange={onSelect}
        style={{ accentColor: 'var(--accent)', flexShrink: 0 }}
      />
      <span style={{ fontSize: 12, fontWeight: 500, flexShrink: 0 }}>Take {n}</span>
      <span style={{
        fontSize: 11, color: 'var(--text-dim)', overflow: 'hidden', textOverflow: 'ellipsis',
        whiteSpace: 'nowrap', minWidth: 0,
      }}>
        {take.provider_id}
      </span>
      <span className="timecode">{take.duration.toFixed(2)}s</span>
      {rateOff && (
        <span title={`time-stretched ×${take.rate_factor.toFixed(2)} to fit slot`}>
          <Badge color="var(--warn)">×{take.rate_factor.toFixed(2)}</Badge>
        </span>
      )}
      <div style={{ flex: 1 }} />
      <Button
        variant="ghost"
        title={playing ? 'Stop' : 'Play take'}
        onClick={() => playTake(api.mediaUrl(pid, take.path), take.id)}
      >
        {playing ? '⏹' : '▶'}
      </Button>
    </div>
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
      padding: 14, borderTop: '1px solid var(--border)',
    }}>
      {segment.synth_dirty && <Badge color="var(--warn)">voice outdated</Badge>}
      <div style={{ flex: 1 }} />
      <Button
        variant="default"
        disabled={hasActiveJob}
        title="translate again, then re-voice"
        onClick={() => void run(['translate', 'synthesize'])}
      >
        Re-translate
      </Button>
      <Button
        variant="primary"
        disabled={hasActiveJob}
        title="re-voice with the current translation"
        onClick={() => void run(['synthesize'])}
      >
        Re-voice
      </Button>
    </div>
  )
}
