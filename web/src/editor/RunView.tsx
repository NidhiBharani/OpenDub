// Live progress for a multi-stage pipeline run: overall bar, per-stage rail, the line being worked
// on, the script as it lands (heard → translated → voiced), and things to do while waiting.
import type { CSSProperties } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import { Button, IconButton, StatusDot } from '../components/primitives'
import { speakerColor, useLatestRun, useProject, useStore } from '../state/store'
import { STAGE_LABELS, STAGE_ORDER } from '../types'
import type { Job, Project, Segment, Speaker, StageKey, StageStatus, StepState } from '../types'
import { playTake, useCurrentlyPlayingTake } from './Inspector'

const HEADLINES: Record<StageKey, string> = {
  ingest: 'Opening the reel',
  separate: 'Lifting voices off the music',
  analyze: 'Studying the picture and the mix',
  transcribe: 'Listening to every line',
  translate: 'Writing the script in its new language',
  synthesize: 'Giving every line its new voice',
  mix: 'Setting the dub into the scene',
  lipsync: 'Matching lips to the new lines',
  review: 'Checking its own work',
  render: 'Printing the final cut',
}

const clock = (sec: number): string => {
  const s = Math.max(0, Math.round(sec))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

const isVoiced = (sg: Segment): boolean => sg.active_take_id != null && !sg.synth_dirty
const isTranslated = (sg: Segment): boolean => sg.translated_text.trim() !== '' && !sg.translate_dirty

/** Ticks once a second while `on`, so elapsed/remaining stay live between server events. */
function useNow(on: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!on) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [on])
  return now
}

export function RunView() {
  const project = useProject()
  const run = useLatestRun()
  const setRunViewOpen = useStore((s) => s.setRunViewOpen)
  const toast = useStore((s) => s.toast)
  const applyJob = useStore((s) => s.applyJob)
  const [notify, setNotify] = useState(false)

  const active = !!run && (run.status === 'queued' || run.status === 'running')
  const now = useNow(active)

  // Desktop notification when the run we were watching ends.
  const wasActive = useRef(active)
  useEffect(() => {
    if (wasActive.current && !active && run && notify && typeof Notification !== 'undefined'
      && Notification.permission === 'granted') {
      new Notification(run.status === 'done' ? 'OpenDub — the final cut is ready' : 'OpenDub — the run stopped', {
        body: project?.name,
      })
    }
    wasActive.current = active
  }, [active, run, notify, project?.name])

  const sorted = useMemo(
    () => [...(project?.segments ?? [])].sort((a, b) => a.start - b.start),
    [project?.segments],
  )

  if (!project || !run) {
    return (
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
        <div style={{ fontSize: 16, fontWeight: 600 }}>No run to show yet</div>
        <Button onClick={() => setRunViewOpen(false)}>Open the editor</Button>
      </div>
    )
  }

  const total = Math.max(1, run.stages.length)
  const stageIdx = run.stage ? Math.max(0, run.stages.indexOf(run.stage)) : 0
  const done = run.status === 'done'
  const failed = run.status === 'error'
  const cancelled = run.status === 'cancelled'
  const progress = done ? 1 : Math.min(1, Math.max(0, run.progress))
  const stageFrac = Math.min(1, Math.max(0, progress * total - stageIdx))

  const startMs = Date.parse(run.started_at ?? run.created_at)
  const endMs = run.finished_at ? Date.parse(run.finished_at) : now
  const elapsed = Math.max(0, (endMs - startMs) / 1000)
  const remaining = active && progress > 0.04 && elapsed >= 3 ? `about ${clock((elapsed * (1 - progress)) / progress)} left` : active ? 'estimating…' : ''

  // The line in the booth right now: synthesize walks segments in order, skipping finished ones.
  const current = active && run.stage === 'synthesize' ? project.segments.find((sg) => !isVoiced(sg)) ?? null : null
  const currentSpeaker = current ? project.speakers.find((sp) => sp.id === current.speaker_id) : undefined
  const voicedN = sorted.filter(isVoiced).length

  const accent = done ? 'var(--ok)' : failed || cancelled ? 'var(--err)' : run.stage === 'synthesize' ? 'var(--ok)' : 'var(--accent)'
  const kicker = done ? 'PIPELINE COMPLETE'
    : failed ? 'RUN FAILED' : cancelled ? 'RUN CANCELLED'
    : run.status === 'queued' ? 'QUEUED' : `PIPELINE RUNNING · STAGE ${stageIdx + 1} OF ${total}`
  const headline = done ? 'Pipeline complete'
    : failed ? `The run stopped at ${run.stage ? STAGE_LABELS[run.stage] : 'start'}.`
    : cancelled ? 'Run cancelled'
    : current && currentSpeaker ? `Giving ${currentSpeaker.name} a new voice`
    : run.stage ? HEADLINES[run.stage] : 'Getting ready'

  async function cancel() {
    try { await api.cancelJob(run!.id) } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to cancel job')
    }
  }
  async function runAgain() {
    try { applyJob(await api.runPipeline(project!.id)) } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to run pipeline')
    }
  }
  async function toggleNotify(on: boolean) {
    if (on && typeof Notification !== 'undefined' && Notification.permission === 'default') {
      await Notification.requestPermission()
    }
    if (on && (typeof Notification === 'undefined' || Notification.permission !== 'granted')) {
      toast('error', 'Notifications are blocked for this site in your browser.')
      return
    }
    setNotify(on)
  }

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      {/* headline + overall progress */}
      <div style={{
        flexShrink: 0, padding: '12px 16px 10px', display: 'flex', flexDirection: 'column', gap: 10,
        borderBottom: '1px solid var(--border)', background: 'var(--bg-panel)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
            <div className="eyebrow" style={{ color: accent }}>{kicker}</div>
            <h1 style={{ margin: 0, fontSize: 16, fontWeight: 600, lineHeight: 1.25 }}>{headline}</h1>
          </div>
          <div style={{ flex: 1 }} />
          <div className="timecode" style={{ display: 'flex', alignItems: 'baseline', gap: 10, color: 'var(--text)' }}>
            <span style={{ fontSize: 18, color: 'var(--text)' }}>{Math.floor(progress * 100)}%</span>
            <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
              {clock(elapsed)} elapsed{remaining && ` · ${remaining}`}
            </span>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {active
              ? <Button variant="danger" onClick={() => void cancel()}>Cancel run</Button>
              : <Button onClick={() => void runAgain()}>{done ? 'Run again' : 'Retry'}</Button>}
            <Button variant={active ? 'default' : 'primary'} onClick={() => setRunViewOpen(false)}>
              Open the editor
            </Button>
          </div>
        </div>
        <div
          role="progressbar" aria-label="Overall pipeline progress"
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.floor(progress * 100)}
          style={{ display: 'flex', gap: 3 }}
        >
          {run.stages.map((key, i) => {
            const st: StageStatus = project.stages?.[key]?.status ?? 'pending'
            const fill = st === 'done' || st === 'skipped' || i < stageIdx || done ? 1 : i === stageIdx && active ? stageFrac : 0
            const isNow = active && i === stageIdx
            return (
              <div key={key} style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
                <div style={{ height: 4, borderRadius: 2, background: 'var(--border-strong)', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', borderRadius: 2, width: `${fill * 100}%`, transition: 'width 300ms ease-out',
                    background: st === 'error' ? 'var(--err)' : st === 'skipped' ? 'var(--text-faint)' : isNow ? 'var(--accent)' : 'var(--ok)',
                  }} />
                </div>
                <span style={{
                  fontSize: 11, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  color: isNow ? 'var(--accent-text)' : fill === 1 ? 'var(--text)' : 'var(--text-dim)',
                }}>
                  {STAGE_LABELS[key]}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      <div style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: '300px minmax(0, 1fr) 320px' }}>
        <StageRail project={project} run={run} active={active} stageFrac={stageFrac} />

        <section aria-label="Live progress" style={{
          minWidth: 0, minHeight: 0, padding: 12, display: 'flex', flexDirection: 'column', gap: 10,
        }}>
          <NowCard
            run={run} active={active} accent={accent} stageFrac={stageFrac}
            current={current} speaker={currentSpeaker}
            speakerTint={current ? speakerColor(project, current.speaker_id) : undefined}
          />
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span className="eyebrow">Script</span>
            <span style={{ fontSize: 11, color: 'var(--text-faint)' }}>heard → translated → voiced</span>
          </div>
          <Feed project={project} segments={sorted} currentId={current?.id ?? null} />
        </section>

        <section aria-label="While you wait" style={{
          minHeight: 0, overflowY: 'auto', padding: 12, background: 'var(--bg-panel)',
          borderLeft: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10,
        }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span className="eyebrow">{active ? 'While you wait · Cast' : 'Cast'}</span>
            <span style={{ fontSize: 12, lineHeight: 1.45, color: 'var(--text-dim)' }}>
              {project.speakers.length > 0
                ? `${project.speakers.length} ${project.speakers.length === 1 ? 'voice' : 'voices'} found. Name your cast and check each reference clip — the dub clones from these.`
                : 'Once the voices are told apart you can name your cast here.'}
            </span>
          </div>
          {project.speakers.length === 0 ? (
            <div style={{
              height: 64, borderRadius: 'var(--r-md)', border: '1px dashed var(--border-dashed)', fontSize: 12,
              display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)',
            }}>
              Listening for distinct voices…
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {project.speakers.map((sp) => (
                <CastCard key={sp.id} project={project} speaker={sp} segments={sorted} />
              ))}
            </div>
          )}
          <div style={{ flex: 1 }} />
          <div style={{
            padding: 10, borderRadius: 'var(--r-md)', background: 'var(--bg-raised)', border: '1px solid var(--border)',
            display: 'flex', flexDirection: 'column', gap: 8,
          }}>
            <div style={{ fontSize: 12, lineHeight: 1.45 }}>
              {done ? 'Every stage has finished. Review the dub, fix any line, and only that line re-runs.'
                : voicedN > 0 ? `${voicedN} of ${sorted.length} lines already have their new voice. No need to wait for the rest — edits only re-run what they touch.`
                : 'Finished lines can be reviewed and edited while the rest are still running.'}
            </div>
            <Button variant={voicedN > 0 ? 'primary' : 'default'} onClick={() => setRunViewOpen(false)}>
              {done ? 'Open the editor' : voicedN > 0 ? `Review ${voicedN} voiced ${voicedN === 1 ? 'line' : 'lines'}` : 'Open the editor anyway'}
            </Button>
          </div>
          {active && (
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, minHeight: 24, fontSize: 12, color: 'var(--text-dim)', cursor: 'pointer' }}>
              <input
                type="checkbox" checked={notify} onChange={(e) => void toggleNotify(e.target.checked)}
                style={{ width: 14, height: 14, accentColor: 'var(--accent)', margin: 0 }}
              />
              Tell me when the final cut is ready
            </label>
          )}
        </section>
      </div>
    </div>
  )
}

// ---- stage rail ---------------------------------------------------------------

function StageRail({ project, run, active, stageFrac }: {
  project: Project; run: Job; active: boolean; stageFrac: number
}) {
  return (
    <section aria-label="Stages" style={{
      minHeight: 0, overflowY: 'auto', padding: '8px 8px', background: 'var(--bg-panel)',
      borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 1,
    }}>
      {STAGE_ORDER.map((key) => {
        const state = project.stages?.[key]
        const status: StageStatus = state?.status ?? 'pending'
        const isRun = active && run.stage === key && status === 'running'
        const inRun = run.stages.includes(key)
        const detail = isRun ? (run.message || 'Working…')
          : status === 'error' ? (state?.detail || 'Failed')
          : status === 'skipped' ? (state?.detail || 'Skipped')
          : status === 'done' ? (state?.detail || 'Done')
          : status === 'queued' ? 'Waiting its turn'
          : status === 'dirty' ? 'Out of date' : inRun ? 'Waiting' : 'Not part of this run'
        return (
          <div key={key} style={{
            display: 'flex', gap: 8, padding: '5px 6px', borderRadius: 'var(--r-md)',
            background: isRun ? 'var(--bg-raised)' : 'transparent',
            border: `1px solid ${isRun ? 'var(--border)' : 'transparent'}`,
          }}>
            <div style={{ width: 14, height: 18, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <StageGlyph status={status} running={isRun} />
            </div>
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <span style={{
                  fontSize: 12, fontWeight: 600,
                  color: status === 'done' || isRun ? 'var(--text)' : status === 'error' ? 'var(--err-text)' : 'var(--text-dim)',
                }}>
                  {STAGE_LABELS[key]}
                </span>
                <div style={{ flex: 1 }} />
                {isRun && <span className="timecode" style={{ fontSize: 11, color: 'var(--accent-text)' }}>{Math.round(stageFrac * 100)}%</span>}
              </div>
              <div style={{ fontSize: 11, lineHeight: 1.4, color: status === 'error' ? 'var(--err-text)' : 'var(--text-dim)', overflowWrap: 'anywhere' }}>
                {detail}
              </div>
              {isRun && (
                <div style={{ marginTop: 3, height: 3, borderRadius: 2, background: 'var(--border-strong)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', borderRadius: 2, width: `${stageFrac * 100}%`, background: 'var(--accent)', transition: 'width 300ms ease-out' }} />
                </div>
              )}
              <StepList steps={state?.steps} />
            </div>
          </div>
        )
      })}
    </section>
  )
}

const STEP_COLORS: Record<StepState['status'], string> = {
  done: 'var(--ok)', skipped: 'var(--text-faint)', error: 'var(--err)',
}

/** The capabilities that ran inside a stage — one row each, in capability-id order. */
function StepList({ steps }: { steps: Record<string, StepState> | undefined }) {
  const entries = Object.entries(steps ?? {}).sort(([a], [b]) => a.localeCompare(b))
  if (entries.length === 0) return null
  return (
    <ul style={{ listStyle: 'none', margin: '6px 0 2px', padding: 0, display: 'flex', flexDirection: 'column', gap: 3 }}>
      {entries.map(([capId, step]) => (
        <li key={capId} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
          <StatusDot color={STEP_COLORS[step.status]} hollow={step.status === 'skipped'} />
          <span className="timecode" style={{ fontSize: 10, flexShrink: 0 }}>{capId}</span>
          <span style={{
            minWidth: 0, flex: 1, color: step.status === 'error' ? 'var(--err-text)' : 'var(--text-dim)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }} title={`${step.provider_id}${step.detail ? ` — ${step.detail}` : ''}`}>
            {step.detail || step.status}
          </span>
          <span style={{
            flexShrink: 0, maxWidth: 110, color: 'var(--text-faint)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {step.provider_id.replace(/^[^.]+\./, '')}
          </span>
        </li>
      ))}
    </ul>
  )
}

function StageGlyph({ status, running }: { status: StageStatus; running: boolean }) {
  if (running) {
    return (
      <div style={{
        width: 12, height: 12, borderRadius: '50%', border: '2px solid var(--border-strong)',
        borderTopColor: 'var(--accent)', animation: 'od-spin 1100ms linear infinite',
      }} />
    )
  }
  if (status === 'done') return <StatusDot color="var(--ok)" size={8} />
  if (status === 'error') return <StatusDot color="var(--err)" size={8} />
  if (status === 'skipped') return <span style={{ width: 8, height: 1.5, background: 'var(--text-faint)' }} />
  if (status === 'dirty') return <StatusDot color="var(--warn)" size={8} />
  return <StatusDot color={status === 'queued' ? 'var(--accent)' : 'var(--text-faint)'} size={8} hollow />
}

// ---- "now" card ---------------------------------------------------------------

const rnd = (i: number): number => { const v = Math.sin(i * 12.9898) * 43758.5453; return v - Math.floor(v) }

function NowCard({ run, active, accent, stageFrac, current, speaker, speakerTint }: {
  run: Job; active: boolean; accent: string; stageFrac: number
  current: Segment | null; speaker: Speaker | undefined; speakerTint: string | undefined
}) {
  // Within synthesize the bar tracks the single line in the booth ("… segment 3/11").
  const m = /(\d+)\s*\/\s*(\d+)/.exec(run.message)
  const lineFrac = current && m ? Math.min(1, Math.max(0, stageFrac * Number(m[2]) - (Number(m[1]) - 1))) : stageFrac
  const fill = active ? lineFrac : run.status === 'done' ? 1 : 0
  const bars = useMemo(() => Array.from({ length: 96 }, (_, i) => 3 + Math.sin((i / 95) * Math.PI) * rnd(i + 3) * 22), [])

  return (
    <div aria-live="polite" style={{
      flexShrink: 0, minHeight: 112, padding: 12, borderRadius: 'var(--r-md)', background: 'var(--bg-panel)',
      border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 8,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
        <StatusDot color={accent} pulse={active} />
        <span className="eyebrow" style={{ color: accent }}>
          {run.status === 'done' ? 'FINAL CUT READY' : run.stage ? STAGE_LABELS[run.stage].toUpperCase() : run.status.toUpperCase()}
        </span>
        {speaker && <span style={{ color: speakerTint, fontWeight: 500 }}>{speaker.name}</span>}
        <div style={{ flex: 1 }} />
        <span className="timecode">{active ? run.message : ''}</span>
      </div>
      {current ? (
        <>
          <div style={{ fontSize: 12, lineHeight: 1.4, color: 'var(--text-dim)' }}>{current.source_text}</div>
          <div style={{ fontSize: 14, fontWeight: 500, lineHeight: 1.35 }}>{current.translated_text || '…'}</div>
        </>
      ) : (
        <div style={{ fontSize: 14, fontWeight: 500, lineHeight: 1.35, overflowWrap: 'anywhere' }}>
          {run.status === 'done' ? 'dubbed.mp4 is ready to watch.'
            : run.status === 'error' ? (run.error || 'Something went wrong.')
            : run.status === 'cancelled' ? 'Nothing was lost — finished lines are kept.'
            : run.message ? `${run.message.charAt(0).toUpperCase()}${run.message.slice(1)}…` : 'Waiting for a free worker…'}
        </div>
      )}
      <div style={{ flex: 1 }} />
      <div aria-hidden="true" style={{ display: 'flex', alignItems: 'center', gap: 2, height: 20 }}>
        {bars.map((h, i) => {
          const lit = i / bars.length < fill
          return (
            <span key={i} style={{
              flex: 1, borderRadius: 1, height: lit ? h : 2,
              background: lit ? accent : 'var(--border-strong)', transition: 'height 200ms ease-out',
            }} />
          )
        })}
      </div>
    </div>
  )
}

// ---- script feed ---------------------------------------------------------------

function Feed({ project, segments, currentId }: { project: Project; segments: Segment[]; currentId: string | null }) {
  const playingId = useCurrentlyPlayingTake()
  const rowRefs = useRef(new Map<string, HTMLDivElement>())

  useEffect(() => {
    if (currentId) rowRefs.current.get(currentId)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [currentId])

  if (segments.length === 0) {
    return (
      <div style={{
        flex: 1, minHeight: 80, borderRadius: 'var(--r-md)', border: '1px dashed var(--border-dashed)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center',
        padding: '0 48px', fontSize: 12, color: 'var(--text-dim)',
      }}>
        Lines appear here the moment they are heard — then pick up their translation and their new voice, one by one.
      </div>
    )
  }

  const step = (on: boolean, color: string): CSSProperties => ({
    width: 14, height: 3, borderRadius: 2, background: on ? color : 'var(--border-strong)',
  })

  return (
    <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 0, border: '1px solid var(--border)', borderRadius: 'var(--r-md)', background: 'var(--bg-panel)' }}>
      {segments.map((sg, i) => {
        const speaker = project.speakers.find((sp) => sp.id === sg.speaker_id)
        const color = speakerColor(project, sg.speaker_id)
        const translated = isTranslated(sg)
        const voiced = isVoiced(sg)
        const isCur = sg.id === currentId
        const take = sg.takes.find((t) => t.id === sg.active_take_id)
        const status = isCur ? 'voicing…' : voiced ? 'ready to play' : translated ? 'translated' : 'heard'
        return (
          <div
            key={sg.id}
            ref={(el) => { if (el) rowRefs.current.set(sg.id, el); else rowRefs.current.delete(sg.id) }}
            style={{
              flexShrink: 0, minHeight: 44, padding: '4px 8px 4px 10px', display: 'flex', alignItems: 'center', gap: 10,
              borderBottom: '1px solid var(--border-subtle)',
              background: isCur ? 'var(--accent-dim)' : 'transparent',
              boxShadow: isCur ? 'inset 2px 0 0 var(--accent)' : 'none',
            }}
          >
            <span className="timecode" style={{ width: 20, fontSize: 10 }}>{String(i + 1).padStart(2, '0')}</span>
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 1 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, fontWeight: 600, color }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: color }} />
                {speaker?.name ?? 'Voice ?'}
              </span>
              <span style={{ fontSize: 12, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {sg.translated_text || sg.source_text}
              </span>
              {sg.translated_text && (
                <span style={{ fontSize: 11, color: 'var(--text-dim)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {sg.source_text}
                </span>
              )}
            </div>
            <div
              aria-label={`heard, ${translated ? 'translated' : 'not translated'}, ${voiced ? 'voiced' : 'not voiced'}`}
              style={{ display: 'flex', gap: 4, flexShrink: 0 }}
            >
              <span title="Heard" style={step(true, 'var(--accent)')} />
              <span title="Translated" style={step(translated, 'var(--text)')} />
              <span title="Voiced" style={step(voiced, 'var(--ok)')} />
            </div>
            <span style={{
              width: 76, flexShrink: 0, textAlign: 'right', fontSize: 11,
              color: isCur ? 'var(--accent-text)' : voiced ? 'var(--ok)' : 'var(--text-dim)',
            }}>
              {status}
            </span>
            {voiced && take ? (
              <IconButton
                name={playingId === take.id ? 'stop' : 'play'}
                title={playingId === take.id ? 'Stop' : `Play line ${i + 1}`}
                active={playingId === take.id}
                onClick={() => playTake(api.mediaUrl(project.id, take.path), take.id)}
              />
            ) : (
              <span style={{ width: 24, height: 24, flexShrink: 0 }} />
            )}
          </div>
        )
      })}
    </div>
  )
}

// ---- cast ---------------------------------------------------------------------

function CastCard({ project, speaker, segments }: { project: Project; speaker: Speaker; segments: Segment[] }) {
  const renameSpeaker = useStore((s) => s.renameSpeaker)
  const toast = useStore((s) => s.toast)
  const playingId = useCurrentlyPlayingTake()
  const [draft, setDraft] = useState(speaker.name)
  useEffect(() => setDraft(speaker.name), [speaker.name])

  const color = speakerColor(project, speaker.id)
  const mine = segments.filter((sg) => sg.speaker_id === speaker.id)
  const voiced = mine.filter(isVoiced).length
  const refId = `ref:${speaker.id}`

  function commit() {
    const name = draft.trim()
    if (!name || name === speaker.name) { setDraft(speaker.name); return }
    void renameSpeaker(speaker.id, name).catch((e: unknown) => {
      toast('error', e instanceof Error ? e.message : 'Failed to rename speaker')
      setDraft(speaker.name)
    })
  }

  return (
    <div style={{
      padding: '6px 8px', borderRadius: 'var(--r-md)', background: 'var(--bg-raised)', border: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column', gap: 6,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <IconButton
          name={playingId === refId ? 'stop' : 'play'}
          disabled={!speaker.reference_path}
          title={speaker.reference_path ? `Play ${speaker.name}'s reference clip` : 'No reference clip yet'}
          active={playingId === refId}
          onClick={() => speaker.reference_path && playTake(api.mediaUrl(project.id, speaker.reference_path), refId)}
          style={{ color }}
        />
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 1 }}>
          <input
            type="text"
            aria-label="Character name"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
            style={{
              width: '100%', height: 22, padding: '0 6px', marginLeft: -6, borderRadius: 'var(--r-md)', fontSize: 12, fontWeight: 600,
              border: '1px solid transparent', background: 'transparent', color,
            }}
          />
          <span className="timecode" style={{ fontSize: 11 }}>{mine.length} lines · {voiced} voiced</span>
        </div>
      </div>
      <div style={{ height: 2, borderRadius: 1, background: 'var(--border-strong)', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${mine.length ? (voiced / mine.length) * 100 : 0}%`, background: color, transition: 'width 300ms ease-out' }} />
      </div>
    </div>
  )
}
