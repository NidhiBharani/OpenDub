// Pipeline chrome: the status strip under the title bar (ten compact stage steps; a click opens a
// popover — nothing runs on a bare click) and the title-bar Run control (primary button, or a
// progress pill with cancel while a job is active).
import { useState } from 'react'
import { api } from '../api/client'
import { Button, IconButton, MenuItem, Popover, StatusDot } from '../components/primitives'
import { Icon } from '../components/Icon'
import { useActiveJob, useProject, useStore } from '../state/store'
import { STAGE_LABELS, STAGE_ORDER } from '../types'
import type { StageKey, StageStatus, StepState } from '../types'

/** Colours for a stage status: `fill` for the dot, `hollow` when it should be an outline. */
export function stageDot(status: StageStatus): { fill: string; color: string; pulse?: boolean; hollow?: boolean } {
  switch (status) {
    case 'done': return { fill: 'var(--ok)', color: 'var(--text)' }
    case 'dirty': return { fill: 'var(--warn)', color: 'var(--warn-text)' }
    case 'error': return { fill: 'var(--err)', color: 'var(--err-text)' }
    case 'queued': return { fill: 'var(--accent)', color: 'var(--accent-text)', hollow: true }
    case 'running': return { fill: 'var(--accent)', color: 'var(--accent-text)', pulse: true }
    case 'skipped': return { fill: 'var(--text-faint)', color: 'var(--text-dim)' }
    default: return { fill: 'var(--text-faint)', color: 'var(--text-dim)', hollow: true }
  }
}

const STATUS_WORDS: Record<StageStatus, string> = {
  pending: 'Not run yet', queued: 'Queued', running: 'Running', done: 'Done',
  dirty: 'Out of date', error: 'Failed', skipped: 'Skipped',
}

const when = (iso: string | null): string => {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

/** 26px strip under the title bar: the ten stages as dot + label steps. */
export function StatusStrip() {
  const project = useProject()
  const activeJob = useActiveJob()
  const toast = useStore((s) => s.toast)
  const applyJob = useStore((s) => s.applyJob)
  const [open, setOpen] = useState<StageKey | null>(null)
  if (!project) return null
  const pid = project.id
  const changed = project.segments.filter((s) => s.translate_dirty || s.synth_dirty).length
  const busy = !!activeJob

  async function run(stages: StageKey[]) {
    setOpen(null)
    try {
      applyJob(await api.runPipeline(pid, stages))
    } catch (e) {
      toast('error', e instanceof Error ? e.message : `Failed to run ${STAGE_LABELS[stages[0]]}`)
    }
  }

  return (
    <nav
      aria-label="Pipeline stages"
      style={{
        height: 26, flexShrink: 0, display: 'flex', alignItems: 'stretch', gap: 12, padding: '0 12px',
        borderBottom: '1px solid var(--border)', background: 'var(--bg-panel)', minWidth: 0,
      }}
    >
      {STAGE_ORDER.map((key, i) => {
        const stage = project.stages?.[key]
        const status: StageStatus = stage?.status ?? 'pending'
        const dot = stageDot(status)
        const isActive = status === 'running' || (busy && activeJob?.stage === key)
        const isOpen = open === key
        return (
          <div key={key} style={{ position: 'relative', display: 'flex', alignItems: 'stretch', flexShrink: 0 }}>
            <button
              type="button"
              onClick={() => setOpen(isOpen ? null : key)}
              aria-expanded={isOpen}
              aria-haspopup="dialog"
              title={status === 'skipped' ? `${STAGE_LABELS[key]}: skipped` : `${STAGE_LABELS[key]}: ${STATUS_WORDS[status]}`}
              style={{
                display: 'flex', alignItems: 'center', gap: 5, padding: '0 2px', border: 'none',
                background: 'transparent', cursor: 'pointer', fontSize: 11, color: dot.color,
                opacity: status === 'skipped' ? 0.5 : 1, whiteSpace: 'nowrap',
                boxShadow: isActive ? 'inset 0 -2px 0 var(--accent)' : isOpen ? 'inset 0 -2px 0 var(--border-strong)' : 'none',
                fontWeight: isActive ? 600 : 500,
              }}
            >
              <StatusDot color={dot.fill} hollow={dot.hollow} pulse={dot.pulse} />
              {STAGE_LABELS[key]}
            </button>
            <Popover open={isOpen} onClose={() => setOpen(null)} align={i >= 6 ? 'right' : 'left'} width={280}>
              <StagePopover
                stageKey={key} status={status} detail={stage?.detail ?? ''} updatedAt={stage?.updated_at ?? null}
                steps={stage?.steps} busy={busy} message={isActive ? activeJob?.message : undefined}
                onRunOnly={() => void run([key])}
                onRunFrom={() => void run(STAGE_ORDER.slice(i))}
                last={i === STAGE_ORDER.length - 1}
              />
            </Popover>
          </div>
        )
      })}
      <div style={{ flex: 1 }} />
      {changed > 0 && (
        <span
          style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--warn-text)', whiteSpace: 'nowrap' }}
          title="Lines edited since the last run — Run picks up only these"
        >
          <StatusDot color="var(--warn)" />
          {changed} {changed === 1 ? 'line' : 'lines'} changed
        </span>
      )}
    </nav>
  )
}

const STEP_COLORS: Record<StepState['status'], string> = {
  done: 'var(--ok)', skipped: 'var(--text-faint)', error: 'var(--err)',
}

function StagePopover({ stageKey, status, detail, updatedAt, steps, busy, message, onRunOnly, onRunFrom, last }: {
  stageKey: StageKey; status: StageStatus; detail: string; updatedAt: string | null
  steps: Record<string, StepState> | undefined; busy: boolean; message?: string
  onRunOnly: () => void; onRunFrom: () => void; last: boolean
}) {
  const dot = stageDot(status)
  const entries = Object.entries(steps ?? {}).sort(([a], [b]) => a.localeCompare(b))
  const stamp = when(updatedAt)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 4px 0', fontSize: 12 }}>
        <StatusDot color={dot.fill} hollow={dot.hollow} pulse={dot.pulse} />
        <span style={{ fontWeight: 600 }}>{STAGE_LABELS[stageKey]}</span>
        <span style={{ color: dot.color }}>{STATUS_WORDS[status]}</span>
        <div style={{ flex: 1 }} />
        {stamp && <span className="timecode" style={{ fontSize: 10 }}>{stamp}</span>}
      </div>
      {(message || detail) && (
        <div style={{ padding: '0 4px', fontSize: 11, lineHeight: 1.45, color: status === 'error' ? 'var(--err-text)' : 'var(--text-dim)', overflowWrap: 'anywhere' }}>
          {message || detail}
        </div>
      )}
      {entries.length > 0 && (
        <ul aria-label="Capability steps" style={{ listStyle: 'none', margin: 0, padding: '2px 4px', display: 'flex', flexDirection: 'column', gap: 3 }}>
          {entries.map(([capId, step]) => (
            <li key={capId} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }} title={step.detail || undefined}>
              <StatusDot color={STEP_COLORS[step.status]} hollow={step.status === 'skipped'} />
              <span className="timecode" style={{ fontSize: 10, width: 22 }}>{capId}</span>
              <span style={{ color: step.status === 'error' ? 'var(--err-text)' : 'var(--text-dim)', width: 52 }}>{step.status}</span>
              <span style={{ color: 'var(--text-faint)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0, flex: 1 }}>
                {step.provider_id}
              </span>
            </li>
          ))}
        </ul>
      )}
      <div style={{ height: 1, background: 'var(--border-subtle)', margin: '2px 0' }} />
      <MenuItem icon="play" label="Run only this stage" onClick={onRunOnly} disabled={busy} />
      {!last && <MenuItem icon="forward" label="Run from here" onClick={onRunFrom} disabled={busy} />}
    </div>
  )
}

/** Title-bar run control: Run button, or a progress pill + cancel while a job is active. */
export function RunControls() {
  const project = useProject()
  const activeJob = useActiveJob()
  const toast = useStore((s) => s.toast)
  const applyJob = useStore((s) => s.applyJob)
  const setRunViewOpen = useStore((s) => s.setRunViewOpen)
  if (!project) return null
  const pid = project.id

  async function runAll() {
    try {
      applyJob(await api.runPipeline(pid))
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to run pipeline')
    }
  }

  async function cancel(jobId: string) {
    try {
      await api.cancelJob(jobId)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to cancel job')
    }
  }

  if (activeJob) {
    const label = activeJob.stage ? STAGE_LABELS[activeJob.stage] : 'Queued'
    const pct = Math.round(activeJob.progress * 100)
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 2, flexShrink: 0 }}>
        <button
          type="button"
          onClick={() => setRunViewOpen(true)}
          title={activeJob.message || 'Show live progress'}
          aria-label={`${label} ${pct}% — show live progress`}
          style={{
            position: 'relative', overflow: 'hidden', display: 'flex', alignItems: 'center', gap: 6, height: 24,
            padding: '0 10px', cursor: 'pointer', borderRadius: 'var(--r-md)', border: '1px solid var(--border-strong)',
            background: 'var(--bg-overlay)', color: 'var(--text)', fontSize: 11, fontWeight: 500, minWidth: 140,
          }}
        >
          <span aria-hidden="true" style={{
            position: 'absolute', left: 0, top: 0, bottom: 0, width: `${pct}%`, background: 'var(--accent-dim)',
            transition: 'width 300ms ease-out',
          }} />
          <StatusDot color="var(--accent)" pulse />
          <span style={{ position: 'relative' }}>{label}</span>
          <div style={{ flex: 1 }} />
          <span className="timecode" style={{ position: 'relative', fontSize: 11, color: 'var(--text)' }}>{pct}%</span>
        </button>
        <IconButton name="close" title="Cancel run" onClick={() => void cancel(activeJob.id)} />
      </div>
    )
  }

  const changed = project.segments.filter((s) => s.translate_dirty || s.synth_dirty).length
  return (
    <Button variant="primary" size="sm" onClick={() => void runAll()} disabled={!project.media} title="Run the pipeline">
      <Icon name="play" size={11} />
      {changed > 0 ? `Run ${changed} changed ${changed === 1 ? 'line' : 'lines'}` : 'Run pipeline'}
    </Button>
  )
}
