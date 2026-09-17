// Header pipeline controls: the stage stepper (click a stage to run just that one) and the
// Run / running-status control.
import { api } from '../api/client'
import { Button } from '../components/primitives'
import { Icon } from '../components/Icon'
import { useActiveJob, useProject, useStore } from '../state/store'
import { STAGE_LABELS, STAGE_ORDER } from '../types'
import type { StageKey, StageStatus } from '../types'

export function stageDot(status: StageStatus): { fill: string; ring: string; color: string; pulse?: boolean } {
  switch (status) {
    case 'done': return { fill: 'var(--ok)', ring: 'var(--ok)', color: 'var(--text)' }
    case 'dirty': return { fill: 'var(--warn)', ring: 'var(--warn)', color: 'var(--warn-text)' }
    case 'error': return { fill: 'var(--err)', ring: 'var(--err)', color: 'var(--err-text)' }
    case 'queued': return { fill: 'transparent', ring: 'var(--accent)', color: 'var(--accent-text)' }
    case 'running': return { fill: 'var(--accent)', ring: 'var(--accent)', color: 'var(--accent-text)', pulse: true }
    default: return { fill: 'transparent', ring: 'var(--text-faint)', color: 'var(--text-dim)' }
  }
}

export function PipelineStepper() {
  const project = useProject()
  const toast = useStore((s) => s.toast)
  if (!project) return null
  const pid = project.id

  async function runStage(key: StageKey) {
    try {
      await api.runPipeline(pid, [key])
    } catch (e) {
      toast('error', e instanceof Error ? e.message : `Failed to run ${STAGE_LABELS[key]}`)
    }
  }

  return (
    <nav
      aria-label="Pipeline stages"
      style={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'center', overflowX: 'auto', scrollbarWidth: 'none' }}
    >
      <div style={{ flex: 1 }} />
      {STAGE_ORDER.map((key, i) => {
        const stage = project.stages?.[key]
        const status: StageStatus = stage?.status ?? 'pending'
        const dot = stageDot(status)
        return (
          <div key={key} style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
            {i > 0 && <div style={{ width: 6, height: 1, background: 'var(--border-strong)' }} />}
            <button
              type="button"
              onClick={() => void runStage(key)}
              title={`${STAGE_LABELS[key]}: ${status}${stage?.detail ? ` — ${stage.detail}` : ''}\nClick to run this stage`}
              style={{
                display: 'flex', alignItems: 'center', gap: 5, height: 32, padding: '0 4px',
                border: 'none', borderRadius: 'var(--r-md)', background: 'transparent', cursor: 'pointer',
                fontSize: 12, color: dot.color,
                textDecoration: status === 'skipped' ? 'line-through' : 'none',
              }}
            >
              <span style={{
                width: 8, height: 8, borderRadius: '50%', background: dot.fill,
                border: `1.5px solid ${dot.ring}`,
                animation: dot.pulse ? 'od-pulse 1000ms ease-in-out infinite' : undefined,
              }} />
              {STAGE_LABELS[key]}
            </button>
          </div>
        )
      })}
      <div style={{ flex: 1 }} />
    </nav>
  )
}

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
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        <button
          type="button"
          onClick={() => setRunViewOpen(true)}
          title={activeJob.message || 'Show live progress'}
          style={{
            display: 'flex', alignItems: 'center', gap: 8, height: 36, padding: '0 12px', cursor: 'pointer',
            borderRadius: 'var(--r-md)', border: '1px solid var(--accent)', background: 'var(--accent-dim)',
            color: 'var(--accent-text)', fontSize: 13, fontWeight: 500,
          }}
        >
          <span style={{
            width: 8, height: 8, borderRadius: '50%', background: 'var(--accent)',
            animation: 'od-pulse 1000ms ease-in-out infinite',
          }} />
          {label}
          <span className="timecode" style={{ color: 'inherit' }}>{Math.round(activeJob.progress * 100)}%</span>
        </button>
        <Button variant="danger" onClick={() => void cancel(activeJob.id)}>Cancel</Button>
      </div>
    )
  }

  const changed = project.segments.filter((s) => s.translate_dirty || s.synth_dirty).length
  return (
    <Button variant="primary" onClick={() => void runAll()} disabled={!project.media}>
      <Icon name="play" size={12} />
      {changed > 0 ? `Run ${changed} changed ${changed === 1 ? 'line' : 'lines'}` : 'Run pipeline'}
    </Button>
  )
}
