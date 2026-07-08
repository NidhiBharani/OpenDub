// Stage chips (click to run one stage) + active job status / "Run pipeline" button.
import { api } from '../api/client'
import { Button, ProgressBar, Spinner } from '../components/primitives'
import { useActiveJob, useProject, useStore } from '../state/store'
import { STAGE_LABELS, STAGE_ORDER } from '../types'
import type { StageKey, StageStatus } from '../types'

function chipVisuals(status: StageStatus): {
  border: string
  background: string
  color: string
  textDecoration: string
  animation?: string
} {
  switch (status) {
    case 'done':
      return { border: 'var(--ok)', background: 'rgb(80 200 120 / 0.10)', color: 'var(--ok)', textDecoration: 'none' }
    case 'dirty':
      return { border: 'var(--warn)', background: 'rgb(232 184 76 / 0.10)', color: 'var(--warn)', textDecoration: 'none' }
    case 'error':
      return { border: 'var(--err)', background: 'rgb(232 96 76 / 0.10)', color: 'var(--err)', textDecoration: 'none' }
    case 'queued':
    case 'running':
      return {
        border: 'var(--running)', background: 'rgb(76 155 232 / 0.12)', color: 'var(--running)',
        textDecoration: 'none', animation: 'od-pulse 1200ms ease-in-out infinite',
      }
    case 'skipped':
      return { border: 'var(--border)', background: 'transparent', color: 'var(--text-faint)', textDecoration: 'line-through' }
    default:
      return { border: 'var(--border)', background: 'transparent', color: 'var(--text-dim)', textDecoration: 'none' }
  }
}

function StatusGlyph({ status }: { status: StageStatus }) {
  if (status === 'done') return <span>✓</span>
  if (status === 'dirty') return <span>●</span>
  if (status === 'error') return <span>✕</span>
  if (status === 'queued' || status === 'running') {
    return (
      <span
        style={{
          display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
          background: 'var(--running)', animation: 'od-pulse 900ms ease-in-out infinite',
        }}
      />
    )
  }
  return null
}

export function PipelineBar() {
  const project = useProject()
  const activeJob = useActiveJob()
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

  async function runAll() {
    try {
      await api.runPipeline(pid)
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

  return (
    <div
      style={{
        height: 44, flexShrink: 0, borderTop: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', gap: 6, padding: '0 12px', overflowX: 'auto',
      }}
    >
      {STAGE_ORDER.map((key) => {
        const stage = project.stages?.[key]
        const status: StageStatus = stage?.status ?? 'pending'
        const visuals = chipVisuals(status)
        const activeStageJob = activeJob && activeJob.stage === key ? activeJob : null
        return (
          <button
            key={key}
            type="button"
            onClick={() => void runStage(key)}
            title={status === 'error' ? (stage?.detail || 'Error') : STAGE_LABELS[key]}
            style={{
              flexShrink: 0, display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px',
              borderRadius: 999, border: `1px solid ${visuals.border}`, background: visuals.background,
              color: visuals.color, textDecoration: visuals.textDecoration, animation: visuals.animation,
              cursor: 'pointer', fontSize: 12, fontWeight: 500,
            }}
          >
            <StatusGlyph status={status} />
            <span>{STAGE_LABELS[key]}</span>
            {activeStageJob && <div style={{ width: 36 }}><ProgressBar value={activeStageJob.progress} /></div>}
          </button>
        )
      })}

      <div style={{ flex: 1 }} />

      {activeJob ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          <Spinner size={14} />
          <span
            style={{
              fontSize: 12, color: 'var(--text-dim)', maxWidth: '40ch',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}
          >
            {activeJob.message}
          </span>
          <div style={{ width: 160 }}><ProgressBar value={activeJob.progress} /></div>
          <Button variant="ghost" onClick={() => void cancel(activeJob.id)}>Cancel</Button>
        </div>
      ) : (
        <Button variant="primary" onClick={() => void runAll()} disabled={!project.media}>
          Run pipeline
        </Button>
      )}
    </div>
  )
}
