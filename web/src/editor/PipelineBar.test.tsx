import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { demoProject } from '../test/fixtures'
import type { Job } from '../types'

const runPipeline = vi.fn()
const cancelJob = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    runPipeline: (...args: unknown[]) => runPipeline(...args),
    cancelJob: (...args: unknown[]) => cancelJob(...args),
  },
}))

import { RunControls, StatusStrip } from './PipelineBar'
import { useStore } from '../state/store'

const job = (over: Partial<Job> = {}): Job => ({
  id: 'job_1', project_id: demoProject.id, kind: 'pipeline', stages: ['translate', 'synthesize', 'mix'],
  segment_id: null, status: 'queued', progress: 0, stage: null, message: '', error: null,
  created_at: '2026-09-24T14:00:00Z', started_at: null, finished_at: null, ...over,
})

describe('StatusStrip — stages as steps; nothing runs on a bare click', () => {
  beforeEach(() => {
    useStore.setState({ project: demoProject, jobs: {}, toasts: [] })
    runPipeline.mockReset()
    runPipeline.mockResolvedValue(job())
  })

  it('renders the ten stages with their statuses', () => {
    render(<StatusStrip />)
    const nav = screen.getByRole('navigation', { name: 'Pipeline stages' })
    const steps = within(nav).getAllByRole('button')
    expect(steps).toHaveLength(10)
    expect(within(nav).getByTitle('Ingest: Done')).toBeInTheDocument()
    expect(within(nav).getByTitle('Analyze: skipped')).toBeInTheDocument()
    expect(within(nav).getByTitle('Review: Not run yet')).toBeInTheDocument()
  })

  it('a click opens a popover with the detail and the capability steps; nothing runs', async () => {
    render(<StatusStrip />)
    await userEvent.click(screen.getByTitle('Transcribe: Done'))
    const pop = screen.getByRole('dialog')
    expect(pop).toHaveTextContent('5 segments, 1 speaker(s)')
    expect(within(pop).getByRole('list', { name: 'Capability steps' })).toHaveTextContent(/A7.*done.*speaker_embed\.mfcc/)
    expect(runPipeline).not.toHaveBeenCalled()
  })

  it('"Run only this stage" posts a run for just that stage', async () => {
    render(<StatusStrip />)
    await userEvent.click(screen.getByTitle('Translate: Done'))
    await userEvent.click(screen.getByRole('button', { name: 'Run only this stage' }))
    await waitFor(() => expect(runPipeline).toHaveBeenCalledWith(demoProject.id, ['translate']))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(useStore.getState().jobs.job_1).toBeTruthy()
  })

  it('"Run from here" runs this stage through Render', async () => {
    render(<StatusStrip />)
    await userEvent.click(screen.getByTitle('Mix: Done'))
    await userEvent.click(screen.getByRole('button', { name: 'Run from here' }))
    await waitFor(() => expect(runPipeline).toHaveBeenCalledWith(demoProject.id, ['mix', 'lipsync', 'review', 'render']))
  })

  it('shows the changed-lines count when lines are dirty', () => {
    const seg = demoProject.segments[0]
    useStore.setState({
      project: { ...demoProject, segments: [{ ...seg, translate_dirty: true }, { ...seg, id: 'seg_2', synth_dirty: true }] },
    })
    render(<StatusStrip />)
    expect(screen.getByText('2 lines changed')).toBeInTheDocument()
  })
})

describe('RunControls', () => {
  beforeEach(() => {
    useStore.setState({ project: demoProject, jobs: {}, toasts: [], runViewOpen: false })
    runPipeline.mockReset()
    cancelJob.mockReset()
  })

  it('offers to run only the changed lines when some are dirty', () => {
    const seg = demoProject.segments[0]
    useStore.setState({ project: { ...demoProject, segments: [{ ...seg, synth_dirty: true }] } })
    render(<RunControls />)
    expect(screen.getByRole('button', { name: /Run 1 changed line$/ })).toBeInTheDocument()
  })

  it('runs the whole pipeline otherwise', async () => {
    runPipeline.mockResolvedValue(job())
    render(<RunControls />)
    await userEvent.click(screen.getByRole('button', { name: 'Run pipeline' }))
    await waitFor(() => expect(runPipeline).toHaveBeenCalledWith(demoProject.id))
  })

  it('becomes a progress pill with cancel while a job is active', async () => {
    useStore.setState({ jobs: { job_1: job({ status: 'running', stage: 'synthesize', progress: 0.42 }) } })
    render(<RunControls />)
    expect(screen.getByRole('button', { name: /Synthesize 42%/ })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Cancel run' }))
    expect(cancelJob).toHaveBeenCalledWith('job_1')
    await userEvent.click(screen.getByRole('button', { name: /Synthesize 42%/ }))
    expect(useStore.getState().runViewOpen).toBe(true)
  })
})
