import { beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { demoProject } from '../test/fixtures'
import type { Project, Segment } from '../types'

const updateSegment = vi.fn(async (_pid: string, sid: string, patch: Partial<Segment>): Promise<Segment> => {
  const seg = project.segments.find((s) => s.id === sid)!
  return { ...seg, ...patch }
})
const regenerateSegment = vi.fn(async () => ({
  id: 'job_1', project_id: demoProject.id, kind: 'segment', stages: ['synthesize'], segment_id: 'seg_a',
  status: 'queued', progress: 0, stage: null, message: '', error: null,
  created_at: '2026-07-08T09:00:00Z', started_at: null, finished_at: null,
}))
const splitSegment = vi.fn(async (): Promise<Project> => project)

vi.mock('../api/client', () => ({
  api: {
    updateSegment: (pid: string, sid: string, patch: Partial<Segment>) => updateSegment(pid, sid, patch),
    regenerateSegment: () => regenerateSegment(),
    splitSegment: () => splitSegment(),
    transcribeSegment: vi.fn(),
    mediaUrl: (pid: string, rel: string) => `/api/media/${pid}/${rel}`,
  },
  subscribeProjectEvents: () => () => {},
}))

// Import AFTER the mock is registered.
import { ScriptPanel, TranscriptList } from './TranscriptList'
import { useStore } from '../state/store'

const base = demoProject.segments[0]
const seg = (over: Partial<Segment>): Segment => ({ ...base, ...over })

const project: Project = {
  ...demoProject,
  segments: [
    seg({
      id: 'seg_a', start: 1, end: 3, source_text: 'こんにちは 世界', translated_text: 'Hello world',
      words: [
        { start: 1.0, end: 1.8, text: 'こんにちは', confidence: 0.95 },
        { start: 1.8, end: 2.6, text: '世界', confidence: 0.42 },
      ],
      takes: [{ id: 'take_1', path: 'takes/1.wav', duration: 1.9, provider_id: 'tts.mock', rate_factor: 1, created_at: '2026-07-08T09:00:00Z', lang: 'en' }],
      active_take_id: 'take_1',
    }),
    seg({ id: 'seg_b', start: 4, end: 6, source_text: 'second', translated_text: 'Second line', translate_dirty: true }),
    seg({ id: 'seg_c', start: 7, end: 9, source_text: 'third', translated_text: 'Third line', skipped: true }),
    seg({
      id: 'seg_d', start: 10, end: 12, source_text: 'fourth', translated_text: 'Fourth line',
      takes: [{ id: 'take_4', path: 'takes/4.wav', duration: 2.8, provider_id: 'tts.mock', rate_factor: 0.71, created_at: '2026-07-08T09:00:00Z', lang: '' }],
      active_take_id: 'take_4',
    }),
  ],
}

function mount() {
  useStore.setState({
    project, selection: null, editing: null, playhead: 0, playing: false, jobs: {}, toasts: [], seekRequest: null,
  })
  return render(<ScriptPanel />)
}

const rows = () => screen.getAllByRole('option')

describe('ScriptPanel — rows and filters', () => {
  beforeEach(() => { updateSegment.mockClear(); splitSegment.mockClear() })

  it('keeps the TranscriptList alias', () => {
    expect(TranscriptList).toBe(ScriptPanel)
  })

  it('renders one row per line in time order with its translation and the counts', () => {
    mount()
    expect(rows()).toHaveLength(4)
    expect(within(rows()[0]).getByText('Hello world')).toBeInTheDocument()
    expect(screen.getByText('4 lines · 1 stale · 1 excluded')).toBeInTheDocument()
  })

  it('filters by stale / excluded / search', async () => {
    mount()
    await userEvent.click(screen.getByRole('button', { name: 'Stale' }))
    expect(rows()).toHaveLength(1)
    expect(within(rows()[0]).getByText('Second line')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Excluded' }))
    expect(rows()).toHaveLength(1)
    expect(within(rows()[0]).getByText('KEPT ORIGINAL')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'All' }))
    await userEvent.type(screen.getByRole('textbox', { name: 'Search lines' }), 'four')
    expect(rows()).toHaveLength(1)
    expect(within(rows()[0]).getByText('Fourth line')).toBeInTheDocument()
    expect(screen.getByText('1 / 4 lines · 1 stale · 1 excluded')).toBeInTheDocument()
  })

  it('shows the hatched badge on excluded lines and an overrun badge when the take is too long', () => {
    mount()
    expect(within(rows()[2]).getByText('KEPT ORIGINAL')).toBeInTheDocument()
    // take_4 runs 2.8s in a 2.0s slot: +0.8s, >30% over
    expect(within(rows()[3]).getByText('+0.8s')).toBeInTheDocument()
    expect(within(rows()[0]).queryByText(/\+\d/)).toBeNull()
  })

  it('selects (and seeks to) a line on click', async () => {
    mount()
    await userEvent.click(within(rows()[1]).getByText('Second line'))
    expect(useStore.getState().selection).toBe('seg_b')
    expect(useStore.getState().seekRequest?.t).toBe(4)
  })
})

describe('ScriptPanel — words', () => {
  it('renders word spans, flags low confidence, and seeks on click', async () => {
    mount()
    const low = screen.getByTitle(/confidence 0\.42/)
    expect(low).toHaveTextContent('世界')
    await userEvent.click(low)
    expect(useStore.getState().seekRequest?.t).toBe(1.8)
    expect(useStore.getState().selection).toBe('seg_a')
  })

  it('hides the confidence indicator from the view-options toggle', async () => {
    mount()
    await userEvent.click(screen.getByRole('button', { name: 'Hide low-confidence words' }))
    expect(screen.queryByTitle(/confidence 0\.42/)).toBeNull()
    expect(screen.getByText('世界')).toBeInTheDocument()
  })
})

describe('ScriptPanel — inline editing', () => {
  beforeEach(() => updateSegment.mockClear())

  it('Enter on the selected row edits the translation; Enter commits and moves to the next row', async () => {
    mount()
    act(() => useStore.getState().selectSegment('seg_a'))
    await userEvent.keyboard('{Enter}')
    expect(useStore.getState().editing).toEqual({ segmentId: 'seg_a', field: 'translation' })
    const ta = screen.getByRole('textbox', { name: 'Edit translation' })
    expect(ta).toHaveValue('Hello world')

    await userEvent.type(ta, '!')
    await userEvent.keyboard('{Enter}')
    expect(updateSegment).toHaveBeenCalledWith(project.id, 'seg_a', { translated_text: 'Hello world!' })
    expect(useStore.getState().editing).toEqual({ segmentId: 'seg_b', field: 'translation' })
    expect(useStore.getState().selection).toBe('seg_b')
    expect(screen.getByRole('textbox', { name: 'Edit translation' })).toHaveValue('Second line')
  })

  it('does not save an unchanged line, and Esc reverts', async () => {
    mount()
    act(() => useStore.getState().setEditing({ segmentId: 'seg_b', field: 'translation' }))
    const ta = screen.getByRole('textbox', { name: 'Edit translation' })
    await userEvent.type(ta, 'zzz')
    await userEvent.keyboard('{Escape}')
    expect(updateSegment).not.toHaveBeenCalled()
    expect(useStore.getState().editing).toBeNull()
    expect(screen.getByText('Second line')).toBeInTheDocument()
  })

  it('Tab moves from the source to the translation of the same row, then on to the next row', async () => {
    mount()
    act(() => useStore.getState().setEditing({ segmentId: 'seg_b', field: 'source' }))
    const src = screen.getByRole('textbox', { name: 'Edit source' })
    await userEvent.type(src, ' edited')
    await userEvent.keyboard('{Tab}')
    expect(updateSegment).toHaveBeenCalledWith(project.id, 'seg_b', { source_text: 'second edited' })
    expect(useStore.getState().editing).toEqual({ segmentId: 'seg_b', field: 'translation' })
    await userEvent.keyboard('{Tab}')
    expect(useStore.getState().editing).toEqual({ segmentId: 'seg_c', field: 'source' })
  })

  it('double-clicking a line starts editing it', async () => {
    mount()
    await userEvent.dblClick(screen.getByText('Fourth line'))
    expect(useStore.getState().editing).toEqual({ segmentId: 'seg_d', field: 'translation' })
  })

  it('reports a failed save as a toast and leaves the editor', async () => {
    updateSegment.mockRejectedValueOnce(new Error('segment not found'))
    mount()
    act(() => useStore.getState().setEditing({ segmentId: 'seg_a', field: 'translation' }))
    await userEvent.type(screen.getByRole('textbox', { name: 'Edit translation' }), '?')
    await userEvent.keyboard('{Enter}')
    await act(async () => { await Promise.resolve() })
    expect(useStore.getState().toasts.map((t) => t.text)).toContain('segment not found')
  })
})

describe('ScriptPanel — row toolbar', () => {
  beforeEach(() => { regenerateSegment.mockClear(); splitSegment.mockClear() })

  it('disables Split unless the playhead is inside the line', async () => {
    mount()
    const split = within(rows()[0]).getByRole('button', { name: /Split at playhead/ })
    expect(split).toBeDisabled()
    act(() => useStore.setState({ playhead: 2 }))
    expect(within(rows()[0]).getByRole('button', { name: /Split at playhead/ })).toBeEnabled()
    fireEvent.click(within(rows()[0]).getByRole('button', { name: /Split at playhead/ }))
    expect(splitSegment).toHaveBeenCalled()
  })

  it('Play from here seeks and starts playback', () => {
    mount()
    fireEvent.click(within(rows()[1]).getByRole('button', { name: 'Play from here' }))
    expect(useStore.getState().seekRequest?.t).toBe(4)
    expect(useStore.getState().playing).toBe(true)
  })

  it('Re-voice starts a segment job; excluded lines cannot be regenerated', async () => {
    mount()
    fireEvent.click(within(rows()[0]).getByRole('button', { name: 'Re-voice' }))
    await act(async () => { await Promise.resolve() })
    expect(regenerateSegment).toHaveBeenCalled()
    expect(useStore.getState().jobs.job_1?.segment_id).toBe('seg_a')
    expect(within(rows()[2]).getByRole('button', { name: /Re-voice — unavailable/ })).toBeDisabled()
  })

  it('shows the empty state when there are no lines', () => {
    useStore.setState({ project: { ...project, segments: [] }, selection: null, editing: null })
    render(<ScriptPanel />)
    expect(screen.getByText(/once Transcribe has run/)).toBeInTheDocument()
    expect(screen.queryAllByRole('option')).toHaveLength(0)
  })
})
