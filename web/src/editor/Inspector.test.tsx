import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { demoProject } from '../test/fixtures'
import type { Project, Segment, Speaker, TimeRange } from '../types'

const updateSegment = vi.fn(async (_pid: string, sid: string, patch: Partial<Segment>): Promise<Segment> => {
  const seg = project.segments.find((s) => s.id === sid)!
  return { ...seg, ...patch }
})
const updateSpeaker = vi.fn(async (_pid: string, spid: string, patch: Partial<Speaker>): Promise<Speaker> => ({
  ...project.speakers.find((s) => s.id === spid)!, ...patch,
}))
const setSkipRanges = vi.fn(async (_pid: string, ranges: TimeRange[]): Promise<Project> => ({ ...project, skip_ranges: ranges }))

vi.mock('../api/client', () => ({
  api: {
    updateSegment: (pid: string, sid: string, patch: Partial<Segment>) => updateSegment(pid, sid, patch),
    updateSpeaker: (pid: string, spid: string, patch: Partial<Speaker>) => updateSpeaker(pid, spid, patch),
    setSkipRanges: (pid: string, ranges: TimeRange[]) => setSkipRanges(pid, ranges),
    regenerateSegment: vi.fn(),
    mediaUrl: (pid: string, rel: string) => `/api/media/${pid}/${rel}`,
  },
  subscribeProjectEvents: () => () => {},
}))

// Import AFTER the mock is registered.
import { Inspector } from './Inspector'
import { useStore } from '../state/store'

const base = demoProject.segments[0]
const project: Project = {
  ...demoProject,
  speakers: [
    ...demoProject.speakers,
    { id: 'spk_2', name: 'Speaker 2', color: '#4fb286', reference_path: null },
  ],
  segments: [
    {
      ...base, id: 'seg_a', start: 1, end: 3, source_text: 'こんにちは', translated_text: 'Hello',
      words: [{ start: 1, end: 2, text: 'こんにちは', confidence: 0.8 }],
      takes: [
        { id: 'take_1', path: 'takes/1.wav', duration: 1.9, provider_id: 'tts.mock', rate_factor: 1, created_at: '2026-07-08T09:00:00Z', lang: 'en' },
        { id: 'take_2', path: 'takes/2.wav', duration: 2.4, provider_id: 'tts.f5_tts', rate_factor: 0.83, created_at: '2026-07-08T09:01:00Z', lang: 'en' },
      ],
      active_take_id: 'take_2',
    },
    { ...base, id: 'seg_b', start: 4, end: 6, source_text: 'second', translated_text: 'Second', speaker_id: 'spk_2' },
  ],
  skip_ranges: [{ id: 'rng_1', start: 3.5, end: 6.5, label: 'song' }],
}

function mount(state: Partial<ReturnType<typeof useStore.getState>> = {}) {
  useStore.setState({ project, selection: null, skipSelection: null, jobs: {}, toasts: [], ...state })
  return render(<Inspector />)
}

describe('Inspector', () => {
  beforeEach(() => { updateSegment.mockClear(); updateSpeaker.mockClear(); setSkipRanges.mockClear() })

  it('shows a hint when nothing is selected', () => {
    mount()
    expect(screen.getByText(/Select a line/)).toBeInTheDocument()
  })

  it('shows the line number, range and tabs for the selected line', () => {
    mount({ selection: 'seg_b' })
    expect(screen.getByText('Line 02')).toBeInTheDocument()
    expect(screen.getByText('0:04.0 – 0:06.0 · 2.0s')).toBeInTheDocument()
    expect(screen.getByRole('group', { name: 'Inspector tabs' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Translation' })).toHaveValue('Second')
  })

  it('switches between the Line, Speaker and Info tabs', async () => {
    mount({ selection: 'seg_a' })
    await userEvent.click(screen.getByRole('button', { name: 'Speaker' }))
    expect(screen.getByRole('textbox', { name: 'Speaker name' })).toHaveValue('Speaker 1')
    expect(screen.getByText('1 of 2')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Play reference clip' })).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Info' }))
    expect(screen.getByText('seg_a')).toBeInTheDocument()
    expect(screen.getByText('0.80')).toBeInTheDocument()
    expect(screen.getByText('take_2')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Line' }))
    expect(screen.getByRole('combobox', { name: 'Speaker' })).toBeInTheDocument()
  })

  it('changing the speaker patches the segment', async () => {
    mount({ selection: 'seg_a' })
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Speaker' }), 'spk_2')
    expect(updateSegment).toHaveBeenCalledWith(project.id, 'seg_a', { speaker_id: 'spk_2' })
  })

  it('lists takes newest first and selecting one sets active_take_id', async () => {
    mount({ selection: 'seg_a' })
    const radios = screen.getAllByRole('radio')
    expect(radios.map((r) => r.getAttribute('aria-label'))).toEqual(['Use take 2', 'Use take 1'])
    expect(radios[0]).toBeChecked()
    expect(screen.getByText('×0.83')).toBeInTheDocument()
    await userEvent.click(radios[1])
    expect(updateSegment).toHaveBeenCalledWith(project.id, 'seg_a', { active_take_id: 'take_1' })
  })

  it('commits an edited translation on blur, and only when it changed', async () => {
    mount({ selection: 'seg_a' })
    const ta = screen.getByRole('textbox', { name: 'Translation' })
    await userEvent.click(ta)
    await userEvent.tab()
    expect(updateSegment).not.toHaveBeenCalled()
    await userEvent.type(ta, ' there')
    await userEvent.tab()
    expect(updateSegment).toHaveBeenCalledWith(project.id, 'seg_a', { translated_text: 'Hello there' })
  })

  it('rejects an invalid timing edit with a toast and reverts the field', async () => {
    mount({ selection: 'seg_a' })
    const end = screen.getByRole('textbox', { name: 'End (seconds)' })
    await userEvent.clear(end)
    await userEvent.type(end, '0.5')
    await userEvent.tab()
    expect(updateSegment).not.toHaveBeenCalled()
    expect(useStore.getState().toasts[0]?.text).toMatch(/Timing must/)
    expect(end).toHaveValue('3.00')
  })

  it('renames the speaker from the Speaker tab', async () => {
    mount({ selection: 'seg_a' })
    await userEvent.click(screen.getByRole('button', { name: 'Speaker' }))
    const name = screen.getByRole('textbox', { name: 'Speaker name' })
    await userEvent.clear(name)
    await userEvent.type(name, 'Hanako{Enter}')
    expect(updateSpeaker).toHaveBeenCalledWith(project.id, 'spk_d639006e', { name: 'Hanako' })
  })

  it('shows the skip range sheet when a range is selected and nothing else is', async () => {
    mount({ skipSelection: 'rng_1' })
    expect(screen.getByText('Keep original')).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Range label' })).toHaveValue('song')
    // seg_b (4–6) lies inside 3.5–6.5
    expect(screen.getByText(/1 inside/)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(setSkipRanges).toHaveBeenCalledWith(project.id, []))
  })

  it('prefers the segment over a stale skip selection', () => {
    mount({ selection: 'seg_a', skipSelection: 'rng_1' })
    expect(screen.getByText('Line 01')).toBeInTheDocument()
  })
})
