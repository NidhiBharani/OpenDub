import { beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { demoProject } from '../test/fixtures'

vi.mock('../api/client', () => ({
  api: {
    mediaUrl: (pid: string, rel: string) => `/api/media/${pid}/${rel}`,
    setSkipRanges: vi.fn(),
  },
}))

import { api } from '../api/client'
import { Timeline } from './Timeline'
import { useStore } from '../state/store'
import { useEditorKeyboard } from './useKeyboard'

const speaker2 = { id: 'spk_2', name: 'Narrator', color: '#4fb286', reference_path: null }
const project = { ...demoProject, speakers: [...demoProject.speakers, speaker2] }

function KeyboardHost() {
  useEditorKeyboard()
  return <Timeline />
}

describe('Timeline — toolbar', () => {
  beforeEach(() => {
    useStore.setState({
      project, range: null, selection: null, skipSelection: null, zoom: 30, scrollX: 0,
      snapping: true, clipHeight: 'M', anchors: null, filmstrip: null, editing: null,
    })
    // jsdom has no layout: give the canvas container a size so Fit has something to fit into
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      width: 1000, height: 300, top: 0, left: 0, right: 1000, bottom: 300, x: 0, y: 0, toJSON: () => ({}),
    })
  })

  it('renders the tools with their shortcuts in the tooltips', () => {
    render(<Timeline />)
    const bar = screen.getByRole('toolbar', { name: 'Timeline tools' })
    expect(bar).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Range' })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('button', { name: /Snapping \(N\)/ })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('group', { name: 'Clip height' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Zoom in (+)' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Zoom out (−)' })).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: 'Zoom' })).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: 'Anchor threshold' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Fit whole timeline \(Shift\+Z\)/ })).toBeInTheDocument()
  })

  it('disables Exclude from dub until a range exists, then excludes it', async () => {
    const saved = { ...project, skip_ranges: [{ id: 'rng_1', start: 2, end: 6, label: '' }] }
    vi.mocked(api.setSkipRanges).mockResolvedValue(saved)
    render(<Timeline />)
    const exclude = screen.getByRole('button', { name: /Exclude from dub/ })
    expect(exclude).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Clear range' })).toBeDisabled()

    act(() => useStore.getState().setRange({ start: 2, end: 6 }))
    expect(exclude).toBeEnabled()
    await act(async () => { fireEvent.click(exclude) })
    expect(api.setSkipRanges).toHaveBeenCalledWith(project.id, [
      expect.objectContaining({ start: 2, end: 6 }),
    ])
    expect(useStore.getState().range).toBeNull()
    expect(useStore.getState().skipSelection).toBe('rng_1')
  })

  it('Clear range drops the range', () => {
    render(<Timeline />)
    act(() => useStore.getState().setRange({ start: 2, end: 6 }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear range' }))
    expect(useStore.getState().range).toBeNull()
  })

  it('Fit zooms so the whole media fills the canvas (button and Shift+Z)', () => {
    render(<KeyboardHost />)
    act(() => useStore.getState().setScrollX(5))
    fireEvent.click(screen.getByRole('button', { name: /Fit whole timeline/ }))
    expect(useStore.getState().zoom).toBeCloseTo(1000 / 29.4, 3)
    expect(useStore.getState().scrollX).toBe(0)

    act(() => useStore.getState().setZoom(80))
    fireEvent.keyDown(window, { key: 'Z', shiftKey: true })
    expect(useStore.getState().zoom).toBeCloseTo(1000 / 29.4, 3)
  })

  it('zoom buttons and slider drive store.zoom', () => {
    render(<Timeline />)
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in (+)' }))
    expect(useStore.getState().zoom).toBeCloseTo(39, 0)
    fireEvent.change(screen.getByRole('slider', { name: 'Zoom' }), { target: { value: '1' } })
    expect(useStore.getState().zoom).toBeCloseTo(500, 3)
    fireEvent.change(screen.getByRole('slider', { name: 'Zoom' }), { target: { value: '0' } })
    expect(useStore.getState().zoom).toBeCloseTo(2, 3)
  })

  it('snapping toggle and clip height write to the store', () => {
    render(<Timeline />)
    fireEvent.click(screen.getByRole('button', { name: /Snapping \(N\)/ }))
    expect(useStore.getState().snapping).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'L' }))
    expect(useStore.getState().clipHeight).toBe('L')
  })

  it('anchor threshold slider writes to the store', () => {
    render(<Timeline />)
    fireEvent.change(screen.getByRole('slider', { name: 'Anchor threshold' }), { target: { value: '0.7' } })
    expect(useStore.getState().anchorThreshold).toBeCloseTo(0.7)
  })
})

describe('Timeline — track headers', () => {
  beforeEach(() => {
    useStore.setState({ project, audioTrack: 'original', clipHeight: 'M' })
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      width: 1000, height: 300, top: 0, left: 0, right: 1000, bottom: 300, x: 0, y: 0, toJSON: () => ({}),
    })
  })

  it('lists the fixed tracks and one row per speaker', () => {
    render(<Timeline />)
    expect(screen.getByText('V')).toBeInTheDocument()
    expect(screen.getByText('Original')).toBeInTheDocument()
    expect(screen.getByText('Dub')).toBeInTheDocument()
    expect(screen.getByText('Speaker 1')).toBeInTheDocument()
    expect(screen.getByText('Narrator')).toBeInTheDocument()
  })

  it('monitor buttons switch the audio track; the dub one needs the mix', () => {
    render(<Timeline />)
    const dub = screen.getByRole('button', { name: 'Monitor dub mix (2)' })
    const orig = screen.getByRole('button', { name: 'Monitor original audio (1)' })
    expect(orig).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(dub)
    expect(useStore.getState().audioTrack).toBe('dub')
    fireEvent.click(orig)
    expect(useStore.getState().audioTrack).toBe('original')
  })

  it('disables the dub monitor while the mix has not run', () => {
    useStore.setState({
      project: { ...project, stages: { ...project.stages, mix: { status: 'pending', detail: '', updated_at: null } } },
    })
    render(<Timeline />)
    expect(screen.getByRole('button', { name: /Monitor dub mix \(run Mix first\)/ })).toBeDisabled()
  })
})

describe('keyboard map', () => {
  beforeEach(() => {
    useStore.setState({
      project, range: null, selection: null, skipSelection: null, playhead: 2, playing: false, rate: 1,
      snapping: true, editing: null, anchors: { version: 1, anchors: [
        { t: 4, kind: 'scene', confidence: 0.9, start: 4, end: 4 },
        { t: 10, kind: 'scene', confidence: 0.9, start: 10, end: 10 },
      ] }, anchorThreshold: 0.35,
    })
  })

  it('I/O set the range at the playhead; X selects the current line; Alt+X clears', () => {
    render(<KeyboardHost />)
    fireEvent.keyDown(window, { key: 'i' })
    expect(useStore.getState().range).toEqual({ start: 2, end: 3 })
    act(() => useStore.getState().seek(6))
    fireEvent.keyDown(window, { key: 'o' })
    expect(useStore.getState().range).toEqual({ start: 2, end: 6 })
    fireEvent.keyDown(window, { key: 'x', altKey: true })
    expect(useStore.getState().range).toBeNull()

    act(() => useStore.getState().seek(2)) // inside seg_019e618e (1.0–3.2)
    fireEvent.keyDown(window, { key: 'x' })
    expect(useStore.getState().range).toEqual({ start: 1.0, end: 3.2 })
  })

  it('J/K/L shuttle and Space reset the rate', () => {
    render(<KeyboardHost />)
    fireEvent.keyDown(window, { key: 'l' })
    expect(useStore.getState()).toMatchObject({ playing: true, rate: 1 })
    fireEvent.keyDown(window, { key: 'l' })
    expect(useStore.getState().rate).toBe(1.5)
    fireEvent.keyDown(window, { key: 'l' })
    fireEvent.keyDown(window, { key: 'l' })
    fireEvent.keyDown(window, { key: 'l' })
    expect(useStore.getState().rate).toBe(4)
    fireEvent.keyDown(window, { key: 'k' })
    expect(useStore.getState()).toMatchObject({ playing: false, rate: 1 })
    fireEvent.keyDown(window, { key: 'j' })
    expect(useStore.getState()).toMatchObject({ playing: true, rate: 0.5 })
    fireEvent.keyDown(window, { key: 'j' })
    expect(useStore.getState().rate).toBe(0.25)
    fireEvent.keyDown(window, { key: ' ' })
    expect(useStore.getState()).toMatchObject({ playing: false, rate: 1 })
  })

  it('Alt+[ / Alt+] jump between visible anchors; Home/End seek the ends', () => {
    render(<KeyboardHost />)
    fireEvent.keyDown(window, { key: ']', code: 'BracketRight', altKey: true })
    expect(useStore.getState().playhead).toBe(4)
    fireEvent.keyDown(window, { key: ']', code: 'BracketRight', altKey: true })
    expect(useStore.getState().playhead).toBe(10)
    fireEvent.keyDown(window, { key: '[', code: 'BracketLeft', altKey: true })
    expect(useStore.getState().playhead).toBe(4)
    fireEvent.keyDown(window, { key: 'End' })
    expect(useStore.getState().playhead).toBe(29.4)
    fireEvent.keyDown(window, { key: 'Home' })
    expect(useStore.getState().playhead).toBe(0)
  })

  it('N toggles snapping; Esc clears the range before the selection', () => {
    render(<KeyboardHost />)
    fireEvent.keyDown(window, { key: 'n' })
    expect(useStore.getState().snapping).toBe(false)
    act(() => {
      useStore.getState().selectSegment('seg_019e618e')
      useStore.getState().setRange({ start: 1, end: 2 })
    })
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(useStore.getState().range).toBeNull()
    expect(useStore.getState().selection).toBe('seg_019e618e')
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(useStore.getState().selection).toBeNull()
  })

  it('Delete removes the selected skip region only', async () => {
    const withSkip = { ...project, skip_ranges: [{ id: 'rng_1', start: 2, end: 6, label: '' }] }
    useStore.setState({ project: withSkip, skipSelection: 'rng_1' })
    vi.mocked(api.setSkipRanges).mockResolvedValue({ ...withSkip, skip_ranges: [] })
    render(<KeyboardHost />)
    await act(async () => { fireEvent.keyDown(window, { key: 'Delete' }) })
    expect(api.setSkipRanges).toHaveBeenCalledWith(project.id, [])
    expect(useStore.getState().skipSelection).toBeNull()
  })

  it('ignores shortcuts while a script row is being edited', () => {
    useStore.setState({ editing: { segmentId: 'seg_019e618e', field: 'translation' } })
    render(<KeyboardHost />)
    fireEvent.keyDown(window, { key: 'i' })
    expect(useStore.getState().range).toBeNull()
  })
})
