import { beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen } from '@testing-library/react'
import { demoProject } from '../test/fixtures'
import type { Version } from '../types'

vi.mock('../api/client', () => ({
  api: { mediaUrl: (pid: string, rel: string) => `/api/media/${pid}/${rel}` },
}))

import { Player } from './Player'
import { useStore } from '../state/store'

const video = () => document.querySelector('video') as HTMLVideoElement
const nextFrame = () => act(() => new Promise<void>((resolve) => requestAnimationFrame(() => resolve())))

const version: Version = {
  id: 'ver_a1', label: 'Run · 24 Sep 14:03', kind: 'auto', created_at: '2026-09-24T14:03:00Z',
  source_lang: 'ja', target_lang: 'en', preset: 'minimal', runtime: 'builtin',
  providers: { tts: 'tts.f5_tts' }, stages: {}, segment_count: 27, voiced_count: 27, skipped_ranges: 2,
  has_mix: true, has_render: false,
  outputs: { dub_mix: 'versions/ver_a1/dub_mix.wav', playback_dub: 'versions/ver_a1/playback_dub.mp4' },
  summary: '', bytes: 14 * 1024 * 1024, job_id: null, parent_id: null,
}

describe('Player — one <video>, source follows the audio track', () => {
  beforeEach(() => {
    useStore.setState({
      project: demoProject, audioTrack: 'original', playing: false, seekRequest: null,
      versions: [], previewVersionId: null, loopRange: false, range: null, rate: 1,
    })
  })

  it('plays the original by default and has no separate audio element to keep in sync', () => {
    render(<Player />)
    expect(video().getAttribute('src')).toMatch(/\/playback\.mp4(\?|$)/)
    expect(document.querySelector('audio')).toBeNull()
  })

  it('switches to the muxed dub preview and back, keeping the playhead', async () => {
    render(<Player />)
    // pretend the viewer is 12 s in and playing when they switch tracks
    Object.defineProperty(video(), 'currentTime', { value: 12, writable: true })
    Object.defineProperty(video(), 'duration', { value: 29.4, writable: true })
    act(() => useStore.setState({ playing: true }))
    // the player samples the position once per animation frame
    await nextFrame()
    act(() => useStore.getState().setAudioTrack('dub'))
    expect(video().getAttribute('src')).toMatch(/\/playback_dub\.mp4\?v=/)

    // the reload lands at 0; loadedmetadata restores where we were
    const v = video()
    v.currentTime = 0
    const play = vi.spyOn(v, 'play').mockResolvedValue()
    act(() => { v.dispatchEvent(new Event('loadedmetadata')) })
    expect(v.currentTime).toBe(12)
    expect(play).toHaveBeenCalled()

    act(() => useStore.getState().setAudioTrack('original'))
    expect(video().getAttribute('src')).toMatch(/\/playback\.mp4\?v=/)
  })

  it('falls back to the original file while the mix has not run', () => {
    useStore.setState({
      project: { ...demoProject, stages: { ...demoProject.stages, mix: { status: 'pending', detail: '', updated_at: null } } },
      audioTrack: 'dub',
    })
    render(<Player />)
    expect(video().getAttribute('src')).toMatch(/\/playback\.mp4(\?|$)/)
  })

  it('previews a version by playing its copied dub output, with a banner to restore or close', () => {
    useStore.setState({ versions: [version], previewVersionId: version.id })
    render(<Player />)
    expect(video().getAttribute('src')).toBe(`/api/media/${demoProject.id}/versions/ver_a1/playback_dub.mp4`)
    expect(screen.getByRole('status')).toHaveTextContent(/Previewing Run · 24 Sep 14:03/)

    act(() => { screen.getByRole('button', { name: 'Close' }).click() })
    expect(useStore.getState().previewVersionId).toBeNull()
    expect(video().getAttribute('src')).toMatch(/\/playback\.mp4(\?|$)/)
  })

  it('wraps playback to the range start when looping past the range end', async () => {
    useStore.setState({ loopRange: true, range: { start: 2, end: 5 } })
    render(<Player />)
    Object.defineProperty(video(), 'currentTime', { value: 5.2, writable: true })
    await nextFrame()
    expect(video().currentTime).toBe(2)
    expect(useStore.getState().playhead).toBe(2)

    // the loop toggle off leaves the position alone
    act(() => useStore.getState().setLoopRange(false))
    video().currentTime = 5.2
    await nextFrame()
    expect(video().currentTime).toBe(5.2)
  })

  it('applies store.rate to the element (J/K/L shuttle writes it)', () => {
    render(<Player />)
    act(() => useStore.getState().setRate(2))
    expect(video().playbackRate).toBe(2)
    expect((screen.getByLabelText('Playback rate') as HTMLSelectElement).value).toBe('2')
  })
})
