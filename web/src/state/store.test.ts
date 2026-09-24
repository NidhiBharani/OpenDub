import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useStore } from './store'
import { demoProject, resolvedCapabilities } from '../test/fixtures'

const getProjectCapabilities = vi.fn(async () => resolvedCapabilities)

// The store imports the api client; stub it so no module-load side effects hit fetch.
vi.mock('../api/client', () => ({
  api: { getProjectCapabilities: () => getProjectCapabilities() },
  subscribeProjectEvents: () => () => {},
}))

function reset() {
  useStore.setState({
    project: null, projects: [], selection: null, playhead: 0,
    playing: false, zoom: 30, scrollX: 0, audioTrack: 'original', toasts: [],
  })
}

describe('store: playback + timeline logic', () => {
  beforeEach(reset)

  it('clamps zoom into the 2..500 range', () => {
    useStore.getState().setZoom(1000)
    expect(useStore.getState().zoom).toBe(500)
    useStore.getState().setZoom(0.1)
    expect(useStore.getState().zoom).toBe(2)
  })

  it('keeps the anchored time fixed while zooming', () => {
    // time=10s under a cursor 100px from the left, zoom to 50 px/s
    useStore.getState().setZoom(50, 10, 100)
    const { zoom, scrollX } = useStore.getState()
    expect(zoom).toBe(50)
    expect(scrollX).toBeCloseTo(10 - 100 / 50) // 8s at left edge
  })

  it('seek clamps to [0, media.duration] and bumps the nonce', () => {
    useStore.setState({ project: demoProject }) // duration 29.4
    useStore.getState().seek(999)
    expect(useStore.getState().seekRequest?.t).toBe(29.4)
    useStore.getState().seek(-3)
    expect(useStore.getState().seekRequest?.t).toBe(0)
    expect(useStore.getState().playhead).toBe(0)
  })

  it('setScrollX never goes negative', () => {
    useStore.getState().setScrollX(-50)
    expect(useStore.getState().scrollX).toBe(0)
  })
})

describe('store: selection + toasts', () => {
  beforeEach(reset)

  it('selectSegment with seek moves the playhead to the segment start', () => {
    useStore.setState({ project: demoProject })
    useStore.getState().selectSegment('seg_019e618e', true)
    expect(useStore.getState().selection).toBe('seg_019e618e')
    expect(useStore.getState().seekRequest?.t).toBe(1.0)
  })

  it('re-resolves the capability map only when the pipeline changes', async () => {
    useStore.setState({ project: demoProject })
    await useStore.getState().loadResolvedCapabilities()
    expect(getProjectCapabilities).toHaveBeenCalledTimes(1)
    expect(useStore.getState().resolvedCapabilities.A1?.enabled).toBe(true)

    // An unrelated project update (a rename, an SSE tick) must not re-fetch.
    useStore.getState().applyProject({ ...demoProject, name: 'Renamed' })
    expect(getProjectCapabilities).toHaveBeenCalledTimes(1)

    // A pipeline edit changes what will run, so it must.
    useStore.getState().applyProject({
      ...demoProject,
      pipeline: { ...demoProject.pipeline, preset: 'custom' },
    })
    expect(getProjectCapabilities).toHaveBeenCalledTimes(2)
  })

  it('toast enqueues and dismiss removes it', () => {
    useStore.getState().toast('error', 'boom')
    const t = useStore.getState().toasts
    expect(t).toHaveLength(1)
    expect(t[0]).toMatchObject({ kind: 'error', text: 'boom' })
    useStore.getState().dismissToast(t[0].id)
    expect(useStore.getState().toasts).toHaveLength(0)
  })
})
