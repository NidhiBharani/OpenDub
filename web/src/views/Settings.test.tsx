import { beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { capabilityMap, demoProject, demoProviders, resolvedCapabilities } from '../test/fixtures'
import type { PresetBody, ProjectPatch } from '../api/client'
import type { Project } from '../types'

// Mock the API layer: Settings and the store both import this module.
const listProviders = vi.fn(async () => demoProviders)
const getCapabilities = vi.fn(async () => capabilityMap)
const getProjectCapabilities = vi.fn(async () => resolvedCapabilities)
const applyPreset = vi.fn(async (_pid: string, _body: PresetBody) => ({
  project: demoProject,
  notes: ['A4 Transcription: no cloud provider ready; using built-in'],
}))
const updateProject = vi.fn(async (_pid: string, patch: ProjectPatch): Promise<Project> => ({
  ...demoProject,
  pipeline: { ...demoProject.pipeline, ...(patch.mode ? { mode: patch.mode } : {}) },
}))

vi.mock('../api/client', () => ({
  api: {
    listProviders: () => listProviders(),
    getCapabilities: () => getCapabilities(),
    getProjectCapabilities: () => getProjectCapabilities(),
    applyPreset: (pid: string, body: PresetBody) => applyPreset(pid, body),
    updateProject: (pid: string, patch: ProjectPatch) => updateProject(pid, patch),
  },
  subscribeProjectEvents: () => () => {},
}))

// Import AFTER the mock is registered.
import { Settings } from './Settings'
import { useStore } from '../state/store'

function mount(mode: 'simple' | 'advanced' = 'simple') {
  useStore.setState({
    view: 'settings',
    project: { ...demoProject, pipeline: { ...demoProject.pipeline, mode } },
    projects: [],
    providers: [],
    capabilityMap: null,
    // What openProject would have loaded; Settings refreshes it on mount anyway.
    resolvedCapabilities,
    toasts: [],
  })
  return render(<Settings />)
}

/** The capability card for `capId`, once the map has loaded. */
function card(capId: string, name: string) {
  return screen.getByRole('article', { name: `${capId} ${name}` })
}

describe('Settings — Simple mode', () => {
  beforeEach(() => vi.clearAllMocks())

  it('applies a preset through the preset endpoint', async () => {
    mount()
    // The preset cards come from GET /api/capabilities.
    await screen.findByRole('radio', { name: /Balanced/ })

    await userEvent.click(screen.getByRole('radio', { name: /Balanced/ }))
    await userEvent.click(screen.getByRole('checkbox', { name: /Subtitles/ }))
    await userEvent.click(screen.getByRole('button', { name: /Apply/ }))

    await waitFor(() => expect(applyPreset).toHaveBeenCalledTimes(1))
    expect(applyPreset).toHaveBeenCalledWith(demoProject.id, {
      preset: 'balanced', runtime: 'builtin', lipsync: false, subtitles: true,
    })
    // Fallback notes from the server are surfaced, not swallowed.
    await waitFor(() =>
      expect(useStore.getState().toasts.some((t) => t.text.includes('using built-in'))).toBe(true))
  })

  it('counts each preset and summarises what will run', async () => {
    mount()
    // "N of 41 capabilities" — here 2 of the 4 in the fixture map.
    expect(await screen.findByText(/2 of 4 capabilities/)).toBeInTheDocument()

    const summary = screen.getByRole('region', { name: 'What will run' })
    await waitFor(() => expect(within(summary).getByText('A1')).toBeInTheDocument())
    // A1 is on with its provider named; the off ones say why the resolver turned them off.
    expect(within(summary).getByText('Passthrough')).toBeInTheDocument()
    expect(within(summary).getAllByText('off').length).toBeGreaterThan(0)
    expect(within(summary).getByText("lip sync provider is 'none'")).toBeInTheDocument()
  })

  it('says so when Advanced has customised the pipeline', async () => {
    mount()
    act(() => useStore.setState({
      project: { ...demoProject, pipeline: { ...demoProject.pipeline, preset: 'custom' } },
    }))
    expect(await screen.findByText(/Customised in Advanced/)).toBeInTheDocument()
    // Apply is still available — it is how you get back onto a preset.
    expect(screen.getByRole('button', { name: /Apply/ })).toBeEnabled()
  })
})

describe('Settings — Advanced mode', () => {
  beforeEach(() => vi.clearAllMocks())

  it('switching modes persists pipeline.mode', async () => {
    mount()
    await userEvent.click(screen.getByRole('button', { name: 'Advanced' }))
    await waitFor(() => expect(updateProject).toHaveBeenCalledWith(demoProject.id, { mode: 'advanced' }))
  })

  it('toggling a non-legacy capability patches capabilities[id] with a full choice', async () => {
    mount('advanced')
    const a3 = await waitFor(() => card('A3', 'Speech / music / singing regions'))

    await userEvent.click(within(a3).getByRole('checkbox', { name: /Enabled/ }))

    await waitFor(() => expect(updateProject).toHaveBeenCalled())
    expect(updateProject.mock.calls.at(-1)).toEqual([demoProject.id, {
      capabilities: {
        A3: {
          enabled: true,
          provider_id: 'region_detect.off',
          options: {},
          params: { song_action: 'passthrough' },
        },
      },
    }])
  })

  it('picking a provider for a legacy capability goes through pipeline[legacy_field]', async () => {
    mount('advanced')
    const a1 = await waitFor(() => card('A1', 'Dialogue / M&E separation'))

    await userEvent.selectOptions(
      within(a1).getByRole('combobox', { name: /Provider/ }), 'separation.demucs',
    )

    await waitFor(() => expect(updateProject).toHaveBeenCalled())
    // Sparse: only the changed kind, never the whole PipelineConfig (mode/preset are not kinds).
    expect(updateProject.mock.calls.at(-1)).toEqual([demoProject.id, {
      pipeline: { separation: { provider_id: 'separation.demucs', options: {} } },
    }])
  })

  it('locks core capabilities on and leaves lip sync to its provider picker', async () => {
    mount('advanced')
    const a1 = await waitFor(() => card('A1', 'Dialogue / M&E separation'))
    expect(within(a1).getByRole('checkbox', { name: /Always on/ })).toBeDisabled()

    await userEvent.click(screen.getByRole('button', { name: /Picture generation/ }))
    const f1 = card('F1', 'Live-action lip sync')
    expect(within(f1).queryByRole('checkbox')).not.toBeInTheDocument()
    expect(within(f1).getByRole('combobox', { name: /Provider/ })).toHaveValue('lipsync.none')
  })

  it('renders deferred capabilities as "later" with no controls', async () => {
    mount('advanced')
    await waitFor(() => card('A1', 'Dialogue / M&E separation'))

    await userEvent.click(screen.getByRole('button', { name: /Audio post-production/ }))
    const e5 = card('E5', 'Watermark + provenance')
    expect(within(e5).getByText('later')).toBeInTheDocument()
    expect(within(e5).queryByRole('checkbox')).not.toBeInTheDocument()
    expect(within(e5).queryByRole('combobox')).not.toBeInTheDocument()
  })

  it('dependency chips jump to the capability they name', async () => {
    mount('advanced')
    const a3 = await waitFor(() => card('A3', 'Speech / music / singing regions'))

    await userEvent.click(within(a3).getByRole('button', { name: 'A1' }))
    // A1 lives in the same phase here; the jump focuses its card.
    await waitFor(() => expect(card('A1', 'Dialogue / M&E separation')).toHaveFocus())
  })
})
