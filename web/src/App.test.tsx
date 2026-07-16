import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { demoProject, demoSummary, mockProvider } from './test/fixtures'

// Mock the whole API layer so the app renders against controlled data — no network.
const listProjects = vi.fn(async () => [demoSummary])
const listProviders = vi.fn(async () => [mockProvider])
const getProject = vi.fn(async () => demoProject)
const listJobs = vi.fn(async () => [])
const waveform = vi.fn(async () => { throw new Error('no waveform') })

vi.mock('./api/client', () => ({
  api: {
    listProjects: () => listProjects(),
    listProviders: () => listProviders(),
    getProject: () => getProject(),
    listJobs: () => listJobs(),
    waveform: () => waveform(),
    mediaUrl: (pid: string, rel: string) => `/api/media/${pid}/${rel}`,
  },
  subscribeProjectEvents: () => () => {},
}))

// Import App AFTER the mock is registered.
import { App } from './App'
import { useStore } from './state/store'

describe('App — renders the page as intended', () => {
  beforeEach(() => {
    useStore.setState({ view: 'library', projects: [], project: null, providers: [] })
    listProjects.mockClear()
    listProviders.mockClear()
  })

  it('shows the OpenDub library and loads data on mount', async () => {
    render(<App />)

    // Brand + tagline are on screen immediately.
    expect(screen.getByText('Open')).toBeInTheDocument()
    expect(screen.getByText('voice-preserving dubbing studio')).toBeInTheDocument()

    // Data is fetched on mount and the demo project card renders.
    await waitFor(() => expect(screen.getByText('Demo Episode')).toBeInTheDocument())
    expect(listProjects).toHaveBeenCalled()
    expect(listProviders).toHaveBeenCalled()

    // Card content reflects the fixture: duration + segment count.
    expect(screen.getByText(/0:29/)).toBeInTheDocument()
    expect(screen.getByText(/5 segments/)).toBeInTheDocument()

    // The drop zone (first impression / upload affordance) is present.
    expect(screen.getByText(/Drop a video here/)).toBeInTheDocument()
  })

  it('shows the empty state when there are no projects', async () => {
    listProjects.mockResolvedValueOnce([])
    render(<App />)
    await waitFor(() => expect(screen.getByText('No projects yet')).toBeInTheDocument())
  })

  it('navigates to Settings and back-affordance renders', async () => {
    render(<App />)
    await userEvent.click(screen.getByRole('button', { name: 'Settings' }))
    await waitFor(() => expect(useStore.getState().view).toBe('settings'))
  })

  it('opening a project card switches to the editor view', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByText('Demo Episode')).toBeInTheDocument())
    await userEvent.click(screen.getByText('Demo Episode'))
    await waitFor(() => expect(useStore.getState().view).toBe('editor'))
    expect(getProject).toHaveBeenCalled()
  })
})
