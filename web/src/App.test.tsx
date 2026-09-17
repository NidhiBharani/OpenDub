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
const deleteProject = vi.fn(async () => ({ ok: true }))

vi.mock('./api/client', () => ({
  api: {
    listProjects: () => listProjects(),
    listProviders: () => listProviders(),
    getProject: () => getProject(),
    deleteProject: () => deleteProject(),
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
    getProject.mockClear()
    deleteProject.mockClear()
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
    expect(screen.getByText(/5 lines/)).toBeInTheDocument()

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

  // The delete dialog is mounted inside the clickable card; its clicks must not open the project.
  async function openDeleteDialog() {
    render(<App />)
    const card = await screen.findByText('Demo Episode')
    await userEvent.hover(card)
    await userEvent.click(screen.getByTitle('Delete project'))
    expect(await screen.findByText(/Cannot be undone/)).toBeInTheDocument()
  }

  it('cancelling the delete dialog stays in the library', async () => {
    await openDeleteDialog()
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(screen.queryByText(/Cannot be undone/)).not.toBeInTheDocument())
    expect(useStore.getState().view).toBe('library')
    expect(getProject).not.toHaveBeenCalled()
  })

  it('confirming delete deletes without opening the project', async () => {
    await openDeleteDialog()
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }))
    await waitFor(() => expect(deleteProject).toHaveBeenCalled())
    expect(useStore.getState().view).toBe('library')
    expect(getProject).not.toHaveBeenCalled()
  })

  it('renders dialogs at the document root, outside the transformed card', async () => {
    await openDeleteDialog()
    const overlay = screen.getByText(/Cannot be undone/).closest('div[style*="position: fixed"]')
    expect(overlay?.parentElement).toBe(document.body)
  })
})
