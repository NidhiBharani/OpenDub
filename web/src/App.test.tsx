import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { capabilityMap, demoProject, demoSummary, mockProvider, resolvedCapabilities } from './test/fixtures'

// Mock the whole API layer so the app renders against controlled data — no network.
const listProjects = vi.fn(async () => [demoSummary])
const listProviders = vi.fn(async () => [mockProvider])
const getCapabilities = vi.fn(async () => capabilityMap)
const getProjectCapabilities = vi.fn(async () => resolvedCapabilities)
const getProject = vi.fn(async () => demoProject)
const listJobs = vi.fn(async () => [])
const waveform = vi.fn(async () => { throw new Error('no waveform') })
const deleteProject = vi.fn(async () => ({ ok: true }))
const createProject = vi.fn(async (..._args: unknown[]) => demoProject)

vi.mock('./api/client', () => ({
  api: {
    listProjects: () => listProjects(),
    listProviders: () => listProviders(),
    getCapabilities: () => getCapabilities(),
    getProjectCapabilities: () => getProjectCapabilities(),
    getProject: () => getProject(),
    deleteProject: () => deleteProject(),
    createProject: (...args: unknown[]) => createProject(...args),
    listJobs: () => listJobs(),
    waveform: () => waveform(),
    mediaUrl: (pid: string, rel: string) => `/api/media/${pid}/${rel}`,
  },
  subscribeProjectEvents: () => () => {},
}))

// Import App AFTER the mock is registered.
import { App } from './App'
import { useStore } from './state/store'

// Node 22+ defines a global `localStorage` getter that yields undefined unless the process is
// started with --localstorage-file, and it shadows jsdom's. Give the tests a real in-memory one.
function memoryStorage(): Storage {
  const map = new Map<string, string>()
  return {
    get length() { return map.size },
    clear: () => map.clear(),
    getItem: (k) => map.get(k) ?? null,
    key: (i) => [...map.keys()][i] ?? null,
    removeItem: (k) => { map.delete(k) },
    setItem: (k, v) => { map.set(k, String(v)) },
  }
}
Object.defineProperty(globalThis, 'localStorage', { value: memoryStorage(), configurable: true, writable: true })

describe('App — renders the page as intended', () => {
  beforeEach(() => {
    useStore.setState({
      view: 'library', projects: [], project: null, providers: [],
      capabilityMap: null, resolvedCapabilities: {},
    })
    listProjects.mockClear()
    listProviders.mockClear()
    getProject.mockClear()
    deleteProject.mockClear()
    createProject.mockClear()
    localStorage.clear()
  })

  it('shows the OpenDub library and loads data on mount', async () => {
    render(<App />)

    // Brand + the library toolbar are on screen immediately.
    expect(screen.getByText('Open')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /New project/ })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Search projects' })).toBeInTheDocument()

    // Data is fetched on mount and the demo project card renders.
    await waitFor(() => expect(screen.getByText('Demo Episode')).toBeInTheDocument())
    expect(listProjects).toHaveBeenCalled()
    expect(listProviders).toHaveBeenCalled()

    // Card content reflects the fixture: duration, target language and segment count.
    expect(screen.getByText(/0:29/)).toBeInTheDocument()
    expect(screen.getByText(/→ EN/)).toBeInTheDocument()
    expect(screen.getByText(/5 lines/)).toBeInTheDocument()
  })

  it('shows the empty state when there are no projects', async () => {
    listProjects.mockResolvedValueOnce([])
    render(<App />)
    await waitFor(() => expect(screen.getByText(/No projects yet/)).toBeInTheDocument())
  })

  it('filters by search and switches between grid and list views', async () => {
    render(<App />)
    await screen.findByText('Demo Episode')

    await userEvent.type(screen.getByRole('textbox', { name: 'Search projects' }), 'nothing like this')
    expect(screen.queryByText('Demo Episode')).not.toBeInTheDocument()
    expect(screen.getByText(/No project matches/)).toBeInTheDocument()
    await userEvent.clear(screen.getByRole('textbox', { name: 'Search projects' }))

    // List view is a table; the choice is remembered.
    await userEvent.click(screen.getByRole('button', { name: 'List' }))
    expect(screen.getByRole('table')).toBeInTheDocument()
    expect(screen.getByRole('row', { name: 'Demo Episode' })).toBeInTheDocument()
    expect(localStorage.getItem('opendub.library.view')).toBe('list')

    await userEvent.click(screen.getByRole('button', { name: 'Grid' }))
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Demo Episode' })).toBeInTheDocument()
  })

  it('creates a project with the chosen languages and remembers the target', async () => {
    render(<App />)
    await screen.findByText('Demo Episode')

    const file = new File(['00'], 'kumo.mp4', { type: 'video/mp4' })
    await userEvent.upload(screen.getByLabelText('Choose a video file'), file)

    const dialog = await screen.findByRole('dialog', { name: 'New project' })
    expect(within(dialog).getByRole('textbox', { name: 'Name' })).toHaveValue('kumo')
    await userEvent.selectOptions(within(dialog).getByRole('combobox', { name: 'Source language' }), 'ja')
    await userEvent.selectOptions(within(dialog).getByRole('combobox', { name: 'Target language' }), 'hi')
    await userEvent.click(within(dialog).getByRole('button', { name: 'Create project' }))

    await waitFor(() => expect(createProject).toHaveBeenCalledTimes(1))
    const [sent, name, source, target] = createProject.mock.calls[0]
    expect(sent).toBe(file)
    expect(name).toBe('kumo')
    expect(source).toBe('ja')
    expect(target).toBe('hi')
    expect(localStorage.getItem('opendub.library.target')).toBe('hi')
    // The new project opens in the editor.
    await waitFor(() => expect(useStore.getState().view).toBe('editor'))
  })

  it('accepts a dropped video anywhere in the library body', async () => {
    render(<App />)
    await screen.findByText('Demo Episode')

    const file = new File(['00'], 'dropped.mkv', { type: 'video/x-matroska' })
    const body = screen.getByText('Demo Episode').closest('div[style*="overflow-y: auto"]')!
    fireEvent.drop(body, { dataTransfer: { files: [file], types: ['Files'] } })

    const dialog = await screen.findByRole('dialog', { name: 'New project' })
    expect(within(dialog).getByRole('textbox', { name: 'Name' })).toHaveValue('dropped')
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
    const card = await screen.findByRole('button', { name: 'Demo Episode' })
    await userEvent.hover(card)
    await userEvent.click(within(card).getByTitle('Delete project'))
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

  it('renders dialogs at the document root, outside the card', async () => {
    await openDeleteDialog()
    const overlay = screen.getByText(/Cannot be undone/).closest('div[style*="position: fixed"]')
    expect(overlay?.parentElement).toBe(document.body)
  })
})
