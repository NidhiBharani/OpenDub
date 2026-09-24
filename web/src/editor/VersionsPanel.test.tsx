import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { demoProject } from '../test/fixtures'
import type { Version } from '../types'

const restoreVersion = vi.fn()
const deleteVersion = vi.fn()
const renameVersion = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    restoreVersion: (...args: unknown[]) => restoreVersion(...args),
    deleteVersion: (...args: unknown[]) => deleteVersion(...args),
    renameVersion: (...args: unknown[]) => renameVersion(...args),
    listVersions: async () => [],
    waveform: async () => { throw new Error('no waveform') },
    mediaUrl: (pid: string, rel: string) => `/api/media/${pid}/${rel}`,
  },
}))

import { VersionsPanel, versionSummary } from './VersionsPanel'
import { useStore } from '../state/store'

const base: Version = {
  id: 'ver_1', label: 'Run · 24 Sep 14:03', kind: 'auto', created_at: '2026-09-24T14:03:00Z',
  source_lang: 'ja', target_lang: 'en', preset: 'minimal', runtime: 'builtin',
  providers: { tts: 'tts.f5_tts' }, stages: {}, segment_count: 27, voiced_count: 27, skipped_ranges: 2,
  has_mix: true, has_render: false,
  outputs: { playback_dub: 'versions/ver_1/playback_dub.mp4' },
  summary: '', bytes: 14 * 1024 * 1024, job_id: null, parent_id: null,
}
const versions: Version[] = [
  base,
  { ...base, id: 'ver_2', label: 'Before restore', kind: 'pre_restore', created_at: '2026-09-23T09:10:00Z', bytes: 0 },
  { ...base, id: 'ver_3', label: 'Hindi first pass', kind: 'manual', target_lang: 'hi', created_at: '2026-09-24T11:00:00Z' },
]

describe('VersionsPanel', () => {
  beforeEach(() => {
    useStore.setState({ project: { ...demoProject, active_version_id: 'ver_1' }, versions, previewVersionId: null, toasts: [] })
    restoreVersion.mockReset()
    deleteVersion.mockReset()
    renameVersion.mockReset()
  })

  it('groups versions by target language, defaulting to the project language, newest first', async () => {
    render(<VersionsPanel />)
    const tabs = screen.getByRole('group', { name: 'Target language' })
    expect(within(tabs).getByRole('button', { name: /EN 2/ })).toHaveAttribute('aria-pressed', 'true')
    expect(within(tabs).getByRole('button', { name: /HI 1/ })).toBeInTheDocument()

    const items = screen.getAllByRole('listitem')
    expect(items.map((li) => li.getAttribute('aria-label'))).toEqual(['Run · 24 Sep 14:03', 'Before restore'])
    expect(within(items[0]).getByText('current')).toBeInTheDocument()
    expect(within(items[1]).getByText('restore point')).toBeInTheDocument()

    await userEvent.click(within(tabs).getByRole('button', { name: /HI 1/ }))
    expect(screen.getAllByRole('listitem').map((li) => li.getAttribute('aria-label'))).toEqual(['Hindi first pass'])
    expect(screen.getByText('manual')).toBeInTheDocument()
  })

  it('summarises a version on one timecode line', () => {
    expect(versionSummary(base)).toMatch(/^\d{1,2}:\d{2}.* · minimal\/builtin · tts\.f5_tts · 27 lines · 27 voiced · 2 kept · 14 MB$/)
  })

  it('shows the empty state when nothing has been snapshotted', () => {
    useStore.setState({ versions: [] })
    render(<VersionsPanel />)
    expect(screen.getByText(/Versions appear after each run that completes Mix/)).toBeInTheDocument()
  })

  it('preview toggles the previewed version', async () => {
    render(<VersionsPanel />)
    const row = screen.getByRole('listitem', { name: 'Before restore' })
    await userEvent.click(within(row).getByRole('button', { name: 'Preview in the viewer' }))
    expect(useStore.getState().previewVersionId).toBe('ver_2')
    expect(row).toHaveAttribute('aria-current', 'true')
    await userEvent.click(within(row).getByRole('button', { name: 'Stop previewing' }))
    expect(useStore.getState().previewVersionId).toBeNull()
  })

  it('restore asks first, then calls the API', async () => {
    restoreVersion.mockResolvedValue({ ...demoProject, active_version_id: 'ver_2' })
    render(<VersionsPanel />)
    const row = screen.getByRole('listitem', { name: 'Before restore' })
    await userEvent.click(within(row).getByRole('button', { name: 'Restore this version' }))
    const dialog = screen.getByRole('dialog', { name: 'Restore this version?' })
    expect(dialog).toHaveTextContent(/saved first as a restore point/)
    expect(restoreVersion).not.toHaveBeenCalled()
    await userEvent.click(within(dialog).getByRole('button', { name: 'Restore' }))
    await waitFor(() => expect(restoreVersion).toHaveBeenCalledWith(demoProject.id, 'ver_2'))
    await waitFor(() => expect(useStore.getState().project?.active_version_id).toBe('ver_2'))
  })

  it('rename edits inline and delete asks first', async () => {
    renameVersion.mockImplementation(async (_pid: string, vid: string, label: string) => ({ ...base, id: vid, label }))
    deleteVersion.mockResolvedValue({ ok: true })
    render(<VersionsPanel />)
    const row = screen.getByRole('listitem', { name: 'Run · 24 Sep 14:03' })
    await userEvent.click(within(row).getByRole('button', { name: 'Rename' }))
    const input = screen.getByRole('textbox', { name: 'Version label' })
    await userEvent.clear(input)
    await userEvent.type(input, 'Final{Enter}')
    await waitFor(() => expect(renameVersion).toHaveBeenCalledWith(demoProject.id, 'ver_1', 'Final'))
    await waitFor(() => expect(screen.getByRole('listitem', { name: 'Final' })).toBeInTheDocument())

    await userEvent.click(within(screen.getByRole('listitem', { name: 'Final' })).getByRole('button', { name: 'Delete' }))
    await userEvent.click(within(screen.getByRole('dialog', { name: 'Delete this version?' })).getByRole('button', { name: 'Delete' }))
    await waitFor(() => expect(deleteVersion).toHaveBeenCalledWith(demoProject.id, 'ver_1'))
    await waitFor(() => expect(screen.queryByRole('listitem', { name: 'Final' })).toBeNull())
  })
})
