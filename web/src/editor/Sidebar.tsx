// Left sidebar: Script | Versions | Cast. The header is a segmented tab control; the Versions tab
// adds a "Snapshot now" button.
import { IconButton, Segmented } from '../components/primitives'
import { useStore } from '../state/store'
import type { SidebarTab } from '../state/store'
import { CastPanel } from './CastPanel'
import { ScriptPanel } from './TranscriptList'
import { VersionsPanel } from './VersionsPanel'

const TABS: { value: SidebarTab; label: string }[] = [
  { value: 'script', label: 'Script' }, { value: 'versions', label: 'Versions' }, { value: 'cast', label: 'Cast' },
]

export function Sidebar() {
  const tab = useStore((s) => s.sidebarTab)
  const setTab = useStore((s) => s.setSidebarTab)
  const snapshotVersion = useStore((s) => s.snapshotVersion)
  const toast = useStore((s) => s.toast)
  const busy = useStore((s) => Object.values(s.jobs).some((j) => j.status === 'queued' || j.status === 'running'))

  async function snapshot() {
    try {
      await snapshotVersion()
      toast('success', 'Snapshot saved')
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to save a snapshot')
    }
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div className="panel-header" style={{ paddingLeft: 8 }}>
        <Segmented<SidebarTab> size="sm" ariaLabel="Sidebar panel" value={tab} onChange={setTab} options={TABS} />
        <div style={{ flex: 1 }} />
        {tab === 'versions' && (
          <IconButton name="camera" title="Snapshot now (⌘⇧D)" onClick={() => void snapshot()} disabled={busy} />
        )}
      </div>
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        {tab === 'script' && <ScriptPanel />}
        {tab === 'versions' && <VersionsPanel />}
        {tab === 'cast' && <CastPanel />}
      </div>
    </div>
  )
}
