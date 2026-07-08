// Right-hand panel: tab strip switching between the segment Inspector and the
// synced Transcript list.
import { useEffect, useState } from 'react'
import { Button } from '../components/primitives'
import { useStore } from '../state/store'
import { Inspector } from './Inspector'
import { TranscriptList } from './TranscriptList'

type Tab = 'segment' | 'transcript'

export function RightPanel() {
  const selection = useStore((s) => s.selection)
  const [tab, setTab] = useState<Tab>('transcript')

  // Auto-switch to the Segment tab whenever a new (non-null) selection comes
  // in. Manual tab choices are respected otherwise — this effect only fires
  // when `selection` itself changes.
  useEffect(() => {
    if (selection) setTab('segment')
  }, [selection])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <div style={{ display: 'flex', flexShrink: 0, borderBottom: '1px solid var(--border)' }}>
        <TabButton label="Segment" active={tab === 'segment'} onClick={() => setTab('segment')} />
        <TabButton label="Transcript" active={tab === 'transcript'} onClick={() => setTab('transcript')} />
      </div>
      <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
        {tab === 'segment' ? <Inspector /> : <TranscriptList />}
      </div>
    </div>
  )
}

function TabButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <Button
      variant="ghost"
      onClick={onClick}
      style={{
        flex: 1,
        justifyContent: 'center',
        borderRadius: 0,
        padding: '10px 4px 8px',
        color: active ? 'var(--text)' : 'var(--text-dim)',
        borderBottom: active ? '2px solid var(--accent)' : '2px solid transparent',
      }}
    >
      {label}
    </Button>
  )
}
