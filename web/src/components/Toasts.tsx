import { StatusDot } from './primitives'
import { useStore } from '../state/store'

export function Toasts() {
  const toasts = useStore((s) => s.toasts)
  const dismiss = useStore((s) => s.dismissToast)
  if (toasts.length === 0) return null
  const colors = { error: 'var(--err)', success: 'var(--ok)' }
  return (
    <div style={{
      position: 'fixed', bottom: 12, left: '50%', transform: 'translateX(-50%)',
      display: 'flex', flexDirection: 'column', gap: 6, zIndex: 200, alignItems: 'center',
    }}>
      {toasts.map((t) => (
        <div
          role="status"
          key={t.id}
          onClick={() => dismiss(t.id)}
          title="Dismiss"
          style={{
            padding: '6px 12px', borderRadius: 'var(--r-md)', cursor: 'pointer',
            background: 'var(--bg-panel)', border: '1px solid var(--border)',
            boxShadow: 'var(--shadow)', display: 'flex', alignItems: 'center', gap: 8,
            fontSize: 12, maxWidth: 480, animation: 'od-slide-up 150ms ease-out', color: 'var(--text)',
          }}
        >
          <StatusDot color={colors[t.kind]} />
          {t.text}
        </div>
      ))}
    </div>
  )
}
