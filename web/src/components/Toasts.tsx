import { useStore } from '../state/store'

export function Toasts() {
  const toasts = useStore((s) => s.toasts)
  const dismiss = useStore((s) => s.dismissToast)
  if (toasts.length === 0) return null
  const colors = { error: 'var(--err)', success: 'var(--ok)' }
  return (
    <div style={{
      position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)',
      display: 'flex', flexDirection: 'column', gap: 8, zIndex: 200, alignItems: 'center',
    }}>
      {toasts.map((t) => (
        <div
          role="status"
          key={t.id}
          onClick={() => dismiss(t.id)}
          style={{
            padding: '8px 16px', borderRadius: 'var(--r-md)', cursor: 'pointer',
            background: 'var(--bg-overlay)', border: '1px solid var(--border-strong)',
            boxShadow: 'var(--shadow)', display: 'flex', alignItems: 'center', gap: 10,
            fontSize: 13, maxWidth: 480, animation: 'od-slide-up 150ms ease-out',
          }}
        >
          <span style={{ width: 8, height: 8, borderRadius: '50%', flexShrink: 0, background: colors[t.kind] }} />
          {t.text}
        </div>
      ))}
    </div>
  )
}
