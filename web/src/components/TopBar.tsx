// App-wide top bar: brand, optional left/center/right slots, theme toggle.
import type { ReactNode } from 'react'
import { useStore } from '../state/store'
import { Icon, LogoMark } from './Icon'

export function Brand({ onClick }: { onClick?: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title="Library"
      style={{
        display: 'flex', alignItems: 'center', gap: 10, border: 'none', background: 'transparent',
        padding: 0, cursor: onClick ? 'pointer' : 'default', color: 'var(--text)',
      }}
    >
      <LogoMark />
      <span className="serif" style={{ fontStyle: 'italic', fontSize: 24, lineHeight: 1 }}>
        <span>Open</span><span>Dub</span>
      </span>
    </button>
  )
}

export function ThemeToggle() {
  const theme = useStore((s) => s.theme)
  const setTheme = useStore((s) => s.setTheme)
  const next = theme === 'dark' ? 'light' : 'dark'
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
      style={{
        width: 36, height: 36, borderRadius: '50%', border: '1px solid var(--border-strong)',
        background: 'transparent', color: 'var(--text-dim)', cursor: 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}
    >
      <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
    </button>
  )
}

export function NavLink({ label, active, onClick }: { label: string; active?: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? 'page' : undefined}
      style={{
        border: 'none', background: 'transparent', cursor: 'pointer', fontSize: 13, padding: '8px 2px',
        color: active ? 'var(--text)' : 'var(--text-dim)', fontWeight: active ? 500 : 400,
      }}
    >
      {label}
    </button>
  )
}

export function TopBar({ onBrand, children, padX = 20 }: { onBrand?: () => void; children?: ReactNode; padX?: number }) {
  return (
    <header
      style={{
        height: 56, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 14,
        padding: `0 ${padX}px`, borderBottom: '1px solid var(--border-strong)', background: 'var(--bg-raised)',
      }}
    >
      <Brand onClick={onBrand} />
      {children}
      <ThemeToggle />
    </header>
  )
}
