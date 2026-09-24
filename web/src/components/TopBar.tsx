// App-wide title bar (38px): brand, optional slots, theme toggle. Neutral chrome, one sans.
import type { ReactNode } from 'react'
import { useStore } from '../state/store'
import { LogoMark } from './Icon'
import { IconButton } from './primitives'

export function Brand({ onClick }: { onClick?: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={onClick ? 'Library' : undefined}
      aria-label="OpenDub"
      style={{
        display: 'flex', alignItems: 'center', gap: 8, border: 'none', background: 'transparent',
        padding: 0, cursor: onClick ? 'pointer' : 'default', color: 'var(--text)', flexShrink: 0,
      }}
    >
      <LogoMark size={16} />
      <span style={{ fontSize: 13, fontWeight: 600, lineHeight: 1, letterSpacing: '-0.01em', whiteSpace: 'nowrap' }}>
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
    <IconButton
      name={theme === 'dark' ? 'sun' : 'moon'}
      title={`Switch to ${next} mode`}
      onClick={() => setTheme(next)}
    />
  )
}

export function NavLink({ label, active, onClick }: { label: string; active?: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? 'page' : undefined}
      style={{
        border: 'none', background: 'transparent', cursor: 'pointer', fontSize: 12, padding: '0 6px', height: 24,
        borderRadius: 'var(--r-md)',
        color: active ? 'var(--text)' : 'var(--text-dim)', fontWeight: active ? 600 : 500,
      }}
    >
      {label}
    </button>
  )
}

export function TopBar({ onBrand, children, padX = 12 }: { onBrand?: () => void; children?: ReactNode; padX?: number }) {
  return (
    <header
      style={{
        height: 'var(--titlebar-h)', flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10,
        padding: `0 ${padX}px`, borderBottom: '1px solid var(--border)', background: 'var(--bg-raised)',
      }}
    >
      <Brand onClick={onBrand} />
      {children}
      <ThemeToggle />
    </header>
  )
}
