// Inline stroke icons (16px grid). `currentColor` so they follow the surrounding text colour.
import type { CSSProperties } from 'react'

const PATHS = {
  play: { d: 'M4.5 2.5v11l9-5.5z', fill: true },
  pause: { d: 'M4 2.5h3v11H4zM9 2.5h3v11H9z', fill: true },
  stop: { d: 'M3.5 3.5h9v9h-9z', fill: true },
  prev: { d: 'M12.5 3v10L5 8zM3.5 3v10' },
  next: { d: 'M3.5 3v10L11 8zM12.5 3v10' },
  back: { d: 'M10 3L5 8l5 5' },
  close: { d: 'M3.5 3.5l9 9M12.5 3.5l-9 9' },
  check: { d: 'M3 8.5l3.2 3.2L13 4.5' },
  warn: { d: 'M8 2l6.5 11.5h-13zM8 6.5v3.3M8 11.6v.3' },
  upload: { d: 'M8 11V2.5M4.5 6L8 2.5 11.5 6M2.5 11v2.5h11V11' },
  download: { d: 'M8 2.5V11M4.5 7.5L8 11l3.5-3.5M2.5 13.5h11' },
  arrow: { d: 'M2.5 8h11M9.5 4l4 4-4 4' },
  sun: { d: 'M8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M3 3l1.2 1.2M11.8 11.8L13 13M3 13l1.2-1.2M11.8 4.2L13 3' },
  moon: { d: 'M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7z' },
  chevron: { d: 'M4 6l4 4 4-4' },
  trash: { d: 'M2.5 4.5h11M6 4.5V3h4v1.5M4 4.5l.7 9h6.6l.7-9' },
} as const

export type IconName = keyof typeof PATHS

export function Icon({ name, size = 16, style }: { name: IconName; size?: number; style?: CSSProperties }) {
  const p = PATHS[name] as { d: string; fill?: boolean }
  return (
    <svg
      width={size} height={size} viewBox="0 0 16 16" aria-hidden="true"
      style={{ flexShrink: 0, display: 'block', ...style }}
      fill={p.fill ? 'currentColor' : 'none'}
      stroke={p.fill ? 'none' : 'currentColor'}
      strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"
    >
      <path d={p.d} />
    </svg>
  )
}

/** The OpenDub mark: two overlapping rings — original (accent) and dub (ok). */
export function LogoMark() {
  return (
    <svg width="26" height="18" viewBox="0 0 26 18" aria-hidden="true" style={{ flexShrink: 0 }}>
      <circle cx="9" cy="9" r="7.5" fill="none" stroke="var(--accent)" strokeWidth="1.6" />
      <circle cx="17" cy="9" r="7.5" fill="none" stroke="var(--ok)" strokeWidth="1.6" />
    </svg>
  )
}
