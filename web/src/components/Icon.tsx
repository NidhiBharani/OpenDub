// Inline stroke icons (16px grid). `currentColor` so they follow the surrounding text colour.
import type { CSSProperties } from 'react'

const PATHS = {
  play: { d: 'M4.5 2.5v11l9-5.5z', fill: true },
  pause: { d: 'M4 2.5h3v11H4zM9 2.5h3v11H9z', fill: true },
  stop: { d: 'M3.5 3.5h9v9h-9z', fill: true },
  prev: { d: 'M12.5 3v10L5 8zM3.5 3v10' },
  next: { d: 'M3.5 3v10L11 8zM12.5 3v10' },
  back: { d: 'M10 3L5 8l5 5' },
  forward: { d: 'M6 3l5 5-5 5' },
  close: { d: 'M3.5 3.5l9 9M12.5 3.5l-9 9' },
  check: { d: 'M3 8.5l3.2 3.2L13 4.5' },
  warn: { d: 'M8 2l6.5 11.5h-13zM8 6.5v3.3M8 11.6v.3' },
  upload: { d: 'M8 11V2.5M4.5 6L8 2.5 11.5 6M2.5 11v2.5h11V11' },
  download: { d: 'M8 2.5V11M4.5 7.5L8 11l3.5-3.5M2.5 13.5h11' },
  arrow: { d: 'M2.5 8h11M9.5 4l4 4-4 4' },
  sun: { d: 'M8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M3 3l1.2 1.2M11.8 11.8L13 13M3 13l1.2-1.2M11.8 4.2L13 3' },
  moon: { d: 'M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7z' },
  chevron: { d: 'M4 6l4 4 4-4' },
  chevronRight: { d: 'M6 4l4 4-4 4' },
  trash: { d: 'M2.5 4.5h11M6 4.5V3h4v1.5M4 4.5l.7 9h6.6l.7-9' },
  refresh: { d: 'M13 8a5 5 0 1 1-1.5-3.6M13 2.5v3h-3' },
  translate: { d: 'M2.5 4h7M6 2.5V4M4 4c.5 3 2.5 5.5 5 6.5M8 4c-.5 3-2.5 5.5-5 6.5M9 13.5l2.5-6 2.5 6M10 11.5h3' },
  mic: { d: 'M8 2a2 2 0 0 0-2 2v4a2 2 0 0 0 4 0V4a2 2 0 0 0-2-2zM4 8a4 4 0 0 0 8 0M8 12v2M6 14h4' },
  split: { d: 'M8 2v12M3 5l3 3-3 3M13 5l-3 3 3 3' },
  merge: { d: 'M3 4l4 4-4 4M13 4l-4 4 4 4' },
  scissors: { d: 'M4.5 3a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM4.5 10a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM6 5l7.5 8M6 11l7.5-8' },
  more: { d: 'M3 8h.01M8 8h.01M13 8h.01', dots: true },
  marker: { d: 'M4 2.5h8v9L8 9l-4 2.5z' },
  skip: { d: 'M3 3l10 10M3 13L13 3M2.5 8h11', },
  ban: { d: 'M8 2.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11zM4.2 4.2l7.6 7.6' },
  history: { d: 'M3 8a5 5 0 1 0 1.5-3.6M3 2.5v3h3M8 5v3l2 1.5' },
  camera: { d: 'M2.5 5h2.5l1-1.5h4l1 1.5h2.5v8h-11zM8 7a2 2 0 1 0 0 4 2 2 0 0 0 0-4z' },
  search: { d: 'M7 2.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9zM10.3 10.3L13.5 13.5' },
  settings: { d: 'M8 5.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5zM8 1.5v1.8M8 12.7v1.8M1.5 8h1.8M12.7 8h1.8M3.4 3.4l1.3 1.3M11.3 11.3l1.3 1.3M3.4 12.6l1.3-1.3M11.3 4.7l1.3-1.3' },
  grid: { d: 'M2.5 2.5h4.5v4.5H2.5zM9 2.5h4.5v4.5H9zM2.5 9h4.5v4.5H2.5zM9 9h4.5v4.5H9z' },
  list: { d: 'M2.5 4h11M2.5 8h11M2.5 12h11' },
  plus: { d: 'M8 3v10M3 8h10' },
  minus: { d: 'M3 8h10' },
  fit: { d: 'M2.5 5.5v-3h3M13.5 5.5v-3h-3M2.5 10.5v3h3M13.5 10.5v3h-3' },
  magnet: { d: 'M4 2.5v6a4 4 0 0 0 8 0v-6M4 2.5h2.5v6a1.5 1.5 0 0 0 3 0v-6H12' },
  loop: { d: 'M3 6.5V5a1.5 1.5 0 0 1 1.5-1.5h7L10 2M13 9.5V11a1.5 1.5 0 0 1-1.5 1.5h-7L6 14' },
  volume: { d: 'M2.5 6h2.5L8.5 3v10L5 10H2.5zM10.5 5.5a3.5 3.5 0 0 1 0 5M12.5 3.5a6 6 0 0 1 0 9' },
  film: { d: 'M2.5 2.5h11v11h-11zM2.5 5.5h11M2.5 10.5h11M5.5 2.5v11M10.5 2.5v11' },
  speaker: { d: 'M8 2.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5zM3 13.5a5 5 0 0 1 10 0' },
  external: { d: 'M9 2.5h4.5V7M13.5 2.5L7 9M11 9v4.5H2.5V5H7' },
  info: { d: 'M8 2.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11zM8 7v4M8 5v.3' },
  restore: { d: 'M3 8a5 5 0 1 0 1.5-3.6M3 2.5v3h3M6 8l2 2 3-3.5' },
  tag: { d: 'M2.5 2.5h5l6 6-5 5-6-6zM5.5 5.5h.01' },
  in: { d: 'M3 2.5v11M3 8h8M8 5l3 3-3 3' },
  out: { d: 'M13 2.5v11M13 8H5M8 5L5 8l3 3' },
  eye: { d: 'M1.5 8s2.5-4.5 6.5-4.5S14.5 8 14.5 8s-2.5 4.5-6.5 4.5S1.5 8 1.5 8zM8 6a2 2 0 1 0 0 4 2 2 0 0 0 0-4z' },
  edit: { d: 'M11.5 2.5l2 2-8 8H3.5v-2zM9.5 4.5l2 2' },
  drag: { d: 'M6 4h.01M10 4h.01M6 8h.01M10 8h.01M6 12h.01M10 12h.01', dots: true },
} as const

export type IconName = keyof typeof PATHS

export function Icon({ name, size = 16, style }: { name: IconName; size?: number; style?: CSSProperties }) {
  const p = PATHS[name] as { d: string; fill?: boolean; dots?: boolean }
  return (
    <svg
      width={size} height={size} viewBox="0 0 16 16" aria-hidden="true"
      style={{ flexShrink: 0, display: 'block', ...style }}
      fill={p.fill ? 'currentColor' : 'none'}
      stroke={p.fill ? 'none' : 'currentColor'}
      strokeWidth={p.dots ? 2.4 : 1.5} strokeLinecap="round" strokeLinejoin="round"
    >
      <path d={p.d} />
    </svg>
  )
}

/** The OpenDub mark: two overlapping rings — original and dub. Neutral so it sits in any chrome. */
export function LogoMark({ size = 18 }: { size?: number }) {
  const w = Math.round(size * 26 / 18)
  return (
    <svg width={w} height={size} viewBox="0 0 26 18" aria-hidden="true" style={{ flexShrink: 0 }}>
      <circle cx="9" cy="9" r="7.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="17" cy="9" r="7.5" fill="none" stroke="var(--accent)" strokeWidth="1.6" />
    </svg>
  )
}
