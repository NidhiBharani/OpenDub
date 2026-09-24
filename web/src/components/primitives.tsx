// Shared UI primitives. Keep these tiny, consistent, and dependency-free.
// Scale (docs/plans/editor-redesign.md §2): controls 24px, buttons 26px, inputs 26px, radii 4px.
import type { CSSProperties, ReactNode } from 'react'
import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Icon, type IconName } from './Icon'

export function Button({
  children, onClick, variant = 'default', disabled, title, style, autoFocus, size = 'md', ariaLabel,
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'default' | 'primary' | 'ok' | 'ghost' | 'danger'
  disabled?: boolean
  title?: string
  style?: CSSProperties
  autoFocus?: boolean
  size?: 'sm' | 'md'
  ariaLabel?: string
}) {
  const colors: Record<string, CSSProperties> = {
    default: { background: 'var(--bg-overlay)', border: '1px solid var(--border-strong)', color: 'var(--text)' },
    primary: { background: 'var(--accent)', border: '1px solid var(--accent)', color: 'var(--accent-ink)', fontWeight: 600 },
    ok: { background: 'var(--ok)', border: '1px solid var(--ok)', color: 'var(--ok-ink)', fontWeight: 600 },
    ghost: { background: 'transparent', border: '1px solid transparent', color: 'var(--text-dim)' },
    danger: { background: 'transparent', border: '1px solid var(--border-strong)', color: 'var(--err)' },
  }
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      autoFocus={autoFocus}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6,
        height: size === 'sm' ? 22 : 26, padding: size === 'sm' ? '0 8px' : '0 10px',
        borderRadius: 'var(--r-md)', cursor: disabled ? 'not-allowed' : 'pointer',
        fontSize: 12, fontWeight: 500, transition: 'background var(--ease), border-color var(--ease)',
        opacity: disabled ? 0.45 : 1, whiteSpace: 'nowrap', userSelect: 'none',
        ...colors[variant], ...style,
      }}
    >
      {children}
    </button>
  )
}

/** 24px icon-only button with a tooltip (`title` should include the shortcut, e.g. "Split (⌘B)"). */
export function IconButton({
  name, title, onClick, active, disabled, size = 24, iconSize = 14, style, danger,
}: {
  name: IconName
  title: string
  onClick?: () => void
  active?: boolean
  disabled?: boolean
  size?: number
  iconSize?: number
  style?: CSSProperties
  danger?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={title}
      aria-pressed={active}
      style={{
        width: size, height: size, borderRadius: 'var(--r-md)', border: '1px solid transparent',
        background: active ? 'var(--accent-dim)' : 'transparent',
        color: danger ? 'var(--err)' : active ? 'var(--accent-text)' : 'var(--text-dim)',
        cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.4 : 1,
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center', padding: 0, flexShrink: 0,
        ...style,
      }}
    >
      <Icon name={name} size={iconSize} />
    </button>
  )
}

/** Mutually exclusive options in one well (FCP-style segmented control). */
export function Segmented<T extends string>({
  value, onChange, options, size = 'md', ariaLabel, style,
}: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[]
  size?: 'sm' | 'md'
  ariaLabel?: string
  style?: CSSProperties
}) {
  const h = size === 'sm' ? 20 : 24
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      style={{
        display: 'inline-flex', padding: 2, borderRadius: 'var(--r-md)', background: 'var(--bg-sunk)',
        border: '1px solid var(--border-subtle)', gap: 1, ...style,
      }}
    >
      {options.map((o) => {
        const on = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            disabled={o.disabled}
            title={o.title}
            aria-pressed={on}
            onClick={() => onChange(o.value)}
            style={{
              height: h - 4, padding: '0 8px', border: 'none', borderRadius: 3, fontSize: size === 'sm' ? 11 : 12,
              fontWeight: on ? 600 : 500, cursor: o.disabled ? 'not-allowed' : 'pointer',
              background: on ? 'var(--bg-overlay)' : 'transparent',
              color: on ? 'var(--text)' : 'var(--text-dim)', opacity: o.disabled ? 0.45 : 1,
              display: 'inline-flex', alignItems: 'center', gap: 5, whiteSpace: 'nowrap',
              boxShadow: on ? '0 1px 2px rgb(0 0 0 / 0.3)' : 'none',
            }}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

export function Select({
  value, onChange, options, disabled, style, ariaLabel,
}: {
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string; disabled?: boolean }[]
  disabled?: boolean
  style?: CSSProperties
  ariaLabel?: string
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      aria-label={ariaLabel}
      style={{
        background: 'var(--bg-field)', border: '1px solid var(--border-strong)',
        borderRadius: 'var(--r-md)', padding: '0 6px', height: 26, fontSize: 12, cursor: 'pointer',
        width: '100%', ...style,
      }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>
      ))}
    </select>
  )
}

export function Field({ label, help, children }: { label: string; help?: string; children: ReactNode }) {
  return (
    <label style={{ display: 'block', marginBottom: 10 }}>
      <div className="eyebrow" style={{ marginBottom: 4 }}>{label}</div>
      {children}
      {help && <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>{help}</div>}
    </label>
  )
}

/** Inspector-style property row: 11px label in a fixed left column, value on the right. */
export function PropertyRow({ label, children, labelWidth = 84, align = 'center' }: {
  label: string; children: ReactNode; labelWidth?: number; align?: 'center' | 'start'
}) {
  return (
    <div style={{ display: 'flex', alignItems: align, gap: 8, minHeight: 26 }}>
      <div style={{
        width: labelWidth, flexShrink: 0, fontSize: 11, color: 'var(--text-dim)', textAlign: 'right',
        paddingTop: align === 'start' ? 6 : 0,
      }}>
        {label}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>{children}</div>
    </div>
  )
}

export function TextInput({
  value, onChange, type = 'text', placeholder, disabled, onBlur, onKeyDown, style, ariaLabel, autoFocus,
}: {
  value: string
  onChange: (v: string) => void
  type?: string
  placeholder?: string
  disabled?: boolean
  onBlur?: () => void
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void
  style?: CSSProperties
  ariaLabel?: string
  autoFocus?: boolean
}) {
  return (
    <input
      type={type}
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      aria-label={ariaLabel}
      autoFocus={autoFocus}
      onChange={(e) => onChange(e.target.value)}
      onBlur={onBlur}
      onKeyDown={onKeyDown}
      style={{
        background: 'var(--bg-field)', border: '1px solid var(--border-strong)', borderRadius: 'var(--r-md)',
        padding: '0 8px', height: 26, fontSize: 12, width: '100%', ...style,
      }}
    />
  )
}

export function Badge({ children, color = 'var(--text-dim)', bg, title }: {
  children: ReactNode; color?: string; bg?: string; title?: string
}) {
  return (
    <span title={title} style={{
      display: 'inline-flex', alignItems: 'center', gap: 4, padding: '0 6px', height: 16,
      borderRadius: 3, fontSize: 10, fontWeight: 600, letterSpacing: '0.02em', color,
      background: bg ?? 'var(--bg-overlay)', whiteSpace: 'nowrap',
    }}>
      {children}
    </span>
  )
}

/** Speaker chip: name on a tint of the speaker colour. */
export function SpeakerChip({ name, color, size = 'md' }: { name: string; color: string; size?: 'sm' | 'md' }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5, padding: size === 'sm' ? '0 5px' : '0 7px',
      height: size === 'sm' ? 15 : 18, borderRadius: 3, fontSize: size === 'sm' ? 10 : 11, fontWeight: 600,
      color, background: `color-mix(in srgb, ${color} 18%, transparent)`, whiteSpace: 'nowrap', maxWidth: '100%',
      overflow: 'hidden', textOverflow: 'ellipsis',
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, flexShrink: 0 }} />
      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{name}</span>
    </span>
  )
}

export function ProgressBar({ value, color = 'var(--accent)', height = 3 }: { value: number; color?: string; height?: number }) {
  return (
    <div style={{ height, background: 'var(--border-subtle)', borderRadius: 2, overflow: 'hidden' }}>
      <div style={{
        height: '100%', borderRadius: 2, width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%`,
        background: color, transition: 'width 300ms ease-out',
      }} />
    </div>
  )
}

export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <div style={{
      width: size, height: size, borderRadius: '50%',
      border: '2px solid var(--border-strong)', borderTopColor: 'var(--accent)',
      animation: 'od-spin 700ms linear infinite',
    }} />
  )
}

/** Small status dot used by stage steps, script rows and version items. */
export function StatusDot({ color, pulse, size = 6, hollow }: { color: string; pulse?: boolean; size?: number; hollow?: boolean }) {
  return (
    <span style={{
      width: size, height: size, borderRadius: '50%', flexShrink: 0, display: 'inline-block',
      background: hollow ? 'transparent' : color, border: hollow ? `1.5px solid ${color}` : 'none',
      animation: pulse ? 'od-pulse 1000ms ease-in-out infinite' : undefined,
    }} />
  )
}

export function Modal({ title, onClose, children, width = 440 }: {
  title: string; onClose: () => void; children: ReactNode; width?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  // Portal to <body>: a transformed ancestor (e.g. a hovered project card) becomes the containing
  // block for `position: fixed` and would trap the overlay inside it. React still bubbles portal
  // events up the component tree, so stop clicks here too — or Cancel also clicks that card.
  return createPortal(
    <div
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'fixed', inset: 0, background: 'rgb(0 0 0 / 0.55)', zIndex: 100,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div ref={ref} role="dialog" aria-label={title} style={{
        width, maxWidth: '92vw', maxHeight: '86vh', overflow: 'auto',
        background: 'var(--bg-panel)', border: '1px solid var(--border)',
        borderRadius: 'var(--r-lg)', boxShadow: 'var(--shadow)',
        animation: 'od-slide-up 120ms ease-out',
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '10px 12px 10px 16px', borderBottom: '1px solid var(--border)', background: 'var(--bg-raised)',
        }}>
          <div style={{ fontSize: 13, fontWeight: 600 }}>{title}</div>
          <IconButton name="close" title="Close (Esc)" onClick={onClose} />
        </div>
        <div style={{ padding: 16 }}>{children}</div>
      </div>
    </div>,
    document.body,
  )
}

/** Anchored popover (absolute, below its trigger). Closes on outside click / Escape. */
export function Popover({ open, onClose, children, align = 'left', width = 260 }: {
  open: boolean; onClose: () => void; children: ReactNode; align?: 'left' | 'right'; width?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    // defer so the click that opened us does not immediately close us
    const id = setTimeout(() => {
      window.addEventListener('mousedown', onDown)
      window.addEventListener('keydown', onKey)
    }, 0)
    return () => {
      clearTimeout(id)
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [open, onClose])
  if (!open) return null
  return (
    <div ref={ref} role="dialog" style={{
      position: 'absolute', top: 'calc(100% + 4px)', [align]: 0, width, zIndex: 50,
      background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)',
      boxShadow: 'var(--shadow)', padding: 8, animation: 'od-slide-up 100ms ease-out',
    }}>
      {children}
    </div>
  )
}

/** A row in a menu-like popover. */
export function MenuItem({ icon, label, shortcut, onClick, danger, disabled }: {
  icon?: IconName; label: string; shortcut?: string; onClick: () => void; danger?: boolean; disabled?: boolean
}) {
  const [hover, setHover] = useState(false)
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: '100%', height: 26, display: 'flex', alignItems: 'center', gap: 8, padding: '0 8px',
        border: 'none', borderRadius: 3, background: hover && !disabled ? 'var(--bg-overlay)' : 'transparent',
        color: danger ? 'var(--err)' : 'var(--text)', cursor: disabled ? 'not-allowed' : 'pointer', fontSize: 12,
        textAlign: 'left', opacity: disabled ? 0.45 : 1,
      }}
    >
      {icon && <Icon name={icon} size={13} style={{ color: danger ? 'var(--err)' : 'var(--text-dim)' }} />}
      <span style={{ flex: 1 }}>{label}</span>
      {shortcut && <span className="timecode" style={{ fontSize: 10 }}>{shortcut}</span>}
    </button>
  )
}

/** Hover/keyboard tooltip-free label helper for keyboard shortcuts in titles. */
export const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)
export const MOD = isMac ? '⌘' : 'Ctrl+'
