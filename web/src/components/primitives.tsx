// Shared UI primitives. Keep these tiny, consistent, and dependency-free.
import type { CSSProperties, ReactNode } from 'react'
import { useEffect, useRef } from 'react'

export function Button({
  children, onClick, variant = 'default', disabled, title, style, autoFocus,
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'default' | 'primary' | 'ghost' | 'danger'
  disabled?: boolean
  title?: string
  style?: CSSProperties
  autoFocus?: boolean
}) {
  const colors: Record<string, CSSProperties> = {
    default: { background: 'var(--bg-overlay)', border: '1px solid var(--border-strong)', color: 'var(--text)' },
    primary: { background: 'var(--accent)', border: '1px solid var(--accent)', color: '#fff' },
    ghost: { background: 'transparent', border: '1px solid transparent', color: 'var(--text-dim)' },
    danger: { background: 'transparent', border: '1px solid var(--border-strong)', color: 'var(--err)' },
  }
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      autoFocus={autoFocus}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 12px',
        borderRadius: 'var(--r-md)', cursor: disabled ? 'not-allowed' : 'pointer',
        fontSize: 13, fontWeight: 500, transition: 'background var(--ease), border-color var(--ease)',
        opacity: disabled ? 0.5 : 1, whiteSpace: 'nowrap', userSelect: 'none',
        ...colors[variant], ...style,
      }}
    >
      {children}
    </button>
  )
}

export function Select({
  value, onChange, options, disabled, style,
}: {
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string; disabled?: boolean }[]
  disabled?: boolean
  style?: CSSProperties
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      style={{
        background: 'var(--bg-overlay)', border: '1px solid var(--border-strong)',
        borderRadius: 'var(--r-md)', padding: '5px 8px', fontSize: 13, cursor: 'pointer',
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
    <label style={{ display: 'block', marginBottom: 12 }}>
      <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-dim)', marginBottom: 4 }}>{label}</div>
      {children}
      {help && <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 4 }}>{help}</div>}
    </label>
  )
}

export function TextInput({
  value, onChange, type = 'text', placeholder, disabled, onBlur, style,
}: {
  value: string
  onChange: (v: string) => void
  type?: string
  placeholder?: string
  disabled?: boolean
  onBlur?: () => void
  style?: CSSProperties
}) {
  return (
    <input
      type={type}
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      onBlur={onBlur}
      style={{
        background: 'var(--bg)', border: '1px solid var(--border-strong)', borderRadius: 'var(--r-md)',
        padding: '6px 10px', fontSize: 13, width: '100%', ...style,
      }}
    />
  )
}

export function Badge({ children, color = 'var(--text-dim)', bg }: {
  children: ReactNode; color?: string; bg?: string
}) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4, padding: '1px 8px',
      borderRadius: 999, fontSize: 11, fontWeight: 500, color,
      background: bg ?? 'var(--bg-overlay)', border: '1px solid var(--border)',
    }}>
      {children}
    </span>
  )
}

export function ProgressBar({ value, color = 'var(--accent)' }: { value: number; color?: string }) {
  return (
    <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
      <div style={{
        height: '100%', width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%`,
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

export function Modal({ title, onClose, children, width = 440 }: {
  title: string; onClose: () => void; children: ReactNode; width?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
      style={{
        position: 'fixed', inset: 0, background: 'rgb(0 0 0 / 0.6)', zIndex: 100,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div ref={ref} style={{
        width, maxWidth: '92vw', maxHeight: '86vh', overflow: 'auto',
        background: 'var(--bg-raised)', border: '1px solid var(--border-strong)',
        borderRadius: 'var(--r-lg)', boxShadow: '0 8px 32px rgb(0 0 0 / 0.5)',
        animation: 'od-slide-up 150ms ease-out',
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '14px 18px', borderBottom: '1px solid var(--border)',
        }}>
          <div style={{ fontSize: 14, fontWeight: 600 }}>{title}</div>
          <Button variant="ghost" onClick={onClose} title="Close">✕</Button>
        </div>
        <div style={{ padding: 18 }}>{children}</div>
      </div>
    </div>
  )
}
