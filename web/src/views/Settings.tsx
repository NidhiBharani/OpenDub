// Settings view: a dense preferences sheet (docs/plans/editor-redesign.md §8).
//
// Simple mode asks four questions (runtime, quality preset, lip sync, subtitles) and lets the
// server write concrete provider choices for all 41 capabilities. Advanced mode opens the whole
// capability map: enable/disable, provider choice, capability params and per-provider config.
import type { CSSProperties, ReactNode } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import { Icon } from '../components/Icon'
import { Badge, Button, IconButton, Segmented, Select, Spinner, StatusDot, TextInput } from '../components/primitives'
import { NavLink, TopBar } from '../components/TopBar'
import { useStore } from '../state/store'
import { SECRET_MASK, TIER_LABELS, capabilityBuiltin } from '../types'
import type {
  Capability, ConfigField, LegacyKind, Phase, PipelineMode, PresetName, Project, ProviderInfo,
  ResolvedCapability, RuntimeName, Tier,
} from '../types'

const RUNTIME_OPTIONS: { value: RuntimeName; label: string; help: string }[] = [
  { value: 'builtin', label: 'Built-in', help: 'Pure ffmpeg/Python fallbacks. Runs anywhere, no models to install.' },
  { value: 'local', label: 'My GPU', help: 'Prefer self-hosted models on this machine.' },
  { value: 'cloud', label: 'Cloud keys', help: 'Prefer hosted APIs, using the keys saved on each provider.' },
]

const PRESET_LABELS: Record<PresetName, string> = {
  minimal: 'Minimal', balanced: 'Balanced', max: 'Max',
}
const PRESET_ORDER: PresetName[] = ['minimal', 'balanced', 'max']

const TIER_COLORS: Record<Tier, { color: string; bg: string }> = {
  core: { color: 'var(--accent-text)', bg: 'var(--accent-dim)' },
  recommended: { color: 'var(--ok)', bg: 'var(--ok-dim)' },
  advanced: { color: 'var(--text-dim)', bg: 'var(--bg-overlay)' },
  experimental: { color: 'var(--warn-text)', bg: 'var(--warn-dim)' },
  deferred: { color: 'var(--text-faint)', bg: 'transparent' },
}

const BUILT_IN = /(mock|passthrough|none|single_speaker|off)$/
/** A provider that means "this capability does not run" — nothing to configure. */
const OFF_PROVIDER = /\.(off|none)$/

/** Draft values keyed by ConfigField.key. Booleans stay booleans; everything
 *  else (string/number/select/secret) is edited as a string and converted on save. */
type Draft = Record<string, string | boolean>

function fieldDraft(fields: ConfigField[], values: Record<string, unknown>, mask = false): Draft {
  const draft: Draft = {}
  for (const field of fields) {
    const raw = values[field.key]
    if (field.type === 'boolean') {
      draft[field.key] = raw !== undefined && raw !== null ? Boolean(raw) : Boolean(field.default)
    } else if (mask && field.type === 'secret') {
      // Never surface the mask (or any prior secret) as an editable value — only the
      // "configured" hint communicates that something is set. Leave blank = unchanged.
      draft[field.key] = raw === SECRET_MASK ? '' : (raw != null ? String(raw) : '')
    } else if (raw !== undefined && raw !== null) {
      draft[field.key] = String(raw)
    } else {
      draft[field.key] = field.default != null ? String(field.default) : ''
    }
  }
  return draft
}

const initDraft = (info: ProviderInfo): Draft => fieldDraft(info.meta.fields, info.configured_options, true)

function buildPayload(fields: ConfigField[], draft: Draft): Record<string, unknown> {
  const payload: Record<string, unknown> = {}
  for (const field of fields) {
    const value = draft[field.key]
    if (field.type === 'boolean') {
      payload[field.key] = Boolean(value)
      continue
    }
    const str = typeof value === 'string' ? value : String(value ?? '')
    if (field.type === 'secret') {
      // A blank secret means "leave unchanged" (the saved key is never echoed back
      // into the form), so it must be omitted rather than sent as a clear.
      if (str !== '') payload[field.key] = str
      continue
    }
    if (str === '') {
      payload[field.key] = '' // explicit empty — the server deletes the saved key
      continue
    }
    if (field.type === 'number') {
      const n = parseFloat(str)
      if (!Number.isNaN(n)) payload[field.key] = n
      continue
    }
    payload[field.key] = str
  }
  return payload
}

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text
}

const errText = (e: unknown, fallback: string): string => (e instanceof Error ? e.message : fallback)

// ---- local layout helpers --------------------------------------------------------

/** 11px uppercase section label (replaces the serif headings). */
function SectionLabel({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div className="eyebrow" style={{ height: 24, display: 'flex', alignItems: 'center', ...style }}>{children}</div>
}

/** Card: `--bg-panel`, 1px border, 4px radius. */
function Panel({ children, style, ...rest }: { children: ReactNode; style?: CSSProperties; 'aria-label'?: string }) {
  return (
    <section
      aria-label={rest['aria-label']}
      style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--r-md)', ...style }}
    >
      {children}
    </section>
  )
}

/** Dense form row: 11px label in a fixed column, control on the right, 11px help under the control. */
function DenseField({ label, help, children, labelWidth = 120 }: {
  label: string; help?: string; children: ReactNode; labelWidth?: number
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'flex-start', gap: 8, padding: '3px 0' }}>
      <span style={{ width: labelWidth, flexShrink: 0, fontSize: 11, color: 'var(--text-dim)', textAlign: 'right', lineHeight: '26px' }}>
        {label}
      </span>
      <span style={{ flex: 1, minWidth: 0 }}>
        {children}
        {help && <span style={{ display: 'block', fontSize: 11, color: 'var(--text-faint)', marginTop: 2 }}>{help}</span>}
      </span>
    </label>
  )
}

const CHECKBOX_STYLE: CSSProperties = { width: 13, height: 13, margin: 0, accentColor: 'var(--accent)', cursor: 'inherit', flexShrink: 0 }

/** Disclosure chevron (24px hit area). */
function Chevron({ open, onClick, title }: { open: boolean; onClick: () => void; title: string }) {
  return (
    <IconButton
      name="chevron"
      title={title}
      onClick={onClick}
      size={22}
      iconSize={12}
      style={{ transform: open ? 'none' : 'rotate(-90deg)', transition: 'transform var(--ease)' }}
    />
  )
}

/** One config field, rendered the same way for provider options and capability params. */
function ConfigFieldInput({ field, value, onChange, onBlur, secretConfigured }: {
  field: ConfigField
  value: string | boolean | undefined
  onChange: (v: string | boolean) => void
  onBlur?: () => void
  secretConfigured?: boolean
}) {
  if (field.type === 'boolean') {
    // Checkbox and its caption sit in the control column, like a macOS preferences sheet.
    return (
      <DenseField label="" help={field.help}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6, minHeight: 26, fontSize: 12, cursor: 'pointer' }}>
          <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} style={CHECKBOX_STYLE} />
          <span>{field.label}</span>
        </span>
      </DenseField>
    )
  }

  if (field.type === 'select') {
    return (
      <DenseField label={field.label} help={field.help}>
        <Select value={String(value ?? '')} onChange={onChange} options={field.options.map((o) => ({ value: o, label: o }))} style={{ maxWidth: 320 }} />
      </DenseField>
    )
  }

  if (field.type === 'secret') {
    return (
      <DenseField label={field.label} help={secretConfigured ? 'Configured — write-only, leave blank to keep the saved key.' : field.help}>
        <TextInput
          type="password"
          value={String(value ?? '')}
          onChange={onChange}
          onBlur={onBlur}
          placeholder={secretConfigured ? SECRET_MASK : field.placeholder}
          style={{ maxWidth: 420 }}
        />
      </DenseField>
    )
  }

  // string | number
  return (
    <DenseField label={field.label} help={field.help}>
      <TextInput
        type={field.type === 'number' ? 'number' : 'text'}
        value={String(value ?? '')}
        onChange={onChange}
        onBlur={onBlur}
        placeholder={field.placeholder}
        style={{ maxWidth: 320 }}
      />
    </DenseField>
  )
}

export function Settings() {
  const project = useStore((s) => s.project)
  const loadProviders = useStore((s) => s.loadProviders)
  const loadCapabilityMap = useStore((s) => s.loadCapabilityMap)
  const loadResolvedCapabilities = useStore((s) => s.loadResolvedCapabilities)
  const setPipelineMode = useStore((s) => s.setPipelineMode)
  const setView = useStore((s) => s.setView)
  const closeProject = useStore((s) => s.closeProject)
  const toast = useStore((s) => s.toast)

  const [mode, setMode] = useState<PipelineMode>(project?.pipeline.mode ?? 'simple')

  useEffect(() => {
    void loadProviders()
    void loadCapabilityMap()
    // This is the one view that shows the resolver's verdict, so never show a stale one.
    void loadResolvedCapabilities()
  }, [loadProviders, loadCapabilityMap, loadResolvedCapabilities])

  const savedMode = project?.pipeline.mode
  useEffect(() => {
    if (savedMode) setMode(savedMode)
  }, [savedMode])

  function chooseMode(next: PipelineMode) {
    setMode(next)
    void setPipelineMode(next).catch((e: unknown) => toast('error', errText(e, 'Failed to save the settings mode')))
  }

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <TopBar onBrand={() => (project ? closeProject() : setView('library'))}>
        <NavLink label="Library" onClick={() => (project ? closeProject() : setView('library'))} />
        {project && <NavLink label="Editor" onClick={() => setView('editor')} />}
        <NavLink label="Settings" active onClick={() => setView('settings')} />
        <div style={{ flex: 1 }} />
      </TopBar>

      <div style={{
        height: 36, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10, padding: '0 12px',
        background: 'var(--bg-raised)', borderBottom: '1px solid var(--border)',
      }}>
        <span style={{ fontSize: 13, fontWeight: 600, whiteSpace: 'nowrap' }}>Pipeline settings</span>
        <span style={{ fontSize: 11, color: 'var(--text-dim)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', minWidth: 0 }}>
          {project ? <>applies to <span style={{ color: 'var(--text)' }}>{project.name}</span></> : 'open a project to configure its pipeline'}
        </span>
        <div style={{ flex: 1 }} />
        <Segmented<PipelineMode>
          value={mode}
          onChange={chooseMode}
          ariaLabel="Settings mode"
          options={[{ value: 'simple', label: 'Simple' }, { value: 'advanced', label: 'Advanced' }]}
        />
      </div>

      {mode === 'simple' ? <SimpleMode project={project} /> : <AdvancedMode project={project} />}
    </div>
  )
}

// ---- simple mode ---------------------------------------------------------------

function SimpleMode({ project }: { project: Project | null }) {
  const capabilityMap = useStore((s) => s.capabilityMap)
  const applyPreset = useStore((s) => s.applyPreset)
  const toast = useStore((s) => s.toast)

  const pipeline = project?.pipeline
  const [runtime, setRuntime] = useState<RuntimeName>('builtin')
  const [preset, setPreset] = useState<PresetName>('minimal')
  const [lipsync, setLipsync] = useState(false)
  const [subtitles, setSubtitles] = useState(false)
  const [applying, setApplying] = useState(false)

  // Re-seed from the project whenever its saved configuration changes ("custom" has no control
  // of its own — it keeps the last real preset/runtime showing as the starting point).
  useEffect(() => {
    if (!pipeline) return
    if (pipeline.runtime !== 'custom') setRuntime(pipeline.runtime)
    if (pipeline.preset !== 'custom') setPreset(pipeline.preset)
    setLipsync(pipeline.lipsync_enabled)
    setSubtitles(pipeline.subtitles_enabled)
  }, [pipeline?.runtime, pipeline?.preset, pipeline?.lipsync_enabled, pipeline?.subtitles_enabled]) // eslint-disable-line react-hooks/exhaustive-deps

  const total = capabilityMap?.capabilities.length ?? 41
  const customised = pipeline?.preset === 'custom'
  const disabled = !project

  async function apply() {
    setApplying(true)
    try {
      await applyPreset({ preset, runtime, lipsync, subtitles })
      toast('success', `${PRESET_LABELS[preset]} preset applied`)
    } catch (e) {
      toast('error', errText(e, 'Failed to apply the preset'))
    } finally {
      setApplying(false)
    }
  }

  return (
    <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 420px) minmax(360px, 1fr)', gap: 16, alignItems: 'start', maxWidth: 1180 }}>
        <Panel aria-label="Pipeline preset" style={{ padding: '4px 12px 12px' }}>
          <SectionLabel>Runtime</SectionLabel>
          <DenseField label="Models run on" help={RUNTIME_OPTIONS.find((r) => r.value === runtime)?.help} labelWidth={96}>
            <Select
              value={runtime}
              onChange={(v) => setRuntime(v as RuntimeName)}
              disabled={disabled}
              ariaLabel="Where the models run"
              options={RUNTIME_OPTIONS.map((r) => ({ value: r.value, label: r.label }))}
            />
          </DenseField>

          <SectionLabel style={{ marginTop: 8 }}>Preset</SectionLabel>
          <div role="radiogroup" aria-label="Quality preset" style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--r-md)', overflow: 'hidden' }}>
            {PRESET_ORDER.map((id, i) => {
              const info = capabilityMap?.presets.find((p) => p.id === id)
              const active = id === preset
              return (
                <button
                  key={id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  disabled={disabled}
                  onClick={() => setPreset(id)}
                  style={{
                    width: '100%', textAlign: 'left', padding: '5px 10px', border: 'none', color: 'var(--text)',
                    borderTop: i === 0 ? 'none' : '1px solid var(--border-subtle)',
                    cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.6 : 1,
                    background: active ? 'var(--accent-dim)' : 'transparent',
                    display: 'flex', flexDirection: 'column', gap: 1,
                  }}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: 8, height: 18 }}>
                    <StatusDot color={active ? 'var(--accent)' : 'var(--text-faint)'} hollow={!active} size={8} />
                    <span style={{ fontSize: 12, fontWeight: 600 }}>{PRESET_LABELS[id]}</span>
                    <span style={{ flex: 1 }} />
                    <span className="timecode" style={{ fontSize: 10 }}>
                      {info ? `${info.capabilities.length} of ${total}` : `— of ${total}`} capabilities
                    </span>
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-dim)', paddingLeft: 16 }}>{info?.description ?? ''}</span>
                </button>
              )
            })}
          </div>

          <SectionLabel style={{ marginTop: 8 }}>Features</SectionLabel>
          <ToggleRow
            label="Lip sync"
            help="Re-times mouth movement in the picture. Slow, and needs a lip-sync model."
            checked={lipsync}
            disabled={disabled}
            onChange={setLipsync}
          />
          <ToggleRow
            label="Subtitles"
            help="Writes reading-speed-compliant subtitles from the same translation."
            checked={subtitles}
            disabled={disabled}
            onChange={setSubtitles}
          />

          {customised && (
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginTop: 10, fontSize: 11, color: 'var(--warn-text)' }}>
              <Icon name="warn" size={12} style={{ flexShrink: 0, marginTop: 1 }} />
              <span>Customised in Advanced — this project no longer follows a preset. Applying one resets every capability.</span>
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 12 }}>
            <Button variant="primary" onClick={() => void apply()} disabled={disabled || applying}>
              {applying ? <Spinner size={12} /> : <Icon name="check" size={12} />}
              {applying ? 'Applying…' : 'Apply'}
            </Button>
          </div>
        </Panel>

        <RunSummary project={project} />
      </div>
    </div>
  )
}

function ToggleRow({ label, help, checked, disabled, onChange }: {
  label: string; help: string; checked: boolean; disabled?: boolean; onChange: (v: boolean) => void
}) {
  return (
    <label style={{
      display: 'flex', alignItems: 'flex-start', gap: 8, padding: '4px 2px',
      cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.6 : 1,
    }}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} style={{ ...CHECKBOX_STYLE, marginTop: 2 }} />
      <span style={{ display: 'flex', flexDirection: 'column', gap: 1, minWidth: 0 }}>
        <span style={{ fontSize: 12, fontWeight: 500 }}>{label}</span>
        <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{help}</span>
      </span>
    </label>
  )
}

const SUMMARY_ROW: CSSProperties = { height: 24, display: 'flex', alignItems: 'center', gap: 8, padding: '0 10px', fontSize: 12 }

/** Read-only "this is what will run", grouped by phase — straight from the resolver. */
function RunSummary({ project }: { project: Project | null }) {
  const capabilityMap = useStore((s) => s.capabilityMap)
  const resolved = useStore((s) => s.resolvedCapabilities)
  const providers = useStore((s) => s.providers)

  const byPhase = useMemo(() => groupByPhase(capabilityMap?.capabilities ?? []), [capabilityMap])
  const on = Object.values(resolved).filter((r) => r.enabled).length

  if (!project) {
    return (
      <section aria-label="What will run" style={{ fontSize: 12, color: 'var(--text-dim)', padding: '4px 0' }}>
        Open a project from the library to see what its pipeline will run.
      </section>
    )
  }

  return (
    <Panel aria-label="What will run" style={{ overflow: 'hidden' }}>
      <div className="panel-header" style={{ padding: '0 10px' }}>
        <span className="eyebrow">What will run</span>
        <span style={{ flex: 1 }} />
        <span className="timecode" style={{ fontSize: 11 }}>
          {on} of {capabilityMap?.capabilities.length ?? Object.keys(resolved).length} on
        </span>
      </div>
      {byPhase.length === 0 && (
        <div style={{ ...SUMMARY_ROW, color: 'var(--text-dim)' }}>The capability map is not available.</div>
      )}
      {byPhase.map(([phase, caps]) => (
        <div key={phase}>
          <div style={{ ...SUMMARY_ROW, background: 'var(--bg-raised)', borderTop: '1px solid var(--border-subtle)', borderBottom: '1px solid var(--border-subtle)' }}>
            <span className="timecode" style={{ fontSize: 11, color: 'var(--accent-text)', width: 22 }}>{phase}</span>
            <span style={{ fontSize: 11, fontWeight: 600 }}>{capabilityMap?.phases[phase] ?? ''}</span>
          </div>
          {caps.map((cap) => {
            const r = resolved[cap.id]
            const enabled = r?.enabled ?? false
            const provider = providers.find((p) => p.meta.id === r?.provider_id)
            return (
              <div key={cap.id} style={{ ...SUMMARY_ROW, opacity: enabled ? 1 : 0.6, borderBottom: '1px solid var(--border-subtle)' }}>
                <StatusDot color={enabled ? 'var(--ok)' : 'var(--text-faint)'} hollow={!enabled} />
                <span className="timecode" style={{ fontSize: 11, width: 22, flexShrink: 0 }}>{cap.id}</span>
                <span style={{ minWidth: 0, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {cap.name}
                </span>
                <span style={{ fontSize: 11, color: 'var(--text-dim)', flexShrink: 0, maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {enabled ? (provider?.meta.name ?? r?.provider_id ?? '—') : (r?.reason || 'off')}
                </span>
              </div>
            )
          })}
        </div>
      ))}
    </Panel>
  )
}

function groupByPhase(caps: Capability[]): [Phase, Capability[]][] {
  const out = new Map<Phase, Capability[]>()
  for (const cap of caps) out.set(cap.phase, [...(out.get(cap.phase) ?? []), cap])
  return [...out.entries()].sort((a, b) => a[0].localeCompare(b[0]))
}

// ---- advanced mode -------------------------------------------------------------

function AdvancedMode({ project }: { project: Project | null }) {
  const capabilityMap = useStore((s) => s.capabilityMap)
  const resolved = useStore((s) => s.resolvedCapabilities)

  const byPhase = useMemo(() => groupByPhase(capabilityMap?.capabilities ?? []), [capabilityMap])
  const [phase, setPhase] = useState<Phase>('A')
  const [jump, setJump] = useState<{ id: string; nonce: number } | null>(null)
  const cardRefs = useRef(new Map<string, HTMLElement>())

  useEffect(() => {
    if (!jump) return
    const el = cardRefs.current.get(jump.id)
    el?.scrollIntoView({ block: 'center', behavior: 'smooth' })
    el?.focus()
  }, [jump, phase])

  function jumpTo(capId: string) {
    const target = capabilityMap?.capabilities.find((c) => c.id === capId)
    if (!target) return
    setPhase(target.phase)
    setJump({ id: capId, nonce: Date.now() })
  }

  const caps = byPhase.find(([p]) => p === phase)?.[1] ?? []

  if (byPhase.length === 0) {
    return (
      <div style={{ flex: 1, padding: 16, fontSize: 12, color: 'var(--text-dim)' }}>
        The capability map is not available — the server may be older than this build.
      </div>
    )
  }

  const onCount = caps.filter((c) => resolved[c.id]?.enabled).length

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
      <nav aria-label="Capability phases" style={{
        width: 240, flexShrink: 0, borderRight: '1px solid var(--border)', background: 'var(--bg-panel)',
        overflowY: 'auto', display: 'flex', flexDirection: 'column',
      }}>
        <div className="panel-header"><span className="eyebrow">Phases</span></div>
        {byPhase.map(([key, phaseCaps]) => {
          const active = key === phase
          const on = phaseCaps.filter((c) => resolved[c.id]?.enabled).length
          return (
            <button
              key={key}
              type="button"
              aria-current={active ? 'true' : undefined}
              onClick={() => setPhase(key)}
              style={{
                height: 28, padding: '0 10px', border: 'none', cursor: 'pointer', textAlign: 'left',
                display: 'flex', alignItems: 'center', gap: 8, color: active ? 'var(--text)' : 'var(--text-dim)',
                background: active ? 'var(--accent-dim)' : 'transparent', fontSize: 12, fontWeight: active ? 600 : 500,
                boxShadow: active ? 'inset 2px 0 0 var(--accent)' : 'none',
              }}
            >
              <span className="timecode" style={{ fontSize: 11, width: 14, color: active ? 'var(--accent-text)' : 'var(--text-faint)' }}>
                {key}
              </span>
              <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {capabilityMap?.phases[key] ?? key}
              </span>
              <span className="timecode" style={{ fontSize: 10, flexShrink: 0 }}>
                {on}/{phaseCaps.length}
              </span>
            </button>
          )
        })}
        <div style={{ flex: 1 }} />
        <p style={{ margin: 0, padding: '10px 10px 12px', fontSize: 11, color: 'var(--text-faint)', borderTop: '1px solid var(--border-subtle)' }}>
          Provider keys and options are saved to{' '}
          <span style={{ fontFamily: 'var(--mono)' }}>configs/settings.yaml</span> and shared by every project.
          Everything else here belongs to this project.
        </p>
      </nav>

      <div style={{ flex: 1, minWidth: 0, overflowY: 'auto', padding: 16 }}>
        <div style={{ maxWidth: 880 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, height: 24, marginBottom: 8 }}>
            <span className="eyebrow">Phase {phase} · {capabilityMap?.phases[phase] ?? phase}</span>
            <span style={{ flex: 1 }} />
            <span className="timecode" style={{ fontSize: 11 }}>{onCount} of {caps.length} on</span>
          </div>
          {caps.map((cap) => (
            <CapabilityCard
              key={cap.id}
              cap={cap}
              project={project}
              resolved={resolved[cap.id]}
              onJump={jumpTo}
              registerRef={(el) => {
                if (el) cardRefs.current.set(cap.id, el)
                else cardRefs.current.delete(cap.id)
              }}
              highlighted={jump?.id === cap.id}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function CapabilityCard({ cap, project, resolved, onJump, registerRef, highlighted }: {
  cap: Capability
  project: Project | null
  resolved: ResolvedCapability | undefined
  onJump: (capId: string) => void
  registerRef: (el: HTMLElement | null) => void
  highlighted: boolean
}) {
  const providers = useStore((s) => s.providers)
  const setCapability = useStore((s) => s.setCapability)
  const setLegacyProvider = useStore((s) => s.setLegacyProvider)
  const toast = useStore((s) => s.toast)

  const deferred = cap.tier === 'deferred'
  const isCore = cap.tier === 'core'
  // Cards open by default so the whole phase is scannable; deferred ones have nothing to show.
  const [open, setOpen] = useState(!deferred)
  const [configuring, setConfiguring] = useState(false)
  const [busy, setBusy] = useState(false)

  // F1 has no separate switch: choosing "lipsync.none" is how lip sync is turned off, and the
  // server derives pipeline.lipsync_enabled from exactly that.
  const providerDrivesToggle = cap.id === 'F1'
  const enabled = resolved?.enabled ?? false
  const providerId = resolved?.provider_id ?? capabilityBuiltin(cap)
  const selected = providers.find((p) => p.meta.id === providerId)
  const tint = TIER_COLORS[cap.tier]

  const options = useMemo(() => {
    const list = providers
      .filter((p) => p.meta.kind === cap.kind)
      .sort((a, b) => {
        if (a.meta.runtime !== b.meta.runtime) return a.meta.runtime === 'local' ? -1 : 1
        return a.meta.name.localeCompare(b.meta.name)
      })
      .map((p) => ({
        value: p.meta.id,
        label: `${p.meta.name} · ${p.available ? 'ready' : truncate(p.reason || 'unavailable', 34)}`,
      }))
    // The saved choice may not be registered on this server — keep it visible rather than
    // silently showing (and then saving) a different provider.
    if (providerId && !list.some((o) => o.value === providerId)) {
      list.unshift({ value: providerId, label: `${providerId} · not registered` })
    }
    return list
  }, [providers, cap.kind, providerId])

  // Only "<kind>.off" is registered: the capability is in the map but nothing implements it here
  // yet, so switching it on would resolve straight back to off ("no provider selected").
  const noProvider = options.length > 0 && options.every((o) => OFF_PROVIDER.test(o.value))

  async function run(action: () => Promise<void>, failure: string) {
    setBusy(true)
    try {
      await action()
    } catch (e) {
      toast('error', errText(e, failure))
    } finally {
      setBusy(false)
    }
  }

  const chooseProvider = (id: string) => void run(async () => {
    if (cap.legacy_field) await setLegacyProvider(cap.legacy_field as LegacyKind, id)
    else await setCapability(cap.id, { provider_id: id, enabled: !id.endsWith('.off') })
  }, `Failed to set the ${cap.name} provider`)

  const toggle = (on: boolean) => void run(
    () => setCapability(cap.id, { enabled: on }),
    `Failed to switch ${cap.id} ${on ? 'on' : 'off'}`,
  )

  const saveParams = (params: Record<string, unknown>) =>
    run(() => setCapability(cap.id, { params }), `Failed to save ${cap.id} settings`)

  const toggleDisabled = isCore || noProvider || !project || busy
  const offProvider = OFF_PROVIDER.test(providerId)

  return (
    <article
      ref={registerRef}
      tabIndex={-1}
      aria-label={`${cap.id} ${cap.name}`}
      style={{
        background: 'var(--bg-panel)', borderRadius: 'var(--r-md)', marginBottom: 8, opacity: deferred ? 0.6 : 1,
        border: `1px solid ${highlighted ? 'var(--accent)' : 'var(--border)'}`, transition: 'border-color var(--ease)',
        overflow: 'hidden',
      }}
    >
      <div style={{
        height: 28, display: 'flex', alignItems: 'center', gap: 8, padding: '0 8px 0 4px',
        background: 'var(--bg-raised)', borderBottom: open ? '1px solid var(--border-subtle)' : 'none',
      }}>
        <Chevron open={open} onClick={() => setOpen((o) => !o)} title={open ? `Collapse ${cap.id}` : `Expand ${cap.id}`} />
        <span className="timecode" style={{ fontSize: 11, color: 'var(--accent-text)', width: 22 }}>{cap.id}</span>
        <span style={{ fontSize: 12, fontWeight: 600, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{cap.name}</span>
        <div style={{ flex: 1 }} />
        {cap.requires && (
          <Badge>{cap.requires === 'lipsync' ? 'needs lip sync on' : 'needs subtitles on'}</Badge>
        )}
        <Badge color={tint.color} bg={tint.bg}>{TIER_LABELS[cap.tier]}</Badge>
        {!deferred && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 11, width: 30, color: enabled ? 'var(--text)' : 'var(--text-faint)' }}>
            <StatusDot color={enabled ? 'var(--ok)' : 'var(--text-faint)'} hollow={!enabled} />
            {enabled ? 'on' : 'off'}
          </span>
        )}
      </div>

      {open && (
        <div style={{ padding: '6px 10px 8px' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>{cap.summary}</div>

          {deferred ? (
            <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>Not built yet — it appears here so the map is complete.</div>
          ) : (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', minHeight: 26 }}>
                {!providerDrivesToggle && (
                  <label
                    title={isCore ? 'Core capability — always on' : noProvider ? 'Nothing implements this capability on this server yet' : undefined}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, flexShrink: 0, whiteSpace: 'nowrap', minWidth: 96,
                      cursor: toggleDisabled ? 'not-allowed' : 'pointer', color: isCore || noProvider ? 'var(--text-dim)' : 'var(--text)',
                    }}
                  >
                    <input type="checkbox" checked={enabled} disabled={toggleDisabled} onChange={(e) => toggle(e.target.checked)} style={CHECKBOX_STYLE} />
                    {isCore ? 'Always on' : 'Enabled'}
                    {isCore && <Badge>locked</Badge>}
                    {!isCore && noProvider && <Badge>no provider</Badge>}
                  </label>
                )}
                <div style={{ flex: 1, minWidth: 220, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, color: 'var(--text-dim)', flexShrink: 0 }}>Provider</span>
                  <Select
                    value={providerId}
                    onChange={chooseProvider}
                    ariaLabel="Provider"
                    disabled={!project || busy || options.length === 0}
                    options={options.length > 0 ? options : [{ value: '', label: 'No provider registered' }]}
                    style={{ flex: 1, width: 'auto', maxWidth: 360 }}
                  />
                  {!offProvider && (
                    <Button size="sm" variant="ghost" onClick={() => setConfiguring((c) => !c)} title={`Configure ${selected?.meta.name ?? providerId}`}>
                      <Icon name="settings" size={12} />
                      Configure
                    </Button>
                  )}
                </div>
              </div>

              {providerDrivesToggle && (
                <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 2 }}>“None” turns lip sync off for this project.</div>
              )}

              {resolved?.reason && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4, fontSize: 11, color: enabled ? 'var(--text-dim)' : 'var(--warn-text)' }}>
                  <Icon name="info" size={11} style={{ flexShrink: 0 }} />
                  {resolved.reason}
                </div>
              )}

              {cap.params.length > 0 && (
                <CapabilityParams cap={cap} values={resolved?.params ?? {}} disabled={!project || busy} onSave={saveParams} />
              )}

              {(cap.needs.length > 0 || cap.uses.length > 0) && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 6, flexWrap: 'wrap' }}>
                  {cap.needs.length > 0 && <IdChips label="Needs" ids={cap.needs} onJump={onJump} />}
                  {cap.uses.length > 0 && <IdChips label="Better with" ids={cap.uses} onJump={onJump} />}
                </div>
              )}

              {configuring && !offProvider && (
                selected
                  ? <ProviderCard info={selected} kind={cap.kind} selected startExpanded />
                  : (
                    <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 6 }}>
                      “{providerId}” is not registered on this server, so there is nothing to configure.
                    </div>
                  )
              )}
            </>
          )}
        </div>
      )}
    </article>
  )
}

function IdChips({ label, ids, onJump }: { label: string; ids: string[]; onJump: (id: string) => void }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap' }}>
      <span style={{ fontSize: 11, color: 'var(--text-faint)', marginRight: 2 }}>{label}</span>
      {ids.map((id) => (
        <button
          key={id}
          type="button"
          onClick={() => onJump(id)}
          title={`Go to ${id}`}
          className="timecode"
          style={{
            height: 18, padding: '0 6px', borderRadius: 3, fontSize: 10, cursor: 'pointer',
            border: '1px solid var(--border-strong)', background: 'transparent', color: 'var(--text-dim)',
          }}
        >
          {id}
        </button>
      ))}
    </span>
  )
}

/** Capability-level knobs (Capability.params) → pipeline.capabilities[id].params. */
function CapabilityParams({ cap, values, disabled, onSave }: {
  cap: Capability
  values: Record<string, unknown>
  disabled: boolean
  onSave: (params: Record<string, unknown>) => Promise<void>
}) {
  const [draft, setDraft] = useState<Draft>(() => fieldDraft(cap.params, values))
  const valuesKey = JSON.stringify(values)
  useEffect(() => {
    setDraft(fieldDraft(cap.params, values))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cap.id, valuesKey])

  const save = (next: Draft) => { void onSave(buildPayload(cap.params, next)) }

  return (
    <div style={{
      marginTop: 6, paddingTop: 4, borderTop: '1px solid var(--border-subtle)',
      display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '0 16px',
      opacity: disabled ? 0.6 : 1, pointerEvents: disabled ? 'none' : undefined,
    }}>
      {cap.params.map((field) => (
        <ConfigFieldInput
          key={field.key}
          field={field}
          value={draft[field.key]}
          onChange={(v) => {
            const next = { ...draft, [field.key]: v }
            setDraft(next)
            // Checkboxes and selects have no blur worth waiting for — save at once.
            if (field.type === 'boolean' || field.type === 'select') save(next)
          }}
          onBlur={() => save(draft)}
        />
      ))}
    </div>
  )
}

// ---- provider card -------------------------------------------------------------

/** One registered provider: availability, its option form, Save and Test connection.
 *  `onSelect` (when given) shows the radio that makes it the provider for `kind`. */
export function ProviderCard({ info, kind, selected, onSelect, startExpanded }: {
  info: ProviderInfo
  kind: string
  selected: boolean
  onSelect?: () => void
  startExpanded?: boolean
}) {
  const loadProviders = useStore((s) => s.loadProviders)
  const toast = useStore((s) => s.toast)

  const { meta, available, reason, configured_options: configuredOptions } = info

  const [expanded, setExpanded] = useState(Boolean(startExpanded))
  const [draft, setDraft] = useState<Draft>(() => initDraft(info))
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)

  // Re-seed the draft only when this specific provider's saved config actually
  // changes (e.g. after this card's own Save), not on unrelated store refreshes.
  const configuredKey = JSON.stringify(configuredOptions)
  useEffect(() => {
    setDraft(initDraft(info))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta.id, configuredKey])

  async function save() {
    setSaving(true)
    try {
      await api.saveProviderOptions(meta.id, buildPayload(meta.fields, draft))
      await loadProviders()
      toast('success', `Saved ${meta.name} settings`)
    } catch (e) {
      toast('error', errText(e, 'Failed to save settings'))
    } finally {
      setSaving(false)
    }
  }

  async function test() {
    setTesting(true)
    try {
      const res = await api.checkProvider(meta.id)
      toast(res.available ? 'success' : 'error', res.reason || (res.available ? `${meta.name} is ready` : 'Not available'))
    } catch (e) {
      toast('error', errText(e, 'Check failed'))
    } finally {
      setTesting(false)
    }
  }

  const runtimeBadge = meta.runtime === 'local'
    ? <Badge color="var(--ok)" bg="var(--ok-dim)">self-hosted</Badge>
    : <Badge color="var(--accent-text)" bg="var(--accent-dim)">cloud API</Badge>

  const isErrorish = /error|fail|exception/i.test(reason)
  const availColor = available ? 'var(--ok)' : (isErrorish ? 'var(--err-text)' : 'var(--warn-text)')
  const availBg = available ? 'var(--ok-dim)' : (isErrorish ? 'var(--err-dim)' : 'var(--warn-dim)')

  return (
    <div style={{
      marginTop: 8, background: 'var(--bg-panel)', borderRadius: 'var(--r-md)', overflow: 'hidden',
      border: `1px solid ${selected && onSelect ? 'var(--accent)' : 'var(--border)'}`,
    }}>
      <div style={{ height: 28, display: 'flex', alignItems: 'center', gap: 8, padding: '0 8px 0 4px', background: 'var(--bg-raised)', borderBottom: expanded ? '1px solid var(--border-subtle)' : 'none' }}>
        <Chevron open={expanded} onClick={() => setExpanded((e) => !e)} title={expanded ? `Collapse ${meta.name}` : `Configure ${meta.name}`} />
        {onSelect && (
          <input
            type="radio"
            name={`provider-${kind}`}
            checked={selected}
            onChange={onSelect}
            style={{ cursor: 'pointer', accentColor: 'var(--accent)', margin: 0 }}
            title={`Use ${meta.name} for ${kind}`}
          />
        )}
        <span style={{ fontSize: 12, fontWeight: 600, whiteSpace: 'nowrap' }}>{meta.name}</span>
        <span className="timecode" style={{ fontSize: 10, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{meta.id}</span>
        <div style={{ flex: 1 }} />
        {selected && onSelect && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--accent-text)' }}>
            <Icon name="check" size={11} /> In use
          </span>
        )}
        {BUILT_IN.test(meta.id) ? <Badge>built-in</Badge> : runtimeBadge}
        <span title={reason || undefined}>
          <Badge color={availColor} bg={availBg}>{available ? 'ready' : truncate(reason || 'unavailable', 40)}</Badge>
        </span>
      </div>

      {expanded && (
        <div style={{ padding: '6px 10px 8px' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>{meta.description}</div>
          {meta.fields.length === 0 ? (
            <div style={{ fontSize: 11, color: 'var(--text-faint)', marginBottom: 8 }}>No configuration required.</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '0 16px', marginBottom: 6 }}>
              {meta.fields.map((field) => (
                <ConfigFieldInput
                  key={field.key}
                  field={field}
                  value={draft[field.key]}
                  onChange={(v) => setDraft((d) => ({ ...d, [field.key]: v }))}
                  secretConfigured={configuredOptions[field.key] === SECRET_MASK}
                />
              ))}
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6 }}>
            <Button size="sm" onClick={() => void test()} disabled={testing}>
              {testing ? <Spinner size={11} /> : 'Test connection'}
            </Button>
            <Button size="sm" variant="primary" onClick={() => void save()} disabled={saving}>
              {saving ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
