// Settings view: per-stage provider selection + dynamic config forms.
// The configurability heart of the product — every pipeline stage's provider
// (self-hosted OSS vs cloud API) is picked and configured here.
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { Icon } from '../components/Icon'
import { Badge, Button, Field, Select, Spinner, TextInput } from '../components/primitives'
import { NavLink, TopBar } from '../components/TopBar'
import { useStore } from '../state/store'
import { PROVIDER_KINDS, SECRET_MASK } from '../types'
import type { ConfigField, ProviderInfo, ProviderKind } from '../types'

const KIND_LABELS: Record<ProviderKind, string> = {
  separation: 'Separation',
  asr: 'Transcription',
  diarization: 'Diarization',
  translation: 'Translation',
  tts: 'Voice',
  lipsync: 'Lip sync',
}

const KIND_DESCRIPTIONS: Record<ProviderKind, string> = {
  separation: 'Splits dialogue vocals from music and sound effects before transcription.',
  asr: 'Transcribes the spoken dialogue in the source language to text.',
  diarization: 'Identifies who is speaking and groups lines by speaker.',
  translation: "Translates the transcript into the project's target language.",
  tts: "Synthesizes the dubbed voice, cloning each speaker's tone and emotional delivery.",
  lipsync: 'Re-times mouth movement in the video to match the new dub audio.',
}

/** Draft values keyed by ConfigField.key. Booleans stay booleans; everything
 *  else (string/number/select/secret) is edited as a string and converted on save. */
type Draft = Record<string, string | boolean>

function initDraft(info: ProviderInfo): Draft {
  const draft: Draft = {}
  for (const field of info.meta.fields) {
    const raw = info.configured_options[field.key]
    if (field.type === 'boolean') {
      draft[field.key] = raw !== undefined && raw !== null ? Boolean(raw) : Boolean(field.default)
    } else if (field.type === 'secret') {
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

export function Settings() {
  const project = useStore((s) => s.project)
  const providers = useStore((s) => s.providers)
  const loadProviders = useStore((s) => s.loadProviders)
  const setView = useStore((s) => s.setView)
  const closeProject = useStore((s) => s.closeProject)

  const [activeKind, setActiveKind] = useState<ProviderKind>('separation')

  useEffect(() => {
    void loadProviders()
  }, [loadProviders])

  const kindProviders = providers
    .filter((p) => p.meta.kind === activeKind)
    .sort((a, b) => {
      if (a.meta.runtime !== b.meta.runtime) return a.meta.runtime === 'local' ? -1 : 1
      return a.meta.name.localeCompare(b.meta.name)
    })

  const BUILT_IN = /(mock|passthrough|none|single_speaker)$/

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <TopBar padX={48} onBrand={() => (project ? closeProject() : setView('library'))}>
        <NavLink label="Library" onClick={() => (project ? closeProject() : setView('library'))} />
        {project && <NavLink label="Editor" onClick={() => setView('editor')} />}
        <NavLink label="Settings" active onClick={() => setView('settings')} />
        <div style={{ flex: 1 }} />
        <div style={{ fontSize: 12, color: 'var(--text-dim)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {project ? <>Provider choices apply to ‹{project.name}›</> : 'Open a project to choose its providers'}
        </div>
      </TopBar>

      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
        <nav aria-label="Pipeline stages" style={{
          width: 380, flexShrink: 0, borderRight: '1px solid var(--border-strong)', padding: '36px 24px 24px 48px',
          overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6,
        }}>
          <h1 className="serif" style={{ margin: '0 0 4px', fontSize: 40, lineHeight: 1 }}>The signal chain</h1>
          <p style={{ margin: '0 0 14px', fontSize: 13, color: 'var(--text-dim)' }}>
            Each stage hands off to one provider. Mix local models, cloud APIs and built-in fallbacks freely.
          </p>
          {PROVIDER_KINDS.map((kind, i) => {
            const active = kind === activeKind
            const selectedId = project?.pipeline[kind]?.provider_id
            const selected = providers.find((p) => p.meta.id === selectedId)
            const tag = !selected ? null : BUILT_IN.test(selected.meta.id) ? 'BUILT-IN' : selected.meta.runtime === 'local' ? 'LOCAL' : 'CLOUD'
            return (
              <button
                key={kind}
                type="button"
                aria-current={active ? 'true' : undefined}
                onClick={() => setActiveKind(kind)}
                style={{
                  minHeight: 58, padding: '0 14px', borderRadius: 10, cursor: 'pointer', textAlign: 'left',
                  display: 'flex', alignItems: 'center', gap: 12, color: 'var(--text)',
                  border: `1px solid ${active ? 'var(--border-strong)' : 'transparent'}`,
                  background: active ? 'var(--bg-raised)' : 'transparent',
                }}
              >
                <span className="timecode" style={{ fontSize: 11, color: active ? 'var(--accent)' : 'var(--text-dim)' }}>
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <span style={{ fontSize: 14, fontWeight: 500 }}>{KIND_LABELS[kind]}</span>
                  {project && (
                    <span style={{
                      fontSize: 12, color: 'var(--text-dim)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    }}>
                      {selected?.meta.name ?? '—'}
                    </span>
                  )}
                </span>
                {tag && (
                  <span className="timecode" style={{
                    fontSize: 10, letterSpacing: '0.06em',
                    color: tag === 'LOCAL' ? 'var(--ok)' : tag === 'CLOUD' ? 'var(--accent-text)' : 'var(--text-dim)',
                  }}>
                    {tag}
                  </span>
                )}
              </button>
            )
          })}
          <div style={{ flex: 1 }} />
          <p style={{ margin: '16px 0 0', fontSize: 12, color: 'var(--text-dim)' }}>
            Saved to <span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>configs/settings.yaml</span>. Environment
            variables override the file.
          </p>
        </nav>

        <div style={{ flex: 1, minWidth: 0, overflowY: 'auto', padding: '36px 48px' }}>
          <div style={{ maxWidth: 920 }}>
            <div className="timecode" style={{ fontSize: 11, letterSpacing: '0.08em', color: 'var(--accent)' }}>
              STAGE {String(PROVIDER_KINDS.indexOf(activeKind) + 1).padStart(2, '0')}
            </div>
            <h2 className="serif" style={{ fontSize: 40, lineHeight: 1, margin: '6px 0' }}>{KIND_LABELS[activeKind]}</h2>
            <div style={{ fontSize: 14, color: 'var(--text-dim)', marginBottom: 22, maxWidth: 640 }}>
              {KIND_DESCRIPTIONS[activeKind]}
            </div>

            {kindProviders.length === 0 && (
              <div style={{ fontSize: 13, color: 'var(--text-dim)' }}>No providers registered for this stage.</div>
            )}

            {kindProviders.map((info) => (
              <ProviderCard key={info.meta.id} info={info} kind={activeKind} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function ProviderCard({ info, kind }: { info: ProviderInfo; kind: ProviderKind }) {
  const project = useStore((s) => s.project)
  const applyProject = useStore((s) => s.applyProject)
  const loadProviders = useStore((s) => s.loadProviders)
  const toast = useStore((s) => s.toast)

  const { meta, available, reason, configured_options: configuredOptions } = info

  const [expanded, setExpanded] = useState(false)
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

  const selected = project?.pipeline[kind]?.provider_id === meta.id

  async function selectProvider() {
    if (!project) return
    try {
      const updated = await api.updateProject(project.id, {
        pipeline: { ...project.pipeline, [kind]: { provider_id: meta.id, options: {} } },
      })
      applyProject(updated)
      toast('success', `${meta.name} set as the ${KIND_LABELS[kind]} provider`)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to update pipeline provider')
    }
  }

  async function save() {
    setSaving(true)
    try {
      await api.saveProviderOptions(meta.id, buildPayload(meta.fields, draft))
      await loadProviders()
      toast('success', `Saved ${meta.name} settings`)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to save settings')
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
      toast('error', e instanceof Error ? e.message : 'Check failed')
    } finally {
      setTesting(false)
    }
  }

  function renderField(field: ConfigField) {
    const value = draft[field.key]

    if (field.type === 'boolean') {
      return (
        <label
          key={field.key}
          style={{
            gridColumn: '1 / -1', display: 'flex', alignItems: 'center', gap: 8,
            fontSize: 13, padding: '6px 0', cursor: 'pointer',
          }}
        >
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => setDraft((d) => ({ ...d, [field.key]: e.target.checked }))}
            style={{ width: 16, height: 16, accentColor: 'var(--accent)', cursor: 'pointer' }}
          />
          <span>{field.label}</span>
          {field.help && <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>— {field.help}</span>}
        </label>
      )
    }

    if (field.type === 'select') {
      return (
        <Field key={field.key} label={field.label} help={field.help}>
          <Select
            value={String(value ?? '')}
            onChange={(v) => setDraft((d) => ({ ...d, [field.key]: v }))}
            options={field.options.map((o) => ({ value: o, label: o }))}
          />
        </Field>
      )
    }

    if (field.type === 'secret') {
      const isConfigured = configuredOptions[field.key] === SECRET_MASK
      return (
        <Field key={field.key} label={field.label} help={isConfigured ? 'Configured — write-only, leave blank to keep the saved key.' : field.help}>
          <TextInput
            type="password"
            value={String(value ?? '')}
            onChange={(v) => setDraft((d) => ({ ...d, [field.key]: v }))}
            placeholder={isConfigured ? SECRET_MASK : field.placeholder}
          />
        </Field>
      )
    }

    // string | number
    return (
      <Field key={field.key} label={field.label} help={field.help}>
        <TextInput
          type={field.type === 'number' ? 'number' : 'text'}
          value={String(value ?? '')}
          onChange={(v) => setDraft((d) => ({ ...d, [field.key]: v }))}
          placeholder={field.placeholder}
        />
      </Field>
    )
  }

  const runtimeBadge = meta.runtime === 'local'
    ? <Badge color="var(--ok)" bg="var(--ok-dim)">self-hosted</Badge>
    : <Badge color="var(--accent-text)" bg="var(--accent-dim)">cloud API</Badge>

  const isErrorish = /error|fail|exception/i.test(reason)
  const availColor = available ? 'var(--ok)' : (isErrorish ? 'var(--err-text)' : 'var(--warn-text)')
  const availBg = available ? 'var(--ok-dim)' : (isErrorish ? 'var(--err-dim)' : 'var(--warn-dim)')

  return (
    <div style={{
      background: selected ? 'var(--accent-dim)' : 'var(--bg-raised)', borderRadius: 'var(--r-lg)',
      border: selected ? '1.5px solid var(--accent)' : '1px solid var(--border-strong)',
      padding: '16px 18px', marginBottom: 12,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        {project && (
          <input
            type="radio"
            name={`provider-${kind}`}
            checked={selected}
            onChange={() => void selectProvider()}
            style={{ cursor: 'pointer', accentColor: 'var(--accent)' }}
            title={`Use ${meta.name} for ${KIND_LABELS[kind]}`}
          />
        )}
        <div style={{ fontSize: 15, fontWeight: 500 }}>{meta.name}</div>
        <span className="timecode" style={{ fontSize: 11 }}>{meta.id}</span>
        <div style={{ flex: 1 }} />
        {selected && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'var(--accent-text)' }}>
            <Icon name="check" size={12} /> In use
          </span>
        )}
        {runtimeBadge}
        <span title={reason || undefined}>
          <Badge color={availColor} bg={availBg}>{available ? 'ready' : truncate(reason || 'unavailable', 40)}</Badge>
        </span>
      </div>
      <div style={{ fontSize: 13, color: 'var(--text-dim)', marginTop: 6 }}>{meta.description}</div>

      <div style={{ marginTop: 10 }}>
        <Button variant="ghost" onClick={() => setExpanded((e) => !e)} style={{ padding: '0 6px', minHeight: 32 }}>
          <Icon name="chevron" size={12} style={{ transform: expanded ? 'none' : 'rotate(-90deg)', transition: 'transform var(--ease)' }} />
          Configure
        </Button>
      </div>

      {expanded && (
        <div style={{ marginTop: 12, borderTop: '1px solid var(--border)', paddingTop: 12 }}>
          {meta.fields.length === 0 ? (
            <div style={{ fontSize: 12, color: 'var(--text-faint)', marginBottom: 12 }}>
              No configuration required.
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0 16px' }}>
              {meta.fields.map((field) => renderField(field))}
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <Button onClick={() => void test()} disabled={testing}>
              {testing ? <Spinner size={12} /> : 'Test connection'}
            </Button>
            <Button variant="primary" onClick={() => void save()} disabled={saving}>
              {saving ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
