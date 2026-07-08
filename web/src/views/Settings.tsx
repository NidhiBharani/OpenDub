// Settings view: per-stage provider selection + dynamic config forms.
// The configurability heart of the product — every pipeline stage's provider
// (self-hosted OSS vs cloud API) is picked and configured here.
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { Badge, Button, Field, Select, Spinner, TextInput } from '../components/primitives'
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
    if (str === '') continue // omit — let the server clear/leave this field
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

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <header
        style={{
          height: 44, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10,
          padding: '0 12px', borderBottom: '1px solid var(--border)',
        }}
      >
        <Button variant="ghost" onClick={() => setView(project ? 'editor' : 'library')} title="Back">‹</Button>
        <div style={{ fontSize: 13, fontWeight: 600 }}>Settings</div>
        <div style={{ flex: 1 }} />
        {project && (
          <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>
            Pipeline choices apply to ‹{project.name}›
          </div>
        )}
      </header>

      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
        <nav style={{ width: 200, flexShrink: 0, borderRight: '1px solid var(--border)', padding: 12, overflowY: 'auto' }}>
          {PROVIDER_KINDS.map((kind) => {
            const active = kind === activeKind
            const selectedId = project?.pipeline[kind]?.provider_id
            const selectedName = providers.find((p) => p.meta.id === selectedId)?.meta.name
            return (
              <button
                key={kind}
                type="button"
                onClick={() => setActiveKind(kind)}
                style={{
                  position: 'relative', display: 'block', width: '100%', textAlign: 'left',
                  border: 'none', cursor: 'pointer', color: 'var(--text)',
                  background: active ? 'var(--bg-overlay)' : 'transparent',
                  fontSize: 13, padding: '8px 10px', borderRadius: 'var(--r-md)', marginBottom: 2,
                }}
              >
                {active && (
                  <div style={{
                    position: 'absolute', left: 0, top: 4, bottom: 4, width: 2,
                    background: 'var(--accent)', borderRadius: 1,
                  }} />
                )}
                <div>{KIND_LABELS[kind]}</div>
                {project && (
                  <div style={{
                    fontSize: 11, color: 'var(--text-dim)', marginTop: 2,
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {selectedName ?? '—'}
                  </div>
                )}
              </button>
            )
          })}
        </nav>

        <div style={{ flex: 1, minWidth: 0, overflowY: 'auto', padding: 20 }}>
          <div style={{ maxWidth: 760 }}>
            <h2 style={{ fontSize: 16, fontWeight: 600, margin: '0 0 4px' }}>{KIND_LABELS[activeKind]}</h2>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 16 }}>
              {KIND_DESCRIPTIONS[activeKind]}
            </div>

            {kindProviders.length === 0 && (
              <div style={{ fontSize: 12, color: 'var(--text-faint)' }}>No providers registered for this stage.</div>
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
            style={{ width: 14, height: 14, accentColor: 'var(--accent)', cursor: 'pointer' }}
          />
          <span>{field.label}</span>
          {field.help && <span style={{ fontSize: 11, color: 'var(--text-faint)' }}>— {field.help}</span>}
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
        <Field key={field.key} label={field.label} help={isConfigured ? 'configured' : field.help}>
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
    ? <Badge color="var(--ok)" bg="rgb(80 200 120 / 0.15)">self-hosted</Badge>
    : <Badge color="var(--running)" bg="rgb(76 155 232 / 0.15)">cloud API</Badge>

  const isErrorish = /error|fail|exception/i.test(reason)
  const availColor = available ? 'var(--ok)' : (isErrorish ? 'var(--err)' : 'var(--warn)')
  const availBg = available
    ? 'rgb(80 200 120 / 0.15)'
    : (isErrorish ? 'rgb(232 96 76 / 0.15)' : 'rgb(232 184 76 / 0.15)')

  return (
    <div style={{
      background: 'var(--bg-raised)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)',
      padding: 16, marginBottom: 12,
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
        <div style={{ fontSize: 14, fontWeight: 600 }}>{meta.name}</div>
        {runtimeBadge}
        <span title={reason || undefined}>
          <Badge color={availColor} bg={availBg}>{available ? 'ready' : truncate(reason || 'unavailable', 40)}</Badge>
        </span>
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4 }}>{meta.description}</div>

      <div style={{ marginTop: 10 }}>
        <Button variant="ghost" onClick={() => setExpanded((e) => !e)} style={{ padding: '3px 6px' }}>
          {expanded ? '▾' : '▸'} Configure
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
            <Button variant="ghost" onClick={() => void test()} disabled={testing}>
              {testing ? <Spinner size={12} /> : 'Test'}
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
