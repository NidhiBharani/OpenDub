// Versions panel (docs/plans/editor-redesign.md §6): snapshots grouped by target language, newest
// first, grouped by day. Preview plays the version in the viewer; Restore is non-destructive (the
// server saves the current state as a restore point first).
import { useEffect, useMemo, useState } from 'react'
import { Badge, Button, IconButton, Modal, Segmented, TextInput } from '../components/primitives'
import { useProject, useStore } from '../state/store'
import type { Version } from '../types'

const fmtBytes = (n: number): string => {
  if (n >= 1024 * 1024 * 1024) return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`
  if (n >= 1024 * 1024) return `${Math.round(n / (1024 * 1024))} MB`
  if (n >= 1024) return `${Math.round(n / 1024)} KB`
  return `${n} B`
}

const dayKey = (iso: string): string => {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function dayLabel(key: string, now = new Date()): string {
  if (!key) return 'Unknown date'
  if (key === dayKey(now.toISOString())) return 'Today'
  const y = new Date(now); y.setDate(y.getDate() - 1)
  if (key === dayKey(y.toISOString())) return 'Yesterday'
  const [yy, mm, dd] = key.split('-').map(Number)
  const d = new Date(yy, mm - 1, dd)
  const opts: Intl.DateTimeFormatOptions = { day: 'numeric', month: 'short' }
  if (yy !== now.getFullYear()) opts.year = 'numeric'
  return d.toLocaleDateString(undefined, opts)
}

const clock = (iso: string): string => {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

/** The `.timecode` summary line under a version's label. */
export function versionSummary(v: Version): string {
  const parts = [clock(v.created_at), `${v.preset}/${v.runtime}`]
  if (v.providers?.tts) parts.push(v.providers.tts)
  parts.push(`${v.segment_count} ${v.segment_count === 1 ? 'line' : 'lines'}`)
  parts.push(`${v.voiced_count} voiced`)
  if (v.skipped_ranges > 0) parts.push(`${v.skipped_ranges} kept`)
  if (v.bytes > 0) parts.push(fmtBytes(v.bytes))
  return parts.filter(Boolean).join(' · ')
}

export function VersionsPanel() {
  const project = useProject()
  const versions = useStore((s) => s.versions)
  const previewVersionId = useStore((s) => s.previewVersionId)
  const setPreviewVersion = useStore((s) => s.setPreviewVersion)
  const restoreVersion = useStore((s) => s.restoreVersion)
  const renameVersion = useStore((s) => s.renameVersion)
  const deleteVersion = useStore((s) => s.deleteVersion)
  const toast = useStore((s) => s.toast)

  const projectLang = project?.target_lang ?? ''
  const langs = useMemo(() => {
    const counts = new Map<string, number>()
    counts.set(projectLang, 0)
    for (const v of versions) counts.set(v.target_lang, (counts.get(v.target_lang) ?? 0) + 1)
    return [...counts.entries()].map(([code, n]) => ({ code, n }))
  }, [versions, projectLang])

  const [lang, setLang] = useState(projectLang)
  useEffect(() => { setLang(projectLang) }, [projectLang])
  const activeLang = langs.some((l) => l.code === lang) ? lang : projectLang

  const [confirm, setConfirm] = useState<{ kind: 'restore' | 'delete'; id: string } | null>(null)
  const [renaming, setRenaming] = useState<{ id: string; draft: string } | null>(null)

  const groups = useMemo(() => {
    const list = versions
      .filter((v) => v.target_lang === activeLang)
      .sort((a, b) => b.created_at.localeCompare(a.created_at))
    const out: { key: string; items: Version[] }[] = []
    for (const v of list) {
      const key = dayKey(v.created_at)
      const g = out[out.length - 1]
      if (g && g.key === key) g.items.push(v)
      else out.push({ key, items: [v] })
    }
    return out
  }, [versions, activeLang])

  if (!project) return null

  const confirmVersion = confirm ? versions.find((v) => v.id === confirm.id) ?? null : null

  async function doRestore(id: string) {
    setConfirm(null)
    try {
      await restoreVersion(id)
      toast('success', 'Version restored — the previous state was saved as a restore point')
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to restore version')
    }
  }
  async function doDelete(id: string) {
    setConfirm(null)
    try {
      await deleteVersion(id)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to delete version')
    }
  }
  async function commitRename() {
    if (!renaming) return
    const { id, draft } = renaming
    setRenaming(null)
    const v = versions.find((x) => x.id === id)
    const label = draft.trim()
    if (!v || !label || label === v.label) return
    try {
      await renameVersion(id, label)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to rename version')
    }
  }

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <div style={{ height: 28, flexShrink: 0, display: 'flex', alignItems: 'center', padding: '0 8px', borderBottom: '1px solid var(--border-subtle)' }}>
        <Segmented
          size="sm"
          ariaLabel="Target language"
          value={activeLang}
          onChange={setLang}
          options={langs.map((l) => ({
            value: l.code,
            label: <span>{l.code.toUpperCase()} <span style={{ color: 'var(--text-faint)', fontWeight: 500 }}>{l.n}</span></span>,
            title: `${l.n} ${l.n === 1 ? 'version' : 'versions'} in ${l.code.toUpperCase()}${l.code === projectLang ? ' (current target)' : ''}`,
          }))}
        />
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
        {groups.length === 0 ? (
          <div style={{ padding: '24px 16px', fontSize: 12, lineHeight: 1.5, color: 'var(--text-dim)', textAlign: 'center' }}>
            Versions appear after each run that completes Mix. Snapshot now to save the current state.
          </div>
        ) : groups.map((g) => (
          <section key={g.key} aria-label={dayLabel(g.key)}>
            <div className="eyebrow" style={{ height: 24, display: 'flex', alignItems: 'center', padding: '0 12px', fontSize: 11 }}>
              {dayLabel(g.key)}
            </div>
            <ul role="list" style={{ listStyle: 'none', margin: 0, padding: 0 }}>
              {g.items.map((v) => (
                <VersionRow
                  key={v.id}
                  version={v}
                  current={v.id === project.active_version_id}
                  previewing={v.id === previewVersionId}
                  renaming={renaming?.id === v.id ? renaming.draft : null}
                  onRenameDraft={(draft) => setRenaming({ id: v.id, draft })}
                  onRenameStart={() => setRenaming({ id: v.id, draft: v.label })}
                  onRenameCommit={() => void commitRename()}
                  onRenameCancel={() => setRenaming(null)}
                  onPreview={() => setPreviewVersion(v.id === previewVersionId ? null : v.id)}
                  onRestore={() => setConfirm({ kind: 'restore', id: v.id })}
                  onDelete={() => setConfirm({ kind: 'delete', id: v.id })}
                />
              ))}
            </ul>
          </section>
        ))}
      </div>

      {confirm && confirmVersion && confirm.kind === 'restore' && (
        <Modal title="Restore this version?" onClose={() => setConfirm(null)} width={400}>
          <div style={{ fontSize: 12, lineHeight: 1.5, color: 'var(--text-dim)', marginBottom: 14 }}>
            <b style={{ color: 'var(--text)', fontWeight: 600 }}>{confirmVersion.label}</b> becomes the working state.
            The current state is saved first as a restore point.
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <Button onClick={() => setConfirm(null)}>Cancel</Button>
            <Button variant="primary" autoFocus onClick={() => void doRestore(confirmVersion.id)}>Restore</Button>
          </div>
        </Modal>
      )}
      {confirm && confirmVersion && confirm.kind === 'delete' && (
        <Modal title="Delete this version?" onClose={() => setConfirm(null)} width={400}>
          <div style={{ fontSize: 12, lineHeight: 1.5, color: 'var(--text-dim)', marginBottom: 14 }}>
            <b style={{ color: 'var(--text)', fontWeight: 600 }}>{confirmVersion.label}</b> and its copied outputs
            ({fmtBytes(confirmVersion.bytes)}) are removed. Cannot be undone.
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <Button onClick={() => setConfirm(null)}>Cancel</Button>
            <Button variant="danger" autoFocus onClick={() => void doDelete(confirmVersion.id)}>Delete</Button>
          </div>
        </Modal>
      )}
    </div>
  )
}

function VersionRow({
  version: v, current, previewing, renaming, onRenameDraft, onRenameStart, onRenameCommit, onRenameCancel,
  onPreview, onRestore, onDelete,
}: {
  version: Version; current: boolean; previewing: boolean; renaming: string | null
  onRenameDraft: (draft: string) => void; onRenameStart: () => void; onRenameCommit: () => void; onRenameCancel: () => void
  onPreview: () => void; onRestore: () => void; onDelete: () => void
}) {
  const [hover, setHover] = useState(false)
  const showActions = hover || previewing
  const kindBadge = v.kind === 'manual' ? 'manual' : v.kind === 'pre_restore' ? 'restore point' : null
  return (
    <li
      aria-label={v.label}
      aria-current={previewing ? 'true' : undefined}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setHover(false) }}
      onDoubleClick={() => { if (!renaming) onRenameStart() }}
      style={{
        display: 'flex', alignItems: 'center', gap: 8, minHeight: 44, padding: '4px 8px 4px 12px',
        borderBottom: '1px solid var(--border-subtle)',
        background: previewing ? 'var(--accent-dim)' : hover ? 'var(--bg-raised)' : 'transparent',
        boxShadow: previewing ? 'inset 2px 0 0 var(--accent)' : 'none',
      }}
    >
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {renaming !== null ? (
          <TextInput
            autoFocus
            ariaLabel="Version label"
            value={renaming}
            onChange={onRenameDraft}
            onBlur={onRenameCommit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { e.preventDefault(); onRenameCommit() }
              else if (e.key === 'Escape') { e.preventDefault(); onRenameCancel() }
            }}
            style={{ height: 22, fontSize: 12 }}
          />
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
            <span style={{
              fontSize: 12, fontWeight: current ? 600 : 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {v.label}
            </span>
            {current && <Badge color="var(--accent-text)" bg="var(--accent-dim)" title="The current state came from this version">current</Badge>}
            {kindBadge && <Badge title={v.kind === 'pre_restore' ? 'Saved automatically before a restore' : 'Saved with Snapshot now'}>{kindBadge}</Badge>}
          </div>
        )}
        <span className="timecode" style={{ fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={versionSummary(v)}>
          {versionSummary(v)}
        </span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 0, flexShrink: 0, opacity: showActions ? 1 : 0, transition: 'opacity var(--ease)' }}>
        <IconButton
          name="eye" active={previewing}
          title={previewing ? 'Stop previewing' : (v.outputs?.playback_dub ? 'Preview in the viewer' : 'Preview (no dub output in this version)')}
          onClick={onPreview}
        />
        <IconButton name="restore" title="Restore this version" onClick={onRestore} />
        <IconButton name="edit" title="Rename" onClick={onRenameStart} />
        <IconButton name="trash" title="Delete" onClick={onDelete} danger />
      </div>
    </li>
  )
}
