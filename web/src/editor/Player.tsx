// Viewer: one <video> whose source is playback.mp4 (original audio), playback_dub.mp4 (the same
// picture muxed with the dub mix) or, while previewing a version, that version's copied
// playback_dub. A single file means a single media clock — the earlier hidden <audio> element,
// re-seeked to follow the muted video, stuttered whenever either stream hiccupped. Drives
// store.playhead / consumes store.seekRequest / store.playing / store.audioTrack / store.rate.
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import { Icon } from '../components/Icon'
import { Button, IconButton, Segmented, Select } from '../components/primitives'
import { useProject, useStore } from '../state/store'
import type { AudioTrack } from '../state/store'
import { formatTime } from '../types'

const RATES = [0.5, 1, 1.5, 2]

export function Player() {
  const project = useProject()
  const playing = useStore((s) => s.playing)
  const setPlaying = useStore((s) => s.setPlaying)
  const audioTrack = useStore((s) => s.audioTrack)
  const seekRequest = useStore((s) => s.seekRequest)
  const setPlayhead = useStore((s) => s.setPlayhead)
  const seek = useStore((s) => s.seek)
  const playhead = useStore((s) => s.playhead)
  const setAudioTrack = useStore((s) => s.setAudioTrack)
  const selection = useStore((s) => s.selection)
  const selectSegment = useStore((s) => s.selectSegment)
  const rate = useStore((s) => s.rate)
  const setRate = useStore((s) => s.setRate)
  const loopRange = useStore((s) => s.loopRange)
  const setLoopRange = useStore((s) => s.setLoopRange)
  const hasRange = useStore((s) => s.range !== null)
  const previewVersionId = useStore((s) => s.previewVersionId)
  const previewVersion = useStore((s) => s.versions.find((v) => v.id === s.previewVersionId) ?? null)
  const setPreviewVersion = useStore((s) => s.setPreviewVersion)
  const restoreVersion = useStore((s) => s.restoreVersion)
  const toast = useStore((s) => s.toast)

  const videoRef = useRef<HTMLVideoElement>(null)
  const rafRef = useRef<number | null>(null)
  const playheadRef = useRef(0)
  // Where to put the playhead back (and whether to keep playing) after the source swaps.
  const resumeRef = useRef<{ t: number; playing: boolean } | null>(null)

  const [volume, setVolume] = useState(1)
  const [duration, setDuration] = useState(project?.media?.duration ?? 0)

  const pid = project?.id ?? null
  const mixState = project?.stages?.mix
  const mixDone = mixState?.status === 'done'
  const mixUpdatedAt = mixState?.updated_at ?? null
  // playback.mp4 only exists once ingest completes; mounting the <video> earlier
  // leaves the element in a dead error state that browsers never retry.
  const ingestState = project?.stages?.ingest
  const ingestDone = ingestState?.status === 'done'
  const ingestUpdatedAt = ingestState?.updated_at ?? null

  const previewPath = previewVersion?.outputs?.playback_dub
  const dub = audioTrack === 'dub' || !!previewPath
  const src = !pid || !ingestDone
    ? null
    : previewPath
      ? api.mediaUrl(pid, previewPath)
      : audioTrack === 'dub' && mixDone
        ? `${api.mediaUrl(pid, 'playback_dub.mp4')}${mixUpdatedAt ? `?v=${encodeURIComponent(mixUpdatedAt)}` : ''}`
        : `${api.mediaUrl(pid, 'playback.mp4')}${ingestUpdatedAt ? `?v=${encodeURIComponent(ingestUpdatedAt)}` : ''}`

  // rAF loop: push video.currentTime -> store.playhead; wrap inside the range when looping.
  useEffect(() => {
    function tick() {
      const video = videoRef.current
      if (video) {
        const { loopRange: loop, range } = useStore.getState()
        if (loop && range && video.currentTime >= range.end) video.currentTime = range.start
        playheadRef.current = video.currentTime
        setPlayhead(video.currentTime)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [setPlayhead])

  // Changing the source reloads the element (position 0, paused): note where we were so
  // onLoadedMetadata can restore it. Playback rate is reset by a reload too, so it is re-applied there.
  const prevSrc = useRef(src)
  useEffect(() => {
    if (prevSrc.current === src) return
    prevSrc.current = src
    resumeRef.current = { t: playheadRef.current, playing }
  }, [src, playing])

  // Consume seek requests from Timeline / Script / keyboard.
  useEffect(() => {
    if (!seekRequest) return
    const video = videoRef.current
    if (video) video.currentTime = seekRequest.t
  }, [seekRequest])

  // play()/pause() when store.playing changes.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    if (playing) void video.play().catch(() => {})
    else video.pause()
  }, [playing])

  // Apply playback rate (J/K/L shuttle writes store.rate) / volume; re-applied when the element remounts.
  useEffect(() => {
    const video = videoRef.current
    if (video) { video.playbackRate = rate; video.volume = volume }
  }, [rate, volume, src])

  const sorted = useMemo(
    () => [...(project?.segments ?? [])].sort((a, b) => a.start - b.start),
    [project?.segments],
  )
  const current = sorted.find((sg) => sg.start <= playhead && playhead < sg.end) ?? null

  function step(dir: 1 | -1) {
    if (sorted.length === 0) return
    const idx = sorted.findIndex((sg) => sg.id === selection)
    const from = idx >= 0 ? idx : sorted.findIndex((sg) => sg.end > playhead) - (dir === 1 ? 1 : 0)
    const next = sorted[Math.max(0, Math.min(sorted.length - 1, from + dir))]
    selectSegment(next.id, true)
  }

  async function restorePreview() {
    if (!previewVersionId) return
    try {
      await restoreVersion(previewVersionId)
      toast('success', 'Version restored')
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to restore version')
    }
  }

  // While a version preview plays, the viewer is pinned to that version's dub: the switch is inert.
  const previewing = !!previewPath
  const trackOptions: { value: AudioTrack; label: string; title: string; disabled?: boolean }[] = [
    {
      value: 'original', label: 'Original', disabled: previewing,
      title: previewing ? 'Close the version preview to switch tracks' : 'Original audio (1)',
    },
    {
      value: 'dub', label: 'Dub', disabled: !mixDone || previewing,
      title: previewing ? 'Close the version preview to switch tracks' : mixDone ? 'Dubbed audio (2)' : 'Dub — run the Mix stage first',
    },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, background: '#000' }}>
      {previewVersion && (
        <div
          role="status"
          style={{
            height: 24, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 8, padding: '0 8px 0 12px',
            background: 'var(--bg-raised)', borderBottom: '1px solid var(--border)', fontSize: 11, color: 'var(--text)',
          }}
        >
          <Icon name="eye" size={12} style={{ color: 'var(--accent-text)' }} />
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 }}>
            Previewing <b style={{ fontWeight: 600 }}>{previewVersion.label}</b>
          </span>
          <span style={{ color: 'var(--text-faint)' }}>—</span>
          <Button size="sm" variant="ghost" onClick={() => void restorePreview()} title="Restore this version (the current state is saved first)" style={{ height: 18, padding: '0 6px', color: 'var(--accent-text)' }}>
            Restore
          </Button>
          <span style={{ color: 'var(--text-faint)' }}>·</span>
          <Button size="sm" variant="ghost" onClick={() => setPreviewVersion(null)} title="Stop previewing" style={{ height: 18, padding: '0 6px' }}>
            Close
          </Button>
        </div>
      )}

      <div
        style={{
          position: 'relative', flex: 1, minHeight: 0, background: '#000',
          display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden',
        }}
      >
        {!ingestDone && (
          <span style={{ fontSize: 11, color: 'var(--text-faint)' }}>Waiting for ingest…</span>
        )}
        {src && (
          <video
            key={ingestUpdatedAt ?? 'video'}
            ref={videoRef}
            src={src}
            onClick={() => setPlaying(!playing)}
            onEnded={() => setPlaying(false)}
            onLoadedMetadata={(e) => {
              const video = e.currentTarget
              setDuration(video.duration || project?.media?.duration || 0)
              video.playbackRate = rate
              video.volume = volume
              const resume = resumeRef.current
              if (resume) {
                resumeRef.current = null
                video.currentTime = resume.t
                if (resume.playing) void video.play().catch(() => {})
              }
            }}
            style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000', cursor: 'pointer', display: 'block' }}
          />
        )}
        {current && (
          <div style={{
            position: 'absolute', left: '6%', right: '6%', bottom: '8%', textAlign: 'center', pointerEvents: 'none',
            display: 'flex', flexDirection: 'column', gap: 3, alignItems: 'center',
            textShadow: '0 1px 2px rgb(0 0 0 / 0.9), 0 0 8px rgb(0 0 0 / 0.6)',
          }}>
            <span style={{ fontSize: 16, lineHeight: 1.35, fontWeight: 500, color: '#fff' }}>
              {dub ? (current.translated_text || current.source_text) : current.source_text}
            </span>
            {dub && current.translated_text && (
              <span style={{ fontSize: 12, lineHeight: 1.35, color: 'rgb(255 255 255 / 0.75)' }}>{current.source_text}</span>
            )}
          </div>
        )}
      </div>

      <div
        role="toolbar"
        aria-label="Transport"
        style={{
          height: 32, flexShrink: 0, display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center',
          padding: '0 8px', gap: 8, background: 'var(--bg-raised)', borderTop: '1px solid var(--border)',
        }}
      >
        <span className="timecode" style={{ fontSize: 12, color: 'var(--text)', whiteSpace: 'nowrap', justifySelf: 'start' }}>
          {formatTime(playhead, true)}
          <span style={{ color: 'var(--text-dim)' }}> / {formatTime(duration, true)}</span>
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <IconButton name="prev" title="Previous line (↑)" onClick={() => step(-1)} />
          <IconButton name="back" title="Step back 1 s (←)" onClick={() => seek(playhead - 1)} />
          <IconButton
            name={playing ? 'pause' : 'play'}
            title={playing ? 'Pause (Space)' : 'Play (Space)'}
            onClick={() => setPlaying(!playing)}
            size={28} iconSize={16}
            style={{ color: 'var(--text)' }}
          />
          <IconButton name="forward" title="Step forward 1 s (→)" onClick={() => seek(playhead + 1)} />
          <IconButton name="next" title="Next line (↓)" onClick={() => step(1)} />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifySelf: 'end' }}>
          <Segmented<AudioTrack>
            size="sm"
            ariaLabel="Audio track"
            value={previewPath ? 'dub' : audioTrack}
            onChange={setAudioTrack}
            options={trackOptions}
          />
          <IconButton
            name="loop"
            title={hasRange ? (loopRange ? 'Loop range: on' : 'Loop range: off') : 'Loop range (set a range with I / O first)'}
            active={loopRange}
            onClick={() => setLoopRange(!loopRange)}
          />
          <Select
            ariaLabel="Playback rate"
            value={String(rate)}
            onChange={(v) => setRate(Number(v))}
            options={(RATES.includes(rate) ? RATES : [...RATES, rate].sort((a, b) => a - b)).map((r) => ({ value: String(r), label: `${r}×` }))}
            style={{ width: 58, height: 22, fontSize: 11, padding: '0 4px' }}
          />
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={volume}
            onChange={(e) => setVolume(Number(e.target.value))}
            aria-label="Volume"
            title={`Volume ${Math.round(volume * 100)}%`}
            style={{ width: 60, accentColor: 'var(--accent)', margin: 0 }}
          />
        </div>
      </div>
    </div>
  )
}
