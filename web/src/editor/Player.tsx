// Video player: plays playback.mp4, keeps a hidden dub_mix.wav <audio> element in sync,
// and drives store.playhead / consumes store.seekRequest / store.playing / store.audioTrack.
import type { CSSProperties } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import { Icon } from '../components/Icon'
import { Select } from '../components/primitives'
import { useProject, useStore } from '../state/store'
import { formatTime } from '../types'

const RATES = [0.5, 1, 1.5, 2]

export function Player() {
  const project = useProject()
  const playing = useStore((s) => s.playing)
  const setPlaying = useStore((s) => s.setPlaying)
  const audioTrack = useStore((s) => s.audioTrack)
  const seekRequest = useStore((s) => s.seekRequest)
  const setPlayhead = useStore((s) => s.setPlayhead)
  const playhead = useStore((s) => s.playhead)
  const setAudioTrack = useStore((s) => s.setAudioTrack)
  const selection = useStore((s) => s.selection)
  const selectSegment = useStore((s) => s.selectSegment)

  const videoRef = useRef<HTMLVideoElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const rafRef = useRef<number | null>(null)

  const [rate, setRate] = useState(1)
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

  // rAF loop: push video.currentTime -> store.playhead, and drift-correct the dub audio.
  useEffect(() => {
    function tick() {
      const video = videoRef.current
      const audio = audioRef.current
      if (video) {
        setPlayhead(video.currentTime)
        if (audio && Math.abs(audio.currentTime - video.currentTime) > 0.08) {
          audio.currentTime = video.currentTime
        }
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [setPlayhead])

  // Consume seek requests from Timeline / TranscriptList / keyboard.
  useEffect(() => {
    if (!seekRequest) return
    const video = videoRef.current
    const audio = audioRef.current
    if (video) video.currentTime = seekRequest.t
    if (audio) audio.currentTime = seekRequest.t
  }, [seekRequest])

  // play()/pause() both elements when store.playing changes.
  useEffect(() => {
    const video = videoRef.current
    const audio = audioRef.current
    if (playing) {
      void video?.play().catch(() => {})
      if (audioTrack === 'dub') void audio?.play().catch(() => {})
    } else {
      video?.pause()
      audio?.pause()
    }
  }, [playing, audioTrack])

  // Mute/unmute video vs. dub audio when the audio track selection changes.
  useEffect(() => {
    const video = videoRef.current
    const audio = audioRef.current
    if (video) video.muted = audioTrack === 'dub'
    if (audio) {
      audio.muted = audioTrack !== 'dub'
      if (audioTrack === 'dub') {
        if (playing) {
          if (video) audio.currentTime = video.currentTime
          void audio.play().catch(() => {})
        }
      } else {
        audio.pause()
      }
    }
  }, [audioTrack, playing])

  // Apply playback rate / volume to both elements (also re-applied when either element remounts).
  useEffect(() => {
    const video = videoRef.current
    const audio = audioRef.current
    if (video) { video.playbackRate = rate; video.volume = volume }
    if (audio) { audio.playbackRate = rate; audio.volume = volume }
  }, [rate, volume, mixDone, mixUpdatedAt, ingestDone, ingestUpdatedAt])

  const sorted = useMemo(
    () => [...(project?.segments ?? [])].sort((a, b) => a.start - b.start),
    [project?.segments],
  )
  const current = sorted.find((sg) => sg.start <= playhead && playhead < sg.end) ?? null
  const dub = audioTrack === 'dub'

  function step(dir: 1 | -1) {
    if (sorted.length === 0) return
    const idx = sorted.findIndex((sg) => sg.id === selection)
    const from = idx >= 0 ? idx : sorted.findIndex((sg) => sg.end > playhead) - (dir === 1 ? 1 : 0)
    const next = sorted[Math.max(0, Math.min(sorted.length - 1, from + dir))]
    selectSegment(next.id, true)
  }

  const roundBtn: CSSProperties = {
    width: 44, height: 44, borderRadius: '50%', border: '1px solid var(--border-strong)',
    background: 'transparent', color: 'var(--text)', cursor: 'pointer', flexShrink: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  }
  const trackBtn = (on: boolean, enabled = true): CSSProperties => ({
    height: 36, padding: '0 12px', border: 'none', borderRadius: 7, fontSize: 13, fontWeight: 500,
    display: 'flex', alignItems: 'center', gap: 8, cursor: enabled ? 'pointer' : 'not-allowed',
    // sits on the (always dark) video, so it keeps fixed colours in both themes
    background: on ? '#f3eeff' : 'transparent', color: on ? '#120b24' : '#d9d2ee',
    opacity: enabled ? 1 : 0.5,
  })

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0,
      padding: '16px 20px', gap: 14, alignItems: 'center',
    }}>
      <div
        style={{
          position: 'relative', flex: 1, minHeight: 0, width: '100%', maxWidth: 960, borderRadius: 10,
          overflow: 'hidden', border: '1px solid var(--border)',
          background: 'repeating-linear-gradient(135deg, #0a0617 0 14px, #130c28 14px 28px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        {!ingestDone && (
          <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: '#7a6fa3' }}>
            video preview · appears after ingest
          </span>
        )}
        {pid && ingestDone && (
          <video
            key={ingestUpdatedAt ?? 'video'}
            ref={videoRef}
            src={`${api.mediaUrl(pid, 'playback.mp4')}${ingestUpdatedAt ? `?v=${encodeURIComponent(ingestUpdatedAt)}` : ''}`}
            onClick={() => setPlaying(!playing)}
            onEnded={() => setPlaying(false)}
            onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || project?.media?.duration || 0)}
            style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000', cursor: 'pointer' }}
          />
        )}
        {pid && mixDone && (
          <audio
            key={mixUpdatedAt ?? 'mix'}
            ref={audioRef}
            // m4a, not wav: browsers (Safari especially) stall streaming multi-MB PCM WAV
            src={`${api.mediaUrl(pid, 'audio/dub_mix.m4a')}${mixUpdatedAt ? `?v=${encodeURIComponent(mixUpdatedAt)}` : ''}`}
            style={{ display: 'none' }}
          />
        )}
        <div style={{
          position: 'absolute', top: 12, left: 12, display: 'flex', alignItems: 'center', gap: 6,
          padding: '4px 9px', borderRadius: 20, background: 'rgb(12 7 26 / 0.8)', pointerEvents: 'none',
          fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '0.06em',
          color: dub ? '#2fe6d6' : '#ff4fa3',
        }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor' }} />
          {dub ? `DUBBED · ${(project?.target_lang ?? '').toUpperCase()}` : `ORIGINAL · ${(project?.source_lang ?? '').toUpperCase()}`}
        </div>
        <div role="group" aria-label="Audio track" style={{
          position: 'absolute', top: 10, right: 10, display: 'flex', padding: 3, borderRadius: 10,
          background: 'rgb(12 7 26 / 0.8)', border: '1px solid rgb(243 238 255 / 0.18)',
        }}>
          <button type="button" style={trackBtn(!dub)} onClick={() => setAudioTrack('original')}>
            <span className="timecode" style={{ fontSize: 11, color: 'inherit', opacity: 0.7 }}>1</span>Original
          </button>
          <button
            type="button"
            disabled={!mixDone}
            title={mixDone ? undefined : 'Run the Mix stage to preview the dub audio'}
            style={trackBtn(dub, mixDone)}
            onClick={() => setAudioTrack('dub')}
          >
            <span className="timecode" style={{ fontSize: 11, color: 'inherit', opacity: 0.7 }}>2</span>Dubbed
          </button>
        </div>
        {current && (
          <div style={{
            position: 'absolute', left: 40, right: 40, bottom: 24, textAlign: 'center', pointerEvents: 'none',
            display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'center',
            textShadow: '0 1px 3px #000, 0 0 12px rgb(0 0 0 / 0.6)',
          }}>
            <span style={{ fontSize: 20, lineHeight: 1.35, fontWeight: 500, color: '#fff' }}>
              {dub ? (current.translated_text || current.source_text) : current.source_text}
            </span>
            {dub && current.translated_text && (
              <span style={{ fontSize: 13, color: '#d9d2ee' }}>{current.source_text}</span>
            )}
          </div>
        )}
      </div>

      <div style={{
        width: '100%', maxWidth: 960, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10,
              }}>
        <button type="button" style={roundBtn} onClick={() => step(-1)} aria-label="Previous line" title="Previous line (↑)">
          <Icon name="prev" />
        </button>
        <button
          type="button"
          onClick={() => setPlaying(!playing)}
          aria-label={playing ? 'Pause' : 'Play'}
          title={playing ? 'Pause (Space)' : 'Play (Space)'}
          style={{ ...roundBtn, width: 52, height: 52, border: 'none', background: 'var(--text)', color: 'var(--bg)' }}
        >
          <Icon name={playing ? 'pause' : 'play'} size={18} />
        </button>
        <button type="button" style={roundBtn} onClick={() => step(1)} aria-label="Next line" title="Next line (↓)">
          <Icon name="next" />
        </button>
        <span className="timecode" style={{ fontSize: 13, color: 'var(--text)', marginLeft: 4, whiteSpace: 'nowrap' }}>
          {formatTime(playhead, true)} <span style={{ color: 'var(--text-dim)' }}>/ {formatTime(duration, true)}</span>
        </span>
        <div style={{ flex: 1 }} />
        <Select
          value={String(rate)}
          onChange={(v) => setRate(Number(v))}
          options={RATES.map((r) => ({ value: String(r), label: `${r}×` }))}
          style={{ width: 66 }}
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
          style={{ width: 60, accentColor: 'var(--accent)' }}
        />
      </div>
    </div>
  )
}
