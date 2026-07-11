// Video player: plays playback.mp4, keeps a hidden dub_mix.wav <audio> element in sync,
// and drives store.playhead / consumes store.seekRequest / store.playing / store.audioTrack.
import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { Button, Select } from '../components/primitives'
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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <div
        style={{
          flex: 1, minHeight: 0, background: '#000', display: 'flex',
          alignItems: 'center', justifyContent: 'center', overflow: 'hidden',
        }}
      >
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
            src={`${api.mediaUrl(pid, 'audio/dub_mix.wav')}${mixUpdatedAt ? `?v=${encodeURIComponent(mixUpdatedAt)}` : ''}`}
            style={{ display: 'none' }}
          />
        )}
      </div>

      <div
        style={{
          height: 36, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10,
          padding: '0 12px', borderTop: '1px solid var(--border)',
        }}
      >
        <Button
          variant="primary"
          onClick={() => setPlaying(!playing)}
          title={playing ? 'Pause (Space)' : 'Play (Space)'}
          style={{ padding: '4px 10px' }}
        >
          {playing ? '⏸' : '▶'}
        </Button>
        <span className="timecode">
          {formatTime(playhead, true)} / {formatTime(duration, true)}
        </span>
        <div style={{ flex: 1 }} />
        <Select
          value={String(rate)}
          onChange={(v) => setRate(Number(v))}
          options={RATES.map((r) => ({ value: String(r), label: `${r}×` }))}
          style={{ width: 68 }}
        />
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={volume}
          onChange={(e) => setVolume(Number(e.target.value))}
          title={`Volume ${Math.round(volume * 100)}%`}
          style={{ width: 90, accentColor: 'var(--accent)' }}
        />
      </div>
    </div>
  )
}
