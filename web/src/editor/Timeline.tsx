// Timeline — the canvas centrepiece of the editor (docs/plans/editor-redesign.md §5).
//   TimelineToolbar (32px, DOM)
//   track headers (120px, DOM) | canvas: ruler + scene strip / V filmstrip / ORIGINAL / speakers / DUB
// A single rAF loop reads the store imperatively (useStore.getState()) and redraws only when a
// relevant snapshot changes; pointer + wheel events drive store actions. Drawing, layout and
// hit-testing are pure functions in timelineDraw.ts.
import { useCallback, useEffect, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, ReactNode } from 'react'
import { api } from '../api/client'
import { Icon } from '../components/Icon'
import { IconButton, SpeakerChip } from '../components/primitives'
import { speakerColor, stageStatus, useStore } from '../state/store'
import type { Anchor, Project } from '../types'
import { formatTime } from '../types'
import { TimelineToolbar, type TimelineTool } from './TimelineToolbar'
import {
  LANES_TOP,
  anchorSpanAt,
  computeLayout,
  drawTimeline,
  hitTest,
  resolveSnap,
  resolveTheme,
  snapCandidates,
  visibleAnchors,
  type DragOverride,
  type Hit,
  type Layout,
  type SkipOverride,
  type Theme,
} from './timelineDraw'
import { FIT_EVENT, timelinePointer } from './useKeyboard'

export const HEADER_W = 120
const MIN_SEG_DURATION = 0.2
const MIN_SKIP_DURATION = 0.1
const round3 = (v: number): number => Math.round(v * 1000) / 1000

type Drag =
  | { mode: 'scrub'; pointerId: number }
  | {
      mode: 'trim'
      pointerId: number
      segmentId: string
      side: 'start' | 'end'
      start: number
      end: number
      moved: boolean
      snaps: number[]
    }
  | {
      mode: 'skipEdge'
      pointerId: number
      rangeId: string
      side: 'start' | 'end'
      start: number
      end: number
      moved: boolean
      snaps: number[]
    }
  | { mode: 'rangeHandle'; pointerId: number; side: 'start' | 'end'; snaps: number[] }
  | { mode: 'rangeCreate'; pointerId: number; anchor: number; moved: boolean; snaps: number[] }

export function Timeline() {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tcRef = useRef<HTMLSpanElement>(null)
  const sizeRef = useRef({ w: 0, h: 0 })
  const themeRef = useRef<Theme | null>(null)
  const dragRef = useRef<Drag | null>(null)
  /** Trim results kept as draw overrides until the server confirms (avoids a snap-back flash). */
  const pendingRef = useRef<DragOverride | null>(null)
  const pendingSkipRef = useRef<SkipOverride | null>(null)
  /** Skimmer x (canvas px) while the pointer is over the canvas. */
  const skimmerRef = useRef<number | null>(null)
  /** Snap guide (seconds) while a drag is snapped to something. */
  const snapRef = useRef<number | null>(null)
  /** True after a wheel pan/zoom during playback: pauses playhead auto-follow
   *  until the playhead scrolls back into view, the user seeks, or playback stops. */
  const userPannedRef = useRef(false)
  const suppressClickRef = useRef(false)
  /** Bumped whenever a ref that affects drawing changes; part of the redraw snapshot. */
  const versionRef = useRef(0)
  const lastSnapRef = useRef<unknown[] | null>(null)
  /** Project with speaker colours swapped for the active theme's palette (cached per project). */
  const paletteRef = useRef<{ source: Project; themed: Project } | null>(null)
  const filmImageRef = useRef<HTMLImageElement | null>(null)
  const anchorsCacheRef = useRef<{ key: unknown[]; list: Anchor[] } | null>(null)

  const [tool, setTool] = useState<TimelineTool>('select')
  const toolRef = useRef<TimelineTool>('select')
  toolRef.current = tool
  /** Canvas height, mirrored into React state so the DOM header column can share the layout. */
  const [canvasH, setCanvasH] = useState(0)

  const project = useStore((s) => s.project)
  const clipHeight = useStore((s) => s.clipHeight)
  const audioTrack = useStore((s) => s.audioTrack)
  const setAudioTrack = useStore((s) => s.setAudioTrack)
  const filmstrip = useStore((s) => s.filmstrip)

  const bump = (): void => {
    versionRef.current++
  }

  const localPos = (e: { clientX: number; clientY: number }): { x: number; y: number } => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  const activeOverride = (): DragOverride | null => {
    const d = dragRef.current
    if (d?.mode === 'trim') return { segmentId: d.segmentId, start: d.start, end: d.end }
    return pendingRef.current
  }
  const activeSkipOverride = (): SkipOverride | null => {
    const d = dragRef.current
    if (d?.mode === 'skipEdge') return { rangeId: d.rangeId, start: d.start, end: d.end }
    return pendingSkipRef.current
  }

  /** Anchors that pass the threshold (cached per anchors/threshold/zoom-band). */
  const currentAnchors = (): Anchor[] => {
    const st = useStore.getState()
    const all = st.anchors?.anchors ?? []
    const key = [all, st.anchorThreshold, st.zoom > 20]
    const c = anchorsCacheRef.current
    if (c && c.key.every((v, i) => v === key[i])) return c.list
    const list = visibleAnchors(all, st.anchorThreshold, st.zoom)
    anchorsCacheRef.current = { key, list }
    return list
  }

  const hitAt = (x: number, y: number): Hit => {
    const st = useStore.getState()
    const p = st.project
    if (!p?.media) return { kind: 'none' }
    const layout = computeLayout(sizeRef.current.w, sizeRef.current.h, p.speakers, st.clipHeight)
    return hitTest(layout, p, x, y, {
      zoom: st.zoom, scrollX: st.scrollX, range: st.range,
      override: activeOverride(), skipOverride: activeSkipOverride(),
    })
  }

  const timeAt = (x: number): number => {
    const st = useStore.getState()
    return st.scrollX + x / st.zoom
  }

  /** Snap `t` against the drag's candidates (when snapping is on); updates the guide. */
  const snapped = (t: number, snaps: number[]): number => {
    const st = useStore.getState()
    if (!st.snapping || snaps.length === 0) {
      snapRef.current = null
      return t
    }
    const r = resolveSnap(t, snaps, st.zoom)
    snapRef.current = r.snapped ? r.t : null
    return r.t
  }

  const candidates = (exclude: { segmentId?: string; skipRangeId?: string } = {}): number[] => {
    const st = useStore.getState()
    if (!st.snapping || !st.project) return []
    return snapCandidates(st.project, currentAnchors(), st.playhead, exclude)
  }

  const capture = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    try { canvasRef.current?.setPointerCapture(e.pointerId) } catch { /* pointer already gone */ }
  }

  // ---------------------------------------------------------- pointer -------

  const onPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    if (e.button !== 0) return
    const canvas = canvasRef.current
    if (!canvas) return
    const st = useStore.getState()
    const p = st.project
    if (!p?.media) return
    suppressClickRef.current = false
    const { x, y } = localPos(e)
    const hit = hitAt(x, y)
    const onLanes = y >= LANES_TOP
    const rangeTool = toolRef.current === 'range'

    if (hit.kind === 'rangeHandle') {
      dragRef.current = { mode: 'rangeHandle', pointerId: e.pointerId, side: hit.side, snaps: candidates() }
      capture(e)
    } else if (hit.kind === 'skipEdge') {
      const r = (p.skip_ranges ?? []).find((sr) => sr.id === hit.rangeId)
      if (!r) return
      st.selectSkipRange(r.id)
      dragRef.current = {
        mode: 'skipEdge', pointerId: e.pointerId, rangeId: r.id, side: hit.side,
        start: r.start, end: r.end, moved: false, snaps: candidates({ skipRangeId: r.id }),
      }
      capture(e)
    } else if (hit.kind === 'edge' && !rangeTool) {
      const seg = p.segments.find((sg) => sg.id === hit.segmentId)
      if (!seg) return
      dragRef.current = {
        mode: 'trim', pointerId: e.pointerId, segmentId: seg.id, side: hit.side,
        start: seg.start, end: seg.end, moved: false, snaps: candidates({ segmentId: seg.id }),
      }
      capture(e)
    } else if (rangeTool && onLanes && hit.kind !== 'scene') {
      const t = snapped(timeAt(x), candidates())
      dragRef.current = { mode: 'rangeCreate', pointerId: e.pointerId, anchor: t, moved: false, snaps: candidates() }
      capture(e)
    } else if (hit.kind === 'ruler' || hit.kind === 'film' || hit.kind === 'wave' || hit.kind === 'dub') {
      // seek + scrub; the playhead handle lives in the ruler, so grabbing it lands here too
      dragRef.current = { mode: 'scrub', pointerId: e.pointerId }
      capture(e)
      st.seek(timeAt(x))
    }
    // segment / skip / lane / scene presses are handled by onClick / onDoubleClick
    skimmerRef.current = null
    timelinePointer.time = null
    bump()
  }

  const onPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    const canvas = canvasRef.current
    if (!canvas) return
    const st = useStore.getState()
    const p = st.project
    if (!p?.media) return
    const { x, y } = localPos(e)
    const drag = dragRef.current
    const dur = p.media.duration

    if (drag && drag.pointerId === e.pointerId) {
      switch (drag.mode) {
        case 'scrub': {
          st.seek(timeAt(x))
          break
        }
        case 'trim': {
          // live trim preview: min duration 0.2s, clamp to [0, media duration], no neighbour clamping
          const t = snapped(timeAt(x), drag.snaps)
          if (drag.side === 'start') drag.start = Math.max(0, Math.min(drag.end - MIN_SEG_DURATION, t))
          else drag.end = Math.min(dur, Math.max(drag.start + MIN_SEG_DURATION, t))
          drag.moved = true
          break
        }
        case 'skipEdge': {
          const t = snapped(timeAt(x), drag.snaps)
          if (drag.side === 'start') drag.start = Math.max(0, Math.min(drag.end - MIN_SKIP_DURATION, t))
          else drag.end = Math.min(dur, Math.max(drag.start + MIN_SKIP_DURATION, t))
          drag.moved = true
          break
        }
        case 'rangeHandle': {
          const r = st.range
          if (!r) break
          const t = Math.max(0, Math.min(dur, snapped(timeAt(x), drag.snaps)))
          if (drag.side === 'start') st.setRange({ start: Math.min(t, r.end - 0.05), end: r.end })
          else st.setRange({ start: r.start, end: Math.max(t, r.start + 0.05) })
          break
        }
        case 'rangeCreate': {
          const t = Math.max(0, Math.min(dur, snapped(timeAt(x), drag.snaps)))
          if (Math.abs(t - drag.anchor) * st.zoom >= 2) {
            drag.moved = true
            st.setRange({ start: Math.min(t, drag.anchor), end: Math.max(t, drag.anchor) })
          }
          break
        }
      }
      bump()
      return
    }

    // idle hover: cursor affordances + skimmer
    const hit = hitAt(x, y)
    const rangeTool = toolRef.current === 'range' && y >= LANES_TOP
    canvas.style.cursor =
      hit.kind === 'edge' || hit.kind === 'skipEdge' || hit.kind === 'rangeHandle' ? 'ew-resize'
        : rangeTool ? 'crosshair'
          : hit.kind === 'segment' || hit.kind === 'skip' || hit.kind === 'scene' ? 'pointer'
            : 'default'
    const t = timeAt(x)
    timelinePointer.time = t >= 0 && t <= dur ? t : null
    const prev = skimmerRef.current
    if (prev === null || Math.abs(prev - x) >= 0.5) {
      skimmerRef.current = x
      bump()
    }
  }

  const onPointerUp = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== e.pointerId) return
    dragRef.current = null
    snapRef.current = null
    const st = useStore.getState()
    if (drag.mode === 'trim') {
      suppressClickRef.current = drag.moved
      if (drag.moved) {
        const start = round3(drag.start)
        const end = round3(drag.end)
        const ov: DragOverride = { segmentId: drag.segmentId, start, end }
        pendingRef.current = ov
        // fire-and-forget persist; keep the preview until the store reflects the change
        st.updateSegment(drag.segmentId, { start, end })
          .catch((err: unknown) => {
            console.error('Timeline: failed to save segment trim', err)
            st.toast('error', 'Failed to save the trim')
          })
          .finally(() => {
            if (pendingRef.current === ov) {
              pendingRef.current = null
              bump()
            }
          })
      }
    } else if (drag.mode === 'skipEdge') {
      suppressClickRef.current = true
      if (drag.moved) {
        const start = round3(drag.start)
        const end = round3(drag.end)
        const ov: SkipOverride = { rangeId: drag.rangeId, start, end }
        pendingSkipRef.current = ov
        st.updateSkipRange(drag.rangeId, { start, end })
          .catch((err: unknown) => {
            console.error('Timeline: failed to save skip range', err)
            st.toast('error', 'Failed to save the excluded range')
          })
          .finally(() => {
            if (pendingSkipRef.current === ov) {
              pendingSkipRef.current = null
              bump()
            }
          })
      }
    } else if (drag.mode === 'rangeCreate') {
      suppressClickRef.current = drag.moved
    } else {
      suppressClickRef.current = true // a scrub / handle drag is never a select-click
    }
    bump()
    try { canvasRef.current?.releasePointerCapture(e.pointerId) } catch { /* already released */ }
  }

  const onPointerCancel = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    const drag = dragRef.current
    if (drag && drag.pointerId === e.pointerId) {
      dragRef.current = null // abandon without persisting
      snapRef.current = null
      bump()
    }
  }

  const onPointerLeave = (): void => {
    timelinePointer.time = null
    if (dragRef.current) return
    if (skimmerRef.current !== null) {
      skimmerRef.current = null
      bump()
    }
    if (canvasRef.current) canvasRef.current.style.cursor = 'default'
  }

  const onClick = (e: ReactMouseEvent<HTMLCanvasElement>): void => {
    if (suppressClickRef.current) {
      suppressClickRef.current = false
      return
    }
    const st = useStore.getState()
    const p = st.project
    if (!p?.media) return
    const { x, y } = localPos(e)
    const hit = hitAt(x, y)
    switch (hit.kind) {
      case 'segment':
      case 'edge':
        st.selectSegment(hit.segmentId)
        break
      case 'skip':
        st.selectSkipRange(hit.rangeId)
        break
      case 'scene': {
        const span = anchorSpanAt(currentAnchors(), timeAt(x), p.media.duration)
        const cur = st.range
        st.setRange(e.shiftKey && cur
          ? { start: Math.min(cur.start, span.start), end: Math.max(cur.end, span.end) }
          : span)
        break
      }
      case 'lane':
        st.selectSegment(null)
        st.selectSkipRange(null)
        break
      default:
        break
    }
  }

  const onDoubleClick = (e: ReactMouseEvent<HTMLCanvasElement>): void => {
    const { x, y } = localPos(e)
    const hit = hitAt(x, y)
    if (hit.kind === 'segment' || hit.kind === 'edge') {
      useStore.getState().selectSegment(hit.segmentId, true)
    }
  }

  // ------------------------------------------------------------- fit --------

  const fit = useCallback((): void => {
    const st = useStore.getState()
    const dur = st.project?.media?.duration
    const w = sizeRef.current.w
    if (!dur || dur <= 0 || w <= 0) return
    st.setZoom(w / dur)
    st.setScrollX(0)
  }, [])

  useEffect(() => {
    window.addEventListener(FIT_EVENT, fit)
    return () => window.removeEventListener(FIT_EVENT, fit)
  }, [fit])

  // ------------------------------------------------------ filmstrip image ---

  useEffect(() => {
    filmImageRef.current = null
    versionRef.current++
    if (!filmstrip || !project) return
    const img = new Image()
    let cancelled = false
    img.onload = () => {
      if (cancelled) return
      filmImageRef.current = img
      versionRef.current++
    }
    img.src = /^(https?:)?\//.test(filmstrip.url) ? filmstrip.url : api.mediaUrl(project.id, filmstrip.url)
    return () => { cancelled = true }
  }, [filmstrip, project?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // ----------------------------------------------- resize + wheel + rAF -----

  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return
    themeRef.current = resolveTheme()
    // theme toggle: CSS variables change synchronously in setTheme, so re-resolve and repaint
    const unsubTheme = useStore.subscribe((st, prev) => {
      if (st.theme !== prev.theme) {
        themeRef.current = resolveTheme()
        paletteRef.current = null
        versionRef.current++
      }
    })

    const setSize = (w: number, h: number): void => {
      sizeRef.current = { w, h }
      setCanvasH((prev) => (prev === h ? prev : h))
      versionRef.current++
    }
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setSize(entry.contentRect.width, entry.contentRect.height)
    })
    ro.observe(container)
    const rect0 = container.getBoundingClientRect()
    setSize(rect0.width, rect0.height)

    // canvas text metrics change once the web font arrives
    void document.fonts.ready.then(() => { versionRef.current++ }).catch(() => undefined)

    // wheel must be non-passive to preventDefault (React registers it passively)
    const onWheel = (e: WheelEvent): void => {
      const st = useStore.getState()
      if (!st.project?.media) return
      e.preventDefault()
      const x = e.clientX - canvas.getBoundingClientRect().left
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? sizeRef.current.h || 400 : 1
      const dx = e.deltaX * unit
      const dy = e.deltaY * unit
      if (e.ctrlKey || e.metaKey) {
        // zoom anchored at the cursor
        st.setZoom(st.zoom * Math.exp(-dy * 0.002), st.scrollX + x / st.zoom, x)
      } else {
        // pan along the dominant axis
        const d = Math.abs(dx) > Math.abs(dy) ? dx : dy
        st.setScrollX(st.scrollX + d / st.zoom)
      }
      // a deliberate viewport change during playback pauses auto-follow
      if (st.playing) userPannedRef.current = true
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })

    const ctx = canvas.getContext('2d', { alpha: false })
    let raf = 0
    let lastSeekNonce = useStore.getState().seekRequest?.nonce ?? 0
    let lastTc = ''
    const frame = (): void => {
      raf = requestAnimationFrame(frame)
      const { w, h } = sizeRef.current
      if (w < 2 || h < 2 || !ctx || !themeRef.current) return
      const dpr = window.devicePixelRatio || 1

      // auto-follow during playback (paused while the user is dragging or has wheel-panned away)
      const pre = useStore.getState()
      if (pre.seekRequest && pre.seekRequest.nonce !== lastSeekNonce) {
        lastSeekNonce = pre.seekRequest.nonce
        userPannedRef.current = false // an explicit seek re-enables following
      }
      if (pre.playing && pre.project?.media && !dragRef.current) {
        const px = (pre.playhead - pre.scrollX) * pre.zoom
        if (px >= 0 && px <= 0.82 * w) userPannedRef.current = false // playhead back in view
        if ((px > 0.82 * w || px < 0) && !userPannedRef.current) {
          pre.setScrollX(pre.playhead - (0.15 * w) / pre.zoom)
        }
      } else if (!pre.playing) {
        userPannedRef.current = false
      }
      const st = useStore.getState() // re-read: auto-follow may have moved scrollX

      // header timecode (DOM text, no React render)
      const tc = formatTime(st.playhead, true)
      if (tc !== lastTc && tcRef.current) {
        tcRef.current.textContent = tc
        lastTc = tc
      }

      // redraw only when a relevant snapshot changed
      const snap: unknown[] = [
        st.project, st.waveforms, st.playhead, st.zoom, st.scrollX, st.selection, st.skipSelection,
        st.range, st.anchors, st.anchorThreshold, st.filmstrip, st.clipHeight, st.playing,
        w, h, dpr, versionRef.current,
      ]
      const last = lastSnapRef.current
      if (last && last.length === snap.length && last.every((v, i) => v === snap[i])) return
      lastSnapRef.current = snap

      const bw = Math.round(w * dpr)
      const bh = Math.round(h * dpr)
      if (canvas.width !== bw) canvas.width = bw
      if (canvas.height !== bh) canvas.height = bh

      const spk = themeRef.current.spk
      if (st.project && paletteRef.current?.source !== st.project) {
        paletteRef.current = {
          source: st.project,
          themed: {
            ...st.project,
            speakers: st.project.speakers.map((sp, i) => ({ ...sp, color: spk[i % spk.length] })),
          },
        }
      }
      drawTimeline(ctx, themeRef.current, {
        width: w,
        height: h,
        dpr,
        project: st.project ? paletteRef.current!.themed : null,
        clipHeight: st.clipHeight,
        waveMain: st.waveforms.vocals ?? st.waveforms.original ?? null,
        waveDub: st.waveforms.dub_mix ?? null,
        filmstrip: st.filmstrip,
        filmImage: filmImageRef.current,
        anchors: currentAnchors(),
        zoom: st.zoom,
        scrollX: st.scrollX,
        playhead: st.playhead,
        selection: st.selection,
        skipSelection: st.skipSelection,
        range: st.range,
        skimmerX: dragRef.current ? null : skimmerRef.current,
        snapT: snapRef.current,
        override: activeOverride(),
        skipOverride: activeSkipOverride(),
      })
    }
    raf = requestAnimationFrame(frame)

    return () => {
      cancelAnimationFrame(raf)
      unsubTheme()
      ro.disconnect()
      canvas.removeEventListener('wheel', onWheel)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const layout: Layout | null = project?.media
    ? computeLayout(HEADER_W, canvasH, project.speakers, clipHeight)
    : null
  const mixDone = stageStatus(project, 'mix') === 'done'

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg-sunk)' }}>
      <TimelineToolbar tool={tool} onToolChange={setTool} onFit={fit} />
      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
        <div
          aria-label="Track headers"
          style={{
            width: HEADER_W, flexShrink: 0, position: 'relative', overflow: 'hidden',
            background: 'var(--bg-panel)', borderRight: '1px solid var(--border)', userSelect: 'none',
          }}
        >
          <HeaderRow top={0} height={LANES_TOP} raised>
            <span ref={tcRef} className="timecode" style={{ fontSize: 10, color: 'var(--text)' }}>
              {formatTime(0, true)}
            </span>
          </HeaderRow>
          {layout && (
            <>
              <HeaderRow top={layout.filmTop} height={layout.filmH}>
                <Icon name="film" size={12} style={{ color: 'var(--text-faint)' }} />
                <HeaderLabel>V</HeaderLabel>
              </HeaderRow>
              <HeaderRow top={layout.waveTop} height={layout.waveH}>
                <HeaderLabel>Original</HeaderLabel>
                <div style={{ flex: 1 }} />
                <IconButton
                  name="volume" size={20} iconSize={12}
                  title="Monitor original audio (1)"
                  active={audioTrack === 'original'}
                  onClick={() => setAudioTrack('original')}
                />
              </HeaderRow>
              {layout.speakerLanes.map((lane) => {
                const sp = project!.speakers.find((s) => s.id === lane.speakerId)
                return (
                  <HeaderRow key={lane.speakerId} top={lane.top} height={lane.height}>
                    <SpeakerChip name={sp?.name || lane.speakerId} color={speakerColor(project, lane.speakerId)} size="sm" />
                  </HeaderRow>
                )
              })}
              <HeaderRow top={layout.dubTop} height={layout.dubH}>
                <HeaderLabel>Dub</HeaderLabel>
                <div style={{ flex: 1 }} />
                <IconButton
                  name="volume" size={20} iconSize={12}
                  title={mixDone ? 'Monitor dub mix (2)' : 'Monitor dub mix (run Mix first)'}
                  active={audioTrack === 'dub'}
                  disabled={!mixDone}
                  onClick={() => setAudioTrack('dub')}
                />
              </HeaderRow>
            </>
          )}
        </div>
        <div
          ref={containerRef}
          style={{ flex: 1, minWidth: 0, position: 'relative', overflow: 'hidden', background: 'var(--bg-sunk)' }}
        >
          <canvas
            ref={canvasRef}
            role="img"
            aria-label="Timeline"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerCancel}
            onPointerLeave={onPointerLeave}
            onClick={onClick}
            onDoubleClick={onDoubleClick}
            style={{ display: 'block', width: '100%', height: '100%', touchAction: 'none' }}
          />
        </div>
      </div>
    </div>
  )
}

function HeaderRow({ top, height, raised, children }: {
  top: number; height: number; raised?: boolean; children: ReactNode
}) {
  return (
    <div style={{
      position: 'absolute', left: 0, right: 0, top, height, display: 'flex', alignItems: 'center', gap: 6,
      padding: '0 6px 0 10px', borderBottom: `1px solid ${raised ? 'var(--border)' : 'var(--border-subtle)'}`,
      background: raised ? 'var(--bg-raised)' : 'transparent', overflow: 'hidden',
    }}>
      {children}
    </div>
  )
}

function HeaderLabel({ children }: { children: ReactNode }) {
  return (
    <span className="eyebrow" style={{ fontSize: 10, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
      {children}
    </span>
  )
}
