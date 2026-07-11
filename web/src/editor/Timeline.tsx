// Timeline — the canvas centerpiece of the editor.
// Vertical lane stack: time ruler / original waveform / one lane per speaker / dub waveform.
// A single rAF loop reads the store imperatively (useStore.getState()) and redraws only
// when a relevant snapshot changes; pointer + wheel events drive store actions.
import { useEffect, useRef } from 'react'
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react'
import { useStore } from '../state/store'
import {
  RULER_H,
  computeLayout,
  drawTimeline,
  hitTest,
  resolveTheme,
  type DragOverride,
  type Hit,
  type Theme,
} from './timelineDraw'

const MIN_SEG_DURATION = 0.2
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
    }

export function Timeline() {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sizeRef = useRef({ w: 0, h: 0 })
  const themeRef = useRef<Theme | null>(null)
  const dragRef = useRef<Drag | null>(null)
  /** Trim result kept as a draw override until the server confirms (avoids a snap-back flash). */
  const pendingRef = useRef<DragOverride | null>(null)
  const hoverRef = useRef<{ x: number; y: number } | null>(null)
  /** True after a wheel pan/zoom during playback: pauses playhead auto-follow
   *  until the playhead scrolls back into view, the user seeks, or playback stops. */
  const userPannedRef = useRef(false)
  const suppressClickRef = useRef(false)
  /** Bumped whenever a ref that affects drawing changes; part of the redraw snapshot. */
  const versionRef = useRef(0)
  const lastSnapRef = useRef<unknown[] | null>(null)

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

  const hitAt = (x: number, y: number): Hit => {
    const st = useStore.getState()
    const p = st.project
    if (!p?.media) return { kind: 'none' }
    const layout = computeLayout(sizeRef.current.w, sizeRef.current.h, p.speakers)
    return hitTest(layout, p, x, y, st.zoom, st.scrollX, activeOverride())
  }

  // ---------------------------------------------------------- pointer -------

  const onPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    if (e.button !== 0) return
    const canvas = canvasRef.current
    if (!canvas) return
    const st = useStore.getState()
    if (!st.project?.media) return
    suppressClickRef.current = false
    const { x, y } = localPos(e)
    const hit = hitAt(x, y)

    if (hit.kind === 'ruler' || hit.kind === 'wave' || hit.kind === 'dub') {
      // seek + scrub; the playhead handle lives in the ruler, so grabbing it lands here too
      dragRef.current = { mode: 'scrub', pointerId: e.pointerId }
      try { canvas.setPointerCapture(e.pointerId) } catch { /* pointer already gone */ }
      hoverRef.current = y < RULER_H ? { x, y } : null
      st.seek(st.scrollX + x / st.zoom)
      bump()
    } else if (hit.kind === 'edge') {
      const seg = st.project.segments.find((sg) => sg.id === hit.segmentId)
      if (!seg) return
      dragRef.current = {
        mode: 'trim',
        pointerId: e.pointerId,
        segmentId: seg.id,
        side: hit.side,
        start: seg.start,
        end: seg.end,
        moved: false,
      }
      try { canvas.setPointerCapture(e.pointerId) } catch { /* pointer already gone */ }
      bump()
    }
    // segment-body presses are handled by onClick / onDoubleClick
  }

  const onPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    const canvas = canvasRef.current
    if (!canvas) return
    const st = useStore.getState()
    const p = st.project
    if (!p?.media) return
    const { x, y } = localPos(e)
    const drag = dragRef.current

    if (drag && drag.pointerId === e.pointerId) {
      if (drag.mode === 'scrub') {
        st.seek(st.scrollX + x / st.zoom)
        hoverRef.current = y >= 0 && y < RULER_H ? { x, y } : null
        bump()
      } else {
        // live trim preview: min duration 0.2s, clamp to [0, media duration], no neighbor clamping
        const t = st.scrollX + x / st.zoom
        if (drag.side === 'start') {
          drag.start = Math.max(0, Math.min(drag.end - MIN_SEG_DURATION, t))
        } else {
          drag.end = Math.min(p.media.duration, Math.max(drag.start + MIN_SEG_DURATION, t))
        }
        drag.moved = true
        bump()
      }
      return
    }

    // idle hover: cursor affordances + ruler ghost line
    const hit = hitAt(x, y)
    canvas.style.cursor =
      hit.kind === 'edge' ? 'ew-resize' : hit.kind === 'segment' ? 'pointer' : 'default'
    const prev = hoverRef.current
    if (hit.kind === 'ruler') {
      if (!prev || Math.abs(prev.x - x) >= 0.5 || Math.abs(prev.y - y) >= 2) {
        hoverRef.current = { x, y }
        bump()
      }
    } else if (prev) {
      hoverRef.current = null
      bump()
    }
  }

  const onPointerUp = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== e.pointerId) return
    dragRef.current = null
    if (drag.mode === 'trim') {
      suppressClickRef.current = drag.moved
      if (drag.moved) {
        const st = useStore.getState()
        const start = round3(drag.start)
        const end = round3(drag.end)
        const ov: DragOverride = { segmentId: drag.segmentId, start, end }
        pendingRef.current = ov
        // fire-and-forget persist; keep the preview until the store reflects the change
        st.updateSegment(drag.segmentId, { start, end })
          .catch((err: unknown) => {
            console.error('Timeline: failed to save segment trim', err)
          })
          .finally(() => {
            if (pendingRef.current === ov) {
              pendingRef.current = null
              bump()
            }
          })
      }
    } else {
      suppressClickRef.current = true // a scrub is never a select-click
    }
    bump()
    try { canvasRef.current?.releasePointerCapture(e.pointerId) } catch { /* already released */ }
  }

  const onPointerCancel = (e: ReactPointerEvent<HTMLCanvasElement>): void => {
    const drag = dragRef.current
    if (drag && drag.pointerId === e.pointerId) {
      dragRef.current = null // abandon without persisting
      bump()
    }
  }

  const onPointerLeave = (): void => {
    if (dragRef.current) return
    if (hoverRef.current) {
      hoverRef.current = null
      bump()
    }
    if (canvasRef.current) canvasRef.current.style.cursor = 'default'
  }

  const onClick = (e: ReactMouseEvent<HTMLCanvasElement>): void => {
    if (suppressClickRef.current) {
      suppressClickRef.current = false
      return
    }
    const { x, y } = localPos(e)
    const hit = hitAt(x, y)
    if (hit.kind === 'segment' || hit.kind === 'edge') {
      useStore.getState().selectSegment(hit.segmentId)
    }
  }

  const onDoubleClick = (e: ReactMouseEvent<HTMLCanvasElement>): void => {
    const { x, y } = localPos(e)
    const hit = hitAt(x, y)
    if (hit.kind === 'segment' || hit.kind === 'edge') {
      useStore.getState().selectSegment(hit.segmentId, true)
    }
  }

  // ----------------------------------------------- resize + wheel + rAF -----

  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return
    themeRef.current = resolveTheme()

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        sizeRef.current = { w: entry.contentRect.width, h: entry.contentRect.height }
      }
      versionRef.current++
    })
    ro.observe(container)
    const rect0 = container.getBoundingClientRect()
    sizeRef.current = { w: rect0.width, h: rect0.height }

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

      // redraw only when a relevant snapshot changed
      const snap: unknown[] = [
        st.project, st.waveforms, st.playhead, st.zoom, st.scrollX,
        st.selection, st.playing, w, h, dpr, versionRef.current,
      ]
      const last = lastSnapRef.current
      if (last && last.length === snap.length && last.every((v, i) => v === snap[i])) return
      lastSnapRef.current = snap

      const bw = Math.round(w * dpr)
      const bh = Math.round(h * dpr)
      if (canvas.width !== bw) canvas.width = bw
      if (canvas.height !== bh) canvas.height = bh

      const d = dragRef.current
      drawTimeline(ctx, themeRef.current, {
        width: w,
        height: h,
        dpr,
        project: st.project,
        waveMain: st.waveforms.vocals ?? st.waveforms.original ?? null,
        waveDub: st.waveforms.dub_mix ?? null,
        zoom: st.zoom,
        scrollX: st.scrollX,
        playhead: st.playhead,
        selection: st.selection,
        rulerHover: hoverRef.current,
        override:
          d?.mode === 'trim'
            ? { segmentId: d.segmentId, start: d.start, end: d.end }
            : pendingRef.current,
      })
    }
    raf = requestAnimationFrame(frame)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      canvas.removeEventListener('wheel', onWheel)
    }
  }, [])

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        minHeight: 0,
        overflow: 'hidden',
        background: 'var(--bg)',
      }}
    >
      <canvas
        ref={canvasRef}
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
  )
}
