// timelineDraw.ts — pure canvas rendering, layout, and hit-testing for the editor Timeline.
// No React / store imports: everything arrives as arguments so the component can drive
// redraws from a single requestAnimationFrame loop and stay at 60fps.

import type { Project, Segment, Speaker, Waveform } from '../types'
import { formatTime } from '../types'

// ---------------------------------------------------------------- layout ----

export const RULER_H = 26
const WAVE_LANE_H = 56
const DUB_LANE_H = 56
const SPEAKER_LANE_H = 38
const SPEAKER_LANE_MIN_H = 26

export interface SpeakerLane {
  speakerId: string
  top: number
  height: number
}

export interface Layout {
  width: number
  height: number
  waveTop: number
  waveH: number
  speakerLanes: SpeakerLane[]
  dubTop: number
  dubH: number
  bottom: number
}

/** Vertical arrangement of lanes. Speaker lanes shrink (down to 26px) when they overflow. */
export function computeLayout(width: number, height: number, speakers: Speaker[]): Layout {
  const n = speakers.length
  let spkH = SPEAKER_LANE_H
  if (n > 0) {
    const avail = height - RULER_H - WAVE_LANE_H - DUB_LANE_H
    if (n * SPEAKER_LANE_H > avail) spkH = Math.max(SPEAKER_LANE_MIN_H, Math.floor(avail / n))
  }
  const speakerLanes: SpeakerLane[] = []
  let y = RULER_H + WAVE_LANE_H
  for (const s of speakers) {
    speakerLanes.push({ speakerId: s.id, top: y, height: spkH })
    y += spkH
  }
  return {
    width,
    height,
    waveTop: RULER_H,
    waveH: WAVE_LANE_H,
    speakerLanes,
    dubTop: y,
    dubH: DUB_LANE_H,
    bottom: y + DUB_LANE_H,
  }
}

// ----------------------------------------------------------------- theme ----

export interface Theme {
  bg: string
  bgRaised: string
  bgOverlay: string
  border: string
  borderStrong: string
  text: string
  textDim: string
  textFaint: string
  accent: string
  warn: string
  font: string
  mono: string
}

const FALLBACK: Theme = {
  bg: '#0d0f13',
  bgRaised: '#14171d',
  bgOverlay: '#1a1e26',
  border: '#232833',
  borderStrong: '#313848',
  text: '#e8eaf0',
  textDim: '#8b93a7',
  textFaint: '#5a6175',
  accent: '#e8604c',
  warn: '#e8b84c',
  font: "'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif",
  mono: "ui-monospace, 'SF Mono', 'Cascadia Code', Menlo, monospace",
}

/** Resolve design tokens from CSS variables once (falls back to the theme.css hexes). */
export function resolveTheme(): Theme {
  const cs = getComputedStyle(document.documentElement)
  const v = (name: string, fallback: string): string => {
    const raw = cs.getPropertyValue(name).trim()
    return raw !== '' ? raw : fallback
  }
  return {
    bg: v('--bg', FALLBACK.bg),
    bgRaised: v('--bg-raised', FALLBACK.bgRaised),
    bgOverlay: v('--bg-overlay', FALLBACK.bgOverlay),
    border: v('--border', FALLBACK.border),
    borderStrong: v('--border-strong', FALLBACK.borderStrong),
    text: v('--text', FALLBACK.text),
    textDim: v('--text-dim', FALLBACK.textDim),
    textFaint: v('--text-faint', FALLBACK.textFaint),
    accent: v('--accent', FALLBACK.accent),
    warn: v('--warn', FALLBACK.warn),
    font: v('--font', FALLBACK.font),
    mono: v('--mono', FALLBACK.mono),
  }
}

/** '#rgb' / '#rrggbb' → 'rgba(r, g, b, a)'. Non-hex inputs are returned unchanged. */
export function withAlpha(color: string, alpha: number): string {
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(color.trim())
  if (!m) return color
  let hex = m[1]
  if (hex.length === 3) hex = hex.split('').map((c) => c + c).join('')
  const r = parseInt(hex.slice(0, 2), 16)
  const g = parseInt(hex.slice(2, 4), 16)
  const b = parseInt(hex.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

/** Dim blue-gray for the original-audio waveform. */
const WAVE_COLOR = '#5b6b85'

// --------------------------------------------------------------- helpers ----

/** Align a coordinate so a 1px-CSS stroke lands crisply on device pixels. */
export function crisp(v: number, dpr: number): number {
  return (Math.round(v * dpr - 0.5) + 0.5) / dpr
}

const TICK_STEPS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300]

/** Smallest tick step whose major spacing is at least 70px at the given zoom. */
export function tickStep(zoom: number): number {
  return TICK_STEPS.find((s) => s * zoom >= 70) ?? 300
}

function hline(ctx: CanvasRenderingContext2D, x0: number, x1: number, y: number, dpr: number): void {
  const yy = crisp(y, dpr)
  ctx.beginPath()
  ctx.moveTo(x0, yy)
  ctx.lineTo(x1, yy)
  ctx.stroke()
}

function vline(ctx: CanvasRenderingContext2D, x: number, y0: number, y1: number, dpr: number): void {
  const xx = crisp(x, dpr)
  ctx.beginPath()
  ctx.moveTo(xx, y0)
  ctx.lineTo(xx, y1)
  ctx.stroke()
}

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
): void {
  ctx.beginPath()
  const rr = (ctx as { roundRect?: (x: number, y: number, w: number, h: number, r: number) => void }).roundRect
  if (typeof rr === 'function') {
    rr.call(ctx, x, y, w, h, r)
    return
  }
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

/** Truncate `text` with an ellipsis so it fits within maxW (approximate, then refined). */
function ellipsize(ctx: CanvasRenderingContext2D, text: string, maxW: number): string {
  if (maxW <= 8) return ''
  const full = ctx.measureText(text).width
  if (full <= maxW) return text
  const avg = full / Math.max(1, text.length)
  let n = Math.max(1, Math.floor(maxW / avg) - 1)
  let out = text.slice(0, n) + '…'
  while (n > 1 && ctx.measureText(out).width > maxW) {
    n = Math.max(1, Math.floor(n * 0.85))
    out = text.slice(0, n) + '…'
  }
  return out
}

// ------------------------------------------------------------ draw input ----

export interface DragOverride {
  segmentId: string
  start: number
  end: number
}

export interface HoverInfo {
  x: number
  y: number
}

export interface DrawInput {
  width: number
  height: number
  dpr: number
  project: Project | null
  /** vocals ?? original */
  waveMain: Waveform | null
  waveDub: Waveform | null
  zoom: number
  scrollX: number
  playhead: number
  selection: string | null
  /** Cursor position while hovering (or scrubbing) the ruler; drives ghost line + tooltip. */
  rulerHover: HoverInfo | null
  /** Live segment-trim preview (during drag, or pending server persist). */
  override: DragOverride | null
}

/** Per-frame render context shared by the draw helpers. */
interface RC {
  ctx: CanvasRenderingContext2D
  theme: Theme
  w: number
  h: number
  dpr: number
  zoom: number
  scrollX: number
}

// ------------------------------------------------------------- main draw ----

export function drawTimeline(ctx: CanvasRenderingContext2D, theme: Theme, s: DrawInput): void {
  const { width: w, height: h, dpr } = s
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.fillStyle = theme.bg
  ctx.fillRect(0, 0, w, h)

  const p = s.project
  if (!p?.media) return // no media: render nothing

  const rc: RC = { ctx, theme, w, h, dpr, zoom: s.zoom, scrollX: s.scrollX }
  const dur = p.media.duration
  const layout = computeLayout(w, h, p.speakers)
  const bottom = Math.min(layout.bottom, h)
  const step = tickStep(s.zoom)

  // lane background tints (alternating speaker lanes, raised dub lane)
  layout.speakerLanes.forEach((lane, i) => {
    if (i % 2 === 1) {
      ctx.fillStyle = withAlpha(theme.bgRaised, 0.45)
      ctx.fillRect(0, lane.top, w, lane.height)
    }
  })
  ctx.fillStyle = withAlpha(theme.bgRaised, 0.5)
  ctx.fillRect(0, layout.dubTop, w, layout.dubH)

  drawGrid(rc, dur, step, RULER_H, bottom)
  drawWaveLane(rc, layout.waveTop, layout.waveH, s.waveMain, WAVE_COLOR)
  drawWaveLane(rc, layout.dubTop, layout.dubH, s.waveDub, theme.accent)
  drawSegments(rc, p, layout, s.selection, s.override)

  // shade the area past the end of the media
  const endX = (dur - s.scrollX) * s.zoom
  if (endX < w) {
    const x0 = Math.max(0, endX)
    ctx.fillStyle = 'rgb(0 0 0 / 0.3)'
    ctx.fillRect(x0, RULER_H, w - x0, Math.max(0, bottom - RULER_H))
  }

  // lane separators
  ctx.strokeStyle = theme.border
  ctx.lineWidth = 1
  hline(ctx, 0, w, layout.waveTop + layout.waveH, dpr)
  for (const lane of layout.speakerLanes) hline(ctx, 0, w, lane.top + lane.height, dpr)
  if (layout.dubTop + layout.dubH <= h) hline(ctx, 0, w, layout.dubTop + layout.dubH, dpr)

  drawLanePills(rc, p, layout)
  drawRuler(rc, dur, step)
  if (p.segments.length === 0) drawEmptyHint(rc)
  drawPlayhead(rc, s.playhead, bottom)
  drawRulerHover(rc, s.rulerHover, bottom)
}

// ------------------------------------------------------------ sub-draws -----

function drawGrid(rc: RC, dur: number, step: number, yTop: number, yBottom: number): void {
  const { ctx, theme, dpr } = rc
  ctx.strokeStyle = withAlpha(theme.border, 0.55)
  ctx.lineWidth = 1
  const i0 = Math.max(0, Math.floor(rc.scrollX / step))
  const i1 = Math.ceil((rc.scrollX + rc.w / rc.zoom) / step)
  for (let i = i0; i <= i1; i++) {
    const t = i * step
    if (t > dur + 1e-6) break
    vline(ctx, (t - rc.scrollX) * rc.zoom, yTop, yBottom, dpr)
  }
}

function drawRuler(rc: RC, dur: number, step: number): void {
  const { ctx, theme, dpr, w } = rc
  ctx.fillStyle = theme.bgRaised
  ctx.fillRect(0, 0, w, RULER_H)
  ctx.strokeStyle = theme.border
  ctx.lineWidth = 1
  hline(ctx, 0, w, RULER_H, dpr)

  const minor = step / 5
  const i0 = Math.max(0, Math.floor(rc.scrollX / minor))
  const i1 = Math.ceil((rc.scrollX + w / rc.zoom) / minor)
  ctx.font = `10px ${theme.mono}`
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  for (let i = i0; i <= i1; i++) {
    const t = i * minor
    if (t > dur + 1e-6) break
    const x = (t - rc.scrollX) * rc.zoom
    const major = i % 5 === 0
    ctx.strokeStyle = major ? theme.borderStrong : theme.border
    vline(ctx, x, RULER_H - (major ? 9 : 5), RULER_H, dpr)
    if (major) {
      ctx.fillStyle = theme.textDim
      ctx.fillText(formatTime(t, step < 1), x + 4, 9)
    }
  }
}

/** Min/max waveform fill: one column per CSS pixel over the visible time range. */
function drawWaveLane(rc: RC, top: number, height: number, wf: Waveform | null, baseColor: string): void {
  const { ctx, dpr, w } = rc
  const mid = top + height / 2

  // center hairline (also the "no data yet" indicator)
  ctx.fillStyle = withAlpha(baseColor, 0.3)
  ctx.fillRect(0, Math.round(mid * dpr) / dpr, w, 1 / dpr)
  if (!wf || wf.peaks.length < 2 || wf.sample_rate <= 0) return

  const amp = Math.max(2, height / 2 - 4)
  const sr = wf.sample_rate
  const total = wf.peaks.length >> 1
  const minH = 1 / dpr
  ctx.fillStyle = withAlpha(baseColor, 0.65)
  for (let x = 0; x < w; x++) {
    const ta = rc.scrollX + x / rc.zoom
    const tb = rc.scrollX + (x + 1) / rc.zoom
    let i0 = Math.floor(ta * sr)
    let i1 = Math.max(i0 + 1, Math.ceil(tb * sr))
    if (i1 <= 0 || i0 >= total) continue
    if (i0 < 0) i0 = 0
    if (i1 > total) i1 = total
    let mn = Infinity
    let mx = -Infinity
    for (let i = i0; i < i1; i++) {
      const lo = wf.peaks[i * 2]
      const hi = wf.peaks[i * 2 + 1]
      if (lo < mn) mn = lo
      if (hi > mx) mx = hi
    }
    if (mn > mx) continue
    const y0 = mid - Math.min(1, Math.max(-1, mx)) * amp
    const y1 = mid - Math.min(1, Math.max(-1, mn)) * amp
    ctx.fillRect(x, y0, 1, Math.max(minH, y1 - y0))
  }
}

function drawSegments(
  rc: RC,
  p: Project,
  layout: Layout,
  selection: string | null,
  override: DragOverride | null,
): void {
  const laneById = new Map<string, SpeakerLane>()
  for (const l of layout.speakerLanes) laneById.set(l.speakerId, l)
  const colorById = new Map<string, string>()
  for (const sp of p.speakers) colorById.set(sp.id, sp.color)

  const t0 = rc.scrollX
  const t1 = rc.scrollX + rc.w / rc.zoom
  let selected: { seg: Segment; lane: SpeakerLane; start: number; end: number } | null = null

  for (const seg of p.segments) {
    const lane = laneById.get(seg.speaker_id)
    if (!lane || lane.top > rc.h) continue
    const ov = override && override.segmentId === seg.id ? override : null
    const start = ov ? ov.start : seg.start
    const end = ov ? ov.end : seg.end
    if (end < t0 || start > t1) continue // cull outside the visible range
    if (selection === seg.id) {
      selected = { seg, lane, start, end } // draw last, on top
      continue
    }
    drawSegmentBlock(rc, seg, colorById.get(seg.speaker_id) ?? rc.theme.textDim, start, end, lane, false)
  }
  if (selected) {
    drawSegmentBlock(
      rc, selected.seg,
      colorById.get(selected.seg.speaker_id) ?? rc.theme.textDim,
      selected.start, selected.end, selected.lane, true,
    )
  }
}

function drawSegmentBlock(
  rc: RC,
  seg: Segment,
  color: string,
  start: number,
  end: number,
  lane: SpeakerLane,
  selected: boolean,
): void {
  const { ctx, theme, dpr } = rc
  const x0 = (start - rc.scrollX) * rc.zoom
  const x1 = (end - rc.scrollX) * rc.zoom
  const hollow = seg.takes.length === 0
  const dirty = seg.translate_dirty || seg.synth_dirty
  const bw = Math.max(2, x1 - x0)
  const by = lane.top + 3
  const bh = lane.height - 6
  const r = Math.min(4, bw / 2, bh / 2)

  roundRectPath(ctx, crisp(x0, dpr), crisp(by, dpr), bw, bh, r)
  ctx.fillStyle = withAlpha(color, hollow ? (selected ? 0.1 : 0.05) : selected ? 0.3 : 0.18)
  ctx.fill()
  if (selected) {
    ctx.strokeStyle = theme.accent
    ctx.lineWidth = 1.5
  } else {
    ctx.strokeStyle = hollow ? withAlpha(color, 0.45) : color
    ctx.lineWidth = 1
  }
  ctx.stroke()
  ctx.lineWidth = 1

  // label — keep it visible when the block extends past the left edge
  const label = seg.translated_text || seg.source_text
  const textX = Math.max(x0, 0) + 6
  const room = Math.min(x1, rc.w) - textX - 8
  if (label && bw > 24 && room > 12) {
    ctx.font = `11px ${theme.font}`
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillStyle = hollow ? theme.textFaint : theme.text
    const shown = ellipsize(ctx, label, room)
    if (shown) ctx.fillText(shown, textX, lane.top + lane.height / 2 + 0.5)
  }

  // amber dirty dot, top-right
  if (dirty && bw > 12) {
    ctx.fillStyle = theme.warn
    ctx.beginPath()
    ctx.arc(Math.min(x1, rc.w) - 6.5, by + 5.5, 2.5, 0, Math.PI * 2)
    ctx.fill()
  }
}

function drawLanePills(rc: RC, p: Project, layout: Layout): void {
  const { theme } = rc
  drawPill(rc, 8, layout.waveTop + 12, 'ORIGINAL', theme.textDim, theme.border)
  layout.speakerLanes.forEach((lane, i) => {
    if (lane.top > rc.h) return
    const spk = p.speakers[i]
    if (!spk) return
    drawPill(rc, 8, lane.top + lane.height / 2, spk.name || spk.id, spk.color, withAlpha(spk.color, 0.35))
  })
  if (layout.dubTop < rc.h) {
    drawPill(rc, 8, layout.dubTop + 12, 'DUB', theme.accent, withAlpha(theme.accent, 0.4))
  }
}

function drawPill(rc: RC, x: number, cy: number, label: string, textColor: string, borderColor: string): void {
  const { ctx, theme, dpr } = rc
  ctx.font = `600 10px ${theme.font}`
  const tw = ctx.measureText(label).width
  const pw = tw + 14
  const ph = 16
  roundRectPath(ctx, crisp(x, dpr), crisp(cy - ph / 2, dpr), pw, ph, 8)
  ctx.fillStyle = withAlpha(theme.bg, 0.75)
  ctx.fill()
  ctx.strokeStyle = borderColor
  ctx.lineWidth = 1
  ctx.stroke()
  ctx.fillStyle = textColor
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  ctx.fillText(label, x + 7, cy + 0.5)
}

function drawEmptyHint(rc: RC): void {
  const { ctx, theme } = rc
  ctx.font = `12px ${theme.font}`
  ctx.fillStyle = theme.textDim
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('Run the pipeline to transcribe segments', rc.w / 2, RULER_H + Math.max(24, (rc.h - RULER_H) / 2))
}

function drawPlayhead(rc: RC, playhead: number, bottom: number): void {
  const { ctx, theme, dpr } = rc
  const x = (playhead - rc.scrollX) * rc.zoom
  if (x < -6 || x > rc.w + 6) return
  ctx.strokeStyle = theme.accent
  ctx.lineWidth = 1
  vline(ctx, x, 0, bottom, dpr)
  // triangle handle in the ruler
  const xx = crisp(x, dpr)
  ctx.fillStyle = theme.accent
  ctx.beginPath()
  ctx.moveTo(xx - 4.5, RULER_H - 10)
  ctx.lineTo(xx + 4.5, RULER_H - 10)
  ctx.lineTo(xx, RULER_H - 1)
  ctx.closePath()
  ctx.fill()
}

function drawRulerHover(rc: RC, hover: HoverInfo | null, bottom: number): void {
  if (!hover) return
  const { ctx, theme, dpr } = rc
  const t = rc.scrollX + hover.x / rc.zoom
  if (t < 0) return
  ctx.strokeStyle = withAlpha(theme.text, 0.22)
  ctx.lineWidth = 1
  vline(ctx, hover.x, 0, bottom, dpr)

  const label = formatTime(t, true)
  ctx.font = `10px ${theme.mono}`
  const tw = ctx.measureText(label).width
  const bw = tw + 12
  const bh = 18
  let bx = hover.x + 10
  if (bx + bw > rc.w - 4) bx = hover.x - 10 - bw
  const by = RULER_H + 6
  roundRectPath(ctx, crisp(bx, dpr), crisp(by, dpr), bw, bh, 4)
  ctx.fillStyle = withAlpha(theme.bgOverlay, 0.95)
  ctx.fill()
  ctx.strokeStyle = theme.borderStrong
  ctx.lineWidth = 1
  ctx.stroke()
  ctx.fillStyle = theme.text
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  ctx.fillText(label, bx + 6, by + bh / 2 + 0.5)
}

// -------------------------------------------------------------- hit test ----

export type Hit =
  | { kind: 'ruler' }
  | { kind: 'wave' }
  | { kind: 'dub' }
  | { kind: 'segment'; segmentId: string }
  | { kind: 'edge'; segmentId: string; side: 'start' | 'end' }
  | { kind: 'lane' }
  | { kind: 'none' }

const EDGE_HOT_PX = 5
const MIN_TRIM_WIDTH = 16

/** Map a canvas-local point to what is under it. Topmost (last-drawn) segment wins. */
export function hitTest(
  layout: Layout,
  project: Project,
  x: number,
  y: number,
  zoom: number,
  scrollX: number,
  override: DragOverride | null,
): Hit {
  if (y < RULER_H) return { kind: 'ruler' }
  if (y < layout.waveTop + layout.waveH) return { kind: 'wave' }
  if (y >= layout.dubTop && y < layout.dubTop + layout.dubH) return { kind: 'dub' }
  const lane = layout.speakerLanes.find((l) => y >= l.top && y < l.top + l.height)
  if (!lane) return { kind: 'none' }
  if (y < lane.top + 3 || y > lane.top + lane.height - 3) return { kind: 'lane' }

  const segs = project.segments
  for (let i = segs.length - 1; i >= 0; i--) {
    const seg = segs[i]
    if (seg.speaker_id !== lane.speakerId) continue
    const ov = override && override.segmentId === seg.id ? override : null
    const x0 = ((ov ? ov.start : seg.start) - scrollX) * zoom
    const x1 = ((ov ? ov.end : seg.end) - scrollX) * zoom
    if (x < x0 - EDGE_HOT_PX || x > x1 + EDGE_HOT_PX) continue
    if (x1 - x0 >= MIN_TRIM_WIDTH) {
      const dStart = Math.abs(x - x0)
      const dEnd = Math.abs(x - x1)
      if (dStart <= EDGE_HOT_PX || dEnd <= EDGE_HOT_PX) {
        return { kind: 'edge', segmentId: seg.id, side: dStart <= dEnd ? 'start' : 'end' }
      }
    }
    if (x >= x0 && x <= x1) return { kind: 'segment', segmentId: seg.id }
  }
  return { kind: 'lane' }
}
