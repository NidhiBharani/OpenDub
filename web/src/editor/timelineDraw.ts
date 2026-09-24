// timelineDraw.ts — pure canvas rendering, layout, hit-testing and snapping for the Timeline.
// No React / store imports: everything arrives as arguments so the component can drive
// redraws from a single requestAnimationFrame loop and stay at 60fps.
//
// Vertical stack (docs/plans/editor-redesign.md §5):
//   ruler 22 · scene strip 8 · V filmstrip 36 · ORIGINAL wave 48 · speaker lanes · DUB wave 48

import type { Anchor, Filmstrip, Project, Segment, Speaker, TimeRange, Waveform } from '../types'
import { formatTime } from '../types'

// ---------------------------------------------------------------- layout ----

export const RULER_H = 22
export const SCENE_H = 8
export const FILM_H = 36
export const WAVE_H = 48
export const DUB_H = 48
/** Top of the first lane below the ruler + scene strip. */
export const LANES_TOP = RULER_H + SCENE_H
const SPEAKER_LANE_MIN_H = 20

export type ClipHeight = 'S' | 'M' | 'L'
export const CLIP_LANE_H: Record<ClipHeight, number> = { S: 24, M: 36, L: 56 }

export interface SpeakerLane {
  speakerId: string
  top: number
  height: number
}

export interface Layout {
  width: number
  height: number
  rulerH: number
  sceneTop: number
  sceneH: number
  filmTop: number
  filmH: number
  waveTop: number
  waveH: number
  speakerLanes: SpeakerLane[]
  dubTop: number
  dubH: number
  /** Bottom of the DUB lane (may exceed `height` when there are many speakers). */
  bottom: number
}

/** Vertical arrangement of lanes. Speaker lanes take `clipHeight` and only shrink (down to
 *  20px) when they would push the DUB lane off the canvas. */
export function computeLayout(
  width: number, height: number, speakers: Speaker[], clipHeight: ClipHeight = 'M',
): Layout {
  const n = speakers.length
  let spkH = CLIP_LANE_H[clipHeight]
  if (n > 0) {
    const avail = height - LANES_TOP - FILM_H - WAVE_H - DUB_H
    if (n * spkH > avail) spkH = Math.max(SPEAKER_LANE_MIN_H, Math.floor(avail / n))
  }
  const filmTop = LANES_TOP
  const waveTop = filmTop + FILM_H
  const speakerLanes: SpeakerLane[] = []
  let y = waveTop + WAVE_H
  for (const s of speakers) {
    speakerLanes.push({ speakerId: s.id, top: y, height: spkH })
    y += spkH
  }
  return {
    width, height,
    rulerH: RULER_H,
    sceneTop: RULER_H, sceneH: SCENE_H,
    filmTop, filmH: FILM_H,
    waveTop, waveH: WAVE_H,
    speakerLanes,
    dubTop: y, dubH: DUB_H,
    bottom: y + DUB_H,
  }
}

// ----------------------------------------------------------------- theme ----

export interface Theme {
  bg: string
  bgPanel: string
  bgRaised: string
  bgOverlay: string
  border: string
  borderStrong: string
  borderSubtle: string
  text: string
  textDim: string
  textFaint: string
  accent: string
  playhead: string
  skimmer: string
  range: string
  ok: string
  warn: string
  err: string
  /** speaker palette for the active theme, indexed by cast order */
  spk: string[]
  font: string
  mono: string
}

const FALLBACK: Theme = {
  bg: '#19191b',
  bgPanel: '#242426',
  bgRaised: '#2c2c2e',
  bgOverlay: '#343436',
  border: '#3a3a3c',
  borderStrong: '#454547',
  borderSubtle: '#2f2f31',
  text: '#e5e5e7',
  textDim: '#9a9aa0',
  textFaint: '#6a6a70',
  accent: '#3d7eff',
  playhead: '#f2f2f2',
  skimmer: '#ff3b30',
  range: '#ffd60a',
  ok: '#30d158',
  warn: '#ff9f0a',
  err: '#ff453a',
  spk: ['#5b8def', '#a06fd8', '#4fb286', '#d9a441', '#e0705e', '#4db3c9', '#c46aa0', '#8f9d5c'],
  font: "-apple-system, 'SF Pro Text', 'Inter', 'Segoe UI', system-ui, sans-serif",
  mono: "ui-monospace, 'SF Mono', 'JetBrains Mono', 'Cascadia Code', Menlo, monospace",
}

/** Resolve design tokens from CSS variables (falls back to the dark theme.css hexes).
 *  Call again whenever the theme changes. */
export function resolveTheme(): Theme {
  const cs = getComputedStyle(document.documentElement)
  const v = (name: string, fallback: string): string => {
    const raw = cs.getPropertyValue(name).trim()
    return raw !== '' ? raw : fallback
  }
  return {
    bg: v('--bg-sunk', FALLBACK.bg),
    bgPanel: v('--bg-panel', FALLBACK.bgPanel),
    bgRaised: v('--bg-raised', FALLBACK.bgRaised),
    bgOverlay: v('--bg-overlay', FALLBACK.bgOverlay),
    border: v('--border', FALLBACK.border),
    borderStrong: v('--border-strong', FALLBACK.borderStrong),
    borderSubtle: v('--border-subtle', FALLBACK.borderSubtle),
    text: v('--text', FALLBACK.text),
    textDim: v('--text-dim', FALLBACK.textDim),
    textFaint: v('--text-faint', FALLBACK.textFaint),
    accent: v('--accent', FALLBACK.accent),
    playhead: v('--playhead', FALLBACK.playhead),
    skimmer: v('--skimmer', FALLBACK.skimmer),
    range: v('--range', FALLBACK.range),
    ok: v('--ok', FALLBACK.ok),
    warn: v('--warn', FALLBACK.warn),
    err: v('--err', FALLBACK.err),
    spk: FALLBACK.spk.map((fb, i) => v(`--spk-${i + 1}`, fb)),
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

/** Which contact-sheet tile shows time `t`, and where it sits in the sheet. */
export function filmstripTile(fs: Filmstrip, t: number): { index: number; sx: number; sy: number } {
  const n = Math.max(1, fs.count)
  const index = Math.max(0, Math.min(n - 1, Math.floor(t / Math.max(1e-6, fs.interval))))
  const cols = Math.max(1, fs.cols)
  return { index, sx: (index % cols) * fs.tile_w, sy: Math.floor(index / cols) * fs.tile_h }
}

/** Anchors that pass the confidence threshold (segment ticks additionally need zoom > 20 px/s). */
export function visibleAnchors(anchors: Anchor[], threshold: number, zoom: number): Anchor[] {
  return anchors.filter((a) => a.confidence >= threshold && (a.kind !== 'segment' || zoom > 20))
}

/** The span between the two visible anchors around `t` (media start/end bound it). */
export function anchorSpanAt(anchors: Anchor[], t: number, duration: number): { start: number; end: number } {
  let start = 0
  let end = duration
  for (const a of anchors) {
    if (a.t <= t && a.t > start) start = a.t
    if (a.t > t && a.t < end) end = a.t
  }
  return { start, end }
}

// ------------------------------------------------------------- snapping ----

export const SNAP_PX = 6

/** Every time a dragged edge may snap to: anchors, segment edges (minus the one being dragged),
 *  skip-range edges (minus the one being dragged) and the playhead. Sorted ascending. */
export function snapCandidates(
  project: Project,
  anchors: Anchor[],
  playhead: number,
  exclude: { segmentId?: string; skipRangeId?: string } = {},
): number[] {
  const out: number[] = [playhead]
  for (const a of anchors) {
    out.push(a.t)
    if (a.kind === 'silence' && a.end > a.start) out.push(a.start, a.end)
  }
  for (const s of project.segments) {
    if (s.id === exclude.segmentId) continue
    out.push(s.start, s.end)
  }
  for (const r of project.skip_ranges ?? []) {
    if (r.id === exclude.skipRangeId) continue
    out.push(r.start, r.end)
  }
  return out.sort((a, b) => a - b)
}

/** Snap `t` to the nearest candidate within `SNAP_PX` at the given zoom. */
export function resolveSnap(
  t: number, candidates: number[], zoom: number, thresholdPx = SNAP_PX,
): { t: number; snapped: boolean } {
  let best = t
  let bestD = thresholdPx / zoom
  let snapped = false
  for (const c of candidates) {
    const d = Math.abs(c - t)
    if (d <= bestD) {
      bestD = d
      best = c
      snapped = true
    }
  }
  return { t: best, snapped }
}

/** Duration of the take a segment will use in the mix (active take, else the newest). */
export function activeTakeDuration(seg: Segment): number | null {
  if (seg.takes.length === 0) return null
  const take = seg.takes.find((t) => t.id === seg.active_take_id) ?? seg.takes[seg.takes.length - 1]
  return take.duration > 0 ? take.duration : null
}

/** How far a segment's voiced take overruns its slot (1 = fits exactly). */
export function overrunRatio(seg: Segment, start: number, end: number): number | null {
  const d = activeTakeDuration(seg)
  const slot = end - start
  if (d === null || slot <= 0) return null
  return d / slot
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

/** Diagonal hatch (top-left → bottom-right) clipped to a rect. */
function hatchRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number,
  spacing: number, color: string, lineWidth = 1,
): void {
  if (w <= 0 || h <= 0) return
  ctx.save()
  ctx.beginPath()
  ctx.rect(x, y, w, h)
  ctx.clip()
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.beginPath()
  // start on a global grid so adjacent hatches line up while panning
  const first = Math.floor((x - h) / spacing) * spacing
  for (let sx = first; sx < x + w; sx += spacing) {
    ctx.moveTo(sx, y)
    ctx.lineTo(sx + h, y + h)
  }
  ctx.stroke()
  ctx.restore()
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

export interface SkipOverride {
  rangeId: string
  start: number
  end: number
}

export interface TimeSpan {
  start: number
  end: number
}

export interface DrawInput {
  width: number
  height: number
  dpr: number
  project: Project | null
  clipHeight: ClipHeight
  /** vocals ?? original */
  waveMain: Waveform | null
  waveDub: Waveform | null
  filmstrip: Filmstrip | null
  /** The decoded contact sheet (null until loaded). */
  filmImage: CanvasImageSource | null
  /** Anchors already filtered by threshold/zoom (see `visibleAnchors`). */
  anchors: Anchor[]
  zoom: number
  scrollX: number
  playhead: number
  selection: string | null
  skipSelection: string | null
  range: TimeSpan | null
  /** Pointer x over the canvas (skimmer), or null. */
  skimmerX: number | null
  /** Vertical snap guide (time), or null. */
  snapT: number | null
  /** Live segment-trim preview (during drag, or pending server persist). */
  override: DragOverride | null
  /** Live skip-range trim preview. */
  skipOverride: SkipOverride | null
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

const tx = (rc: RC, t: number): number => (t - rc.scrollX) * rc.zoom

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
  const layout = computeLayout(w, h, p.speakers, s.clipHeight)
  const bottom = Math.min(layout.bottom, h)
  const step = tickStep(s.zoom)

  // lane surfaces: content lanes on the sunk bg, the filmstrip row a shade darker
  ctx.fillStyle = withAlpha('#000000', 0.35)
  ctx.fillRect(0, layout.filmTop, w, layout.filmH)

  drawGrid(rc, dur, step, LANES_TOP, bottom)
  drawFilmstrip(rc, layout, s.filmstrip, s.filmImage, dur)
  drawWaveLane(rc, layout.waveTop, layout.waveH, s.waveMain, theme.textFaint, 0.55)
  drawWaveLane(rc, layout.dubTop, layout.dubH, s.waveDub, theme.accent, 0.45)
  drawSegments(rc, p, layout, s.selection, s.override)
  drawSkipRanges(rc, p.skip_ranges ?? [], layout, bottom, s.skipSelection, s.skipOverride)

  // shade the area past the end of the media
  const endX = tx(rc, dur)
  if (endX < w) {
    const x0 = Math.max(0, endX)
    ctx.fillStyle = withAlpha('#000000', 0.35)
    ctx.fillRect(x0, LANES_TOP, w - x0, Math.max(0, bottom - LANES_TOP))
  }

  // lane separators
  ctx.strokeStyle = theme.borderSubtle
  ctx.lineWidth = 1
  hline(ctx, 0, w, layout.filmTop + layout.filmH, dpr)
  hline(ctx, 0, w, layout.waveTop + layout.waveH, dpr)
  for (const lane of layout.speakerLanes) hline(ctx, 0, w, lane.top + lane.height, dpr)
  if (layout.dubTop + layout.dubH <= h) hline(ctx, 0, w, layout.dubTop + layout.dubH, dpr)

  drawRange(rc, s.range, bottom)
  drawRuler(rc, dur, step)
  drawSceneStrip(rc, s.anchors, layout)
  drawRangeHandles(rc, s.range)
  if (p.segments.length === 0) drawEmptyHint(rc)
  if (s.snapT !== null) drawSnapGuide(rc, s.snapT, bottom)
  drawPlayhead(rc, s.playhead, bottom)
  drawSkimmer(rc, s.skimmerX, bottom)
}

// ------------------------------------------------------------ sub-draws -----

function drawGrid(rc: RC, dur: number, step: number, yTop: number, yBottom: number): void {
  const { ctx, theme, dpr } = rc
  ctx.strokeStyle = withAlpha(theme.border, 0.45)
  ctx.lineWidth = 1
  const i0 = Math.max(0, Math.floor(rc.scrollX / step))
  const i1 = Math.ceil((rc.scrollX + rc.w / rc.zoom) / step)
  for (let i = i0; i <= i1; i++) {
    const t = i * step
    if (t > dur + 1e-6) break
    vline(ctx, tx(rc, t), yTop, yBottom, dpr)
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
    const x = tx(rc, t)
    const major = i % 5 === 0
    ctx.strokeStyle = major ? theme.textFaint : theme.borderStrong
    vline(ctx, x, RULER_H - (major ? 8 : 4), RULER_H, dpr)
    if (major) {
      ctx.fillStyle = theme.textDim
      ctx.fillText(formatTime(t, step < 1), x + 4, 8)
    }
  }
}

function drawSceneStrip(rc: RC, anchors: Anchor[], layout: Layout): void {
  const { ctx, theme, dpr, w } = rc
  const top = layout.sceneTop
  const hgt = layout.sceneH
  ctx.fillStyle = theme.bgPanel
  ctx.fillRect(0, top, w, hgt)
  ctx.strokeStyle = theme.border
  ctx.lineWidth = 1
  hline(ctx, 0, w, top + hgt, dpr)

  const t0 = rc.scrollX
  const t1 = rc.scrollX + w / rc.zoom
  // silences first so ticks stay legible over them
  for (const a of anchors) {
    if (a.kind !== 'silence') continue
    if (a.end < t0 || a.start > t1) continue
    const x0 = tx(rc, a.start)
    const x1 = tx(rc, a.end)
    ctx.fillStyle = withAlpha(theme.textDim, 0.35)
    ctx.fillRect(x0, top + 2, Math.max(1, x1 - x0), hgt - 4)
  }
  for (const a of anchors) {
    if (a.kind === 'silence') continue
    if (a.t < t0 || a.t > t1) continue
    if (a.kind === 'scene') {
      // taller = more confident (min half the strip)
      const th = Math.max(hgt / 2, Math.round(hgt * Math.max(0, Math.min(1, a.confidence))))
      ctx.strokeStyle = theme.textDim
      vline(ctx, tx(rc, a.t), top + hgt - th, top + hgt, dpr)
    } else {
      ctx.strokeStyle = withAlpha(theme.textDim, 0.4)
      vline(ctx, tx(rc, a.t), top, top + hgt, dpr)
    }
  }
}

function drawFilmstrip(
  rc: RC, layout: Layout, fs: Filmstrip | null, img: CanvasImageSource | null, dur: number,
): void {
  if (!fs || !img || fs.tile_h <= 0 || fs.tile_w <= 0 || fs.count <= 0) return
  const { ctx } = rc
  const top = layout.filmTop
  const hgt = layout.filmH
  const tileW = Math.max(8, Math.round(fs.tile_w * hgt / fs.tile_h))
  // time-anchored columns, one tile wide, so tiles do not swim while panning
  const colDur = tileW / rc.zoom
  const k0 = Math.max(0, Math.floor(rc.scrollX / colDur))
  const k1 = Math.ceil((rc.scrollX + rc.w / rc.zoom) / colDur)
  ctx.save()
  ctx.beginPath()
  ctx.rect(0, top, Math.min(rc.w, tx(rc, dur)), hgt)
  ctx.clip()
  for (let k = k0; k <= k1; k++) {
    const t = k * colDur
    if (t >= dur) break
    const tile = filmstripTile(fs, t + colDur / 2)
    const x = tx(rc, t)
    try {
      ctx.drawImage(img, tile.sx, tile.sy, fs.tile_w, fs.tile_h, x, top, tileW, hgt)
    } catch {
      break // image not decodable yet
    }
  }
  ctx.restore()
}

/** Display auto-gain: scale so the 98th-percentile |peak| fills ~90% of the lane (quiet dialog
 * stays readable, like real editors). Cached per Waveform object. */
const gainCache = new WeakMap<Waveform, number>()
function displayGain(wf: Waveform): number {
  let g = gainCache.get(wf)
  if (g === undefined) {
    const abs = wf.peaks.map(Math.abs).sort((a, b) => a - b)
    const p98 = abs[Math.min(abs.length - 1, Math.floor(abs.length * 0.98))] || 0
    g = p98 > 0.001 ? Math.min(8, 0.9 / p98) : 1
    gainCache.set(wf, g)
  }
  return g
}

/** Min/max waveform fill: one column per CSS pixel over the visible time range. */
function drawWaveLane(
  rc: RC, top: number, height: number, wf: Waveform | null, baseColor: string, alpha: number,
): void {
  const { ctx, dpr, w } = rc
  const mid = top + height / 2

  // center hairline (also the "no data yet" indicator)
  ctx.fillStyle = withAlpha(baseColor, alpha * 0.5)
  ctx.fillRect(0, Math.round(mid * dpr) / dpr, w, 1 / dpr)
  if (!wf || wf.peaks.length < 2 || wf.sample_rate <= 0) return

  const gain = displayGain(wf)
  const amp = Math.max(2, height / 2 - 4)
  const sr = wf.sample_rate
  const total = wf.peaks.length >> 1
  const minH = 1 / dpr
  ctx.fillStyle = withAlpha(baseColor, alpha)
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
    const y0 = mid - Math.min(1, Math.max(-1, mx * gain)) * amp
    const y1 = mid - Math.min(1, Math.max(-1, mn * gain)) * amp
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
  const x0 = tx(rc, start)
  const x1 = tx(rc, end)
  const outlineOnly = seg.skipped
  const dirty = seg.translate_dirty || seg.synth_dirty
  const bw = Math.max(2, x1 - x0)
  const by = lane.top + 2
  const bh = lane.height - 4
  const r = Math.min(3, bw / 2, bh / 2)

  roundRectPath(ctx, crisp(x0, dpr), crisp(by, dpr), bw, bh, r)
  if (!outlineOnly) {
    ctx.fillStyle = withAlpha(color, selected ? 0.5 : 0.35)
    ctx.fill()
  }
  if (selected) {
    ctx.strokeStyle = theme.accent
    ctx.lineWidth = 2
  } else {
    ctx.strokeStyle = withAlpha(color, outlineOnly ? 0.45 : 1)
    ctx.lineWidth = 1
  }
  ctx.stroke()
  ctx.lineWidth = 1

  // overrun tail: the voiced take is longer than its slot
  const ratio = overrunRatio(seg, start, end)
  if (ratio !== null && ratio > 1.1 && bw > 14) {
    const tailX = Math.min(x1, rc.w) - 7
    hatchRect(ctx, tailX, by + 1, 6, bh - 2, 4, withAlpha(ratio > 1.3 ? theme.err : theme.warn, 0.9), 1)
  }

  // label — keep it visible when the block extends past the left edge
  const label = seg.translated_text || seg.source_text
  const textX = Math.max(x0, 0) + 6
  const room = Math.min(x1, rc.w) - textX - (ratio !== null && ratio > 1.1 ? 12 : 6)
  if (label && bw > 24 && room > 12 && bh >= 14) {
    ctx.font = `11px ${theme.font}`
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillStyle = outlineOnly ? theme.textFaint : theme.text
    const shown = ellipsize(ctx, label, room)
    if (shown) ctx.fillText(shown, textX, lane.top + lane.height / 2 + 0.5)
  }

  // amber stale dot, top-right
  if (dirty && bw > 12) {
    ctx.fillStyle = theme.warn
    ctx.beginPath()
    ctx.arc(Math.min(x1, rc.w) - 6, by + 5, 2.5, 0, Math.PI * 2)
    ctx.fill()
  }
}

function drawSkipRanges(
  rc: RC,
  ranges: TimeRange[],
  layout: Layout,
  bottom: number,
  selectedId: string | null,
  override: SkipOverride | null,
): void {
  const { ctx, theme, dpr } = rc
  const top = layout.filmTop
  const hgt = Math.max(0, bottom - top)
  if (hgt <= 0) return
  const t0 = rc.scrollX
  const t1 = rc.scrollX + rc.w / rc.zoom
  for (const r of ranges) {
    const ov = override && override.rangeId === r.id ? override : null
    const start = ov ? ov.start : r.start
    const end = ov ? ov.end : r.end
    if (end < t0 || start > t1) continue
    const x0 = tx(rc, start)
    const x1 = tx(rc, end)
    const w = Math.max(1, x1 - x0)
    ctx.fillStyle = withAlpha('#000000', 0.3)
    ctx.fillRect(x0, top, w, hgt)
    hatchRect(ctx, x0, top, w, hgt, 8, withAlpha(theme.textFaint, 0.28))
    const selected = r.id === selectedId
    ctx.strokeStyle = selected ? theme.accent : theme.borderStrong
    ctx.lineWidth = 1
    ctx.strokeRect(crisp(x0, dpr), crisp(top, dpr), Math.max(1, Math.round(x1) - Math.round(x0)), hgt)
    // label, top-left, kept on-screen while the region scrolls past the left edge
    const text = r.label ? `KEEP ORIGINAL · ${r.label}` : 'KEEP ORIGINAL'
    ctx.font = `600 10px ${theme.font}`
    const lx = Math.max(x0, 0) + 6
    const room = Math.min(x1, rc.w) - lx - 4
    if (room > 20) {
      ctx.textAlign = 'left'
      ctx.textBaseline = 'middle'
      ctx.fillStyle = selected ? theme.text : theme.textDim
      ctx.fillText(ellipsize(ctx, text, room), lx, top + 9)
    }
  }
}

function drawRange(rc: RC, range: TimeSpan | null, bottom: number): void {
  if (!range) return
  const { ctx, theme, dpr } = rc
  const x0 = tx(rc, range.start)
  const x1 = tx(rc, range.end)
  if (x1 < 0 || x0 > rc.w) return
  ctx.fillStyle = 'rgba(255, 255, 255, 0.08)'
  ctx.fillRect(x0, LANES_TOP, Math.max(1, x1 - x0), Math.max(0, bottom - LANES_TOP))
  ctx.strokeStyle = theme.range
  ctx.lineWidth = 1
  vline(ctx, x0, LANES_TOP, bottom, dpr)
  vline(ctx, x1, LANES_TOP, bottom, dpr)
}

export const RANGE_HANDLE_W = 6

/** Yellow in/out brackets in the ruler (drawn over the ruler so they stay visible). */
function drawRangeHandles(rc: RC, range: TimeSpan | null): void {
  if (!range) return
  const { ctx, theme } = rc
  const x0 = tx(rc, range.start)
  const x1 = tx(rc, range.end)
  if (x1 < 0 || x0 > rc.w) return
  ctx.fillStyle = withAlpha(theme.range, 0.12)
  ctx.fillRect(x0, 0, Math.max(1, x1 - x0), RULER_H - 1)
  ctx.fillStyle = theme.range
  ctx.fillRect(Math.round(x0), 0, RANGE_HANDLE_W, RULER_H - 1)
  ctx.fillRect(Math.round(x1) - RANGE_HANDLE_W, 0, RANGE_HANDLE_W, RULER_H - 1)
  // bracket notches so the handles read as in/out points
  ctx.fillStyle = theme.bg
  ctx.fillRect(Math.round(x0) + 2, 8, 2, RULER_H - 17)
  ctx.fillRect(Math.round(x1) - RANGE_HANDLE_W + 2, 8, 2, RULER_H - 17)
}

function drawEmptyHint(rc: RC): void {
  const { ctx, theme } = rc
  ctx.font = `12px ${theme.font}`
  ctx.fillStyle = theme.textDim
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('Run the pipeline to transcribe segments', rc.w / 2, LANES_TOP + Math.max(24, (rc.h - LANES_TOP) / 2))
}

function drawSnapGuide(rc: RC, t: number, bottom: number): void {
  const { ctx, theme, dpr } = rc
  const x = tx(rc, t)
  if (x < 0 || x > rc.w) return
  ctx.strokeStyle = theme.range
  ctx.lineWidth = 1
  vline(ctx, x, 0, bottom, dpr)
}

function drawPlayhead(rc: RC, playhead: number, bottom: number): void {
  const { ctx, theme, dpr } = rc
  const x = tx(rc, playhead)
  if (x < -6 || x > rc.w + 6) return
  ctx.strokeStyle = theme.playhead
  ctx.lineWidth = 1
  vline(ctx, x, 0, bottom, dpr)
  // 8px downward triangle in the ruler
  const xx = crisp(x, dpr)
  ctx.fillStyle = theme.playhead
  ctx.beginPath()
  ctx.moveTo(xx - 4, RULER_H - 8)
  ctx.lineTo(xx + 4, RULER_H - 8)
  ctx.lineTo(xx, RULER_H - 1)
  ctx.closePath()
  ctx.fill()
}

function drawSkimmer(rc: RC, x: number | null, bottom: number): void {
  if (x === null) return
  const { ctx, theme, dpr } = rc
  const t = rc.scrollX + x / rc.zoom
  if (t < 0 || x < 0 || x > rc.w) return
  ctx.strokeStyle = theme.skimmer
  ctx.lineWidth = 1
  vline(ctx, x, 0, bottom, dpr)
  // timecode next to the skimmer in the ruler
  const label = formatTime(t, true)
  ctx.font = `10px ${theme.mono}`
  const tw = ctx.measureText(label).width
  const bw = tw + 8
  let bx = x + 5
  if (bx + bw > rc.w - 2) bx = x - 5 - bw
  ctx.fillStyle = theme.bgRaised
  ctx.fillRect(bx, 1, bw, RULER_H - 3)
  ctx.fillStyle = theme.text
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  ctx.fillText(label, bx + 4, RULER_H / 2 - 0.5)
}

// -------------------------------------------------------------- hit test ----

export type Hit =
  | { kind: 'ruler' }
  | { kind: 'rangeHandle'; side: 'start' | 'end' }
  | { kind: 'scene' }
  | { kind: 'film' }
  | { kind: 'wave' }
  | { kind: 'dub' }
  | { kind: 'segment'; segmentId: string }
  | { kind: 'edge'; segmentId: string; side: 'start' | 'end' }
  | { kind: 'skip'; rangeId: string }
  | { kind: 'skipEdge'; rangeId: string; side: 'start' | 'end' }
  | { kind: 'lane' }
  | { kind: 'none' }

const EDGE_HOT_PX = 5
const MIN_TRIM_WIDTH = 16

export interface HitContext {
  zoom: number
  scrollX: number
  range: TimeSpan | null
  override: DragOverride | null
  skipOverride: SkipOverride | null
}

/** Map a canvas-local point to what is under it. Topmost (last-drawn) segment wins. */
export function hitTest(layout: Layout, project: Project, x: number, y: number, hc: HitContext): Hit {
  const { zoom, scrollX } = hc
  const px = (t: number): number => (t - scrollX) * zoom
  // older servers do not send skip_ranges yet
  const skipRanges = project.skip_ranges ?? []

  if (y < RULER_H) {
    if (hc.range) {
      const x0 = px(hc.range.start)
      const x1 = px(hc.range.end)
      if (x >= x0 - 2 && x <= x0 + RANGE_HANDLE_W + 2) return { kind: 'rangeHandle', side: 'start' }
      if (x >= x1 - RANGE_HANDLE_W - 2 && x <= x1 + 2) return { kind: 'rangeHandle', side: 'end' }
    }
    return { kind: 'ruler' }
  }
  if (y < layout.sceneTop + layout.sceneH) return { kind: 'scene' }

  // skip-range edges are grabbable in every lane below the ruler
  const skipEdge = (): Hit | null => {
    for (let i = skipRanges.length - 1; i >= 0; i--) {
      const r = skipRanges[i]
      const ov = hc.skipOverride && hc.skipOverride.rangeId === r.id ? hc.skipOverride : null
      const x0 = px(ov ? ov.start : r.start)
      const x1 = px(ov ? ov.end : r.end)
      if (x1 - x0 < MIN_TRIM_WIDTH) continue
      if (Math.abs(x - x0) <= EDGE_HOT_PX) return { kind: 'skipEdge', rangeId: r.id, side: 'start' }
      if (Math.abs(x - x1) <= EDGE_HOT_PX) return { kind: 'skipEdge', rangeId: r.id, side: 'end' }
    }
    return null
  }
  const skipBody = (): Hit | null => {
    for (let i = skipRanges.length - 1; i >= 0; i--) {
      const r = skipRanges[i]
      const ov = hc.skipOverride && hc.skipOverride.rangeId === r.id ? hc.skipOverride : null
      const x0 = px(ov ? ov.start : r.start)
      const x1 = px(ov ? ov.end : r.end)
      if (x >= x0 && x <= x1) return { kind: 'skip', rangeId: r.id }
    }
    return null
  }

  if (y < layout.filmTop + layout.filmH) return skipEdge() ?? { kind: 'film' }
  if (y < layout.waveTop + layout.waveH) return skipEdge() ?? { kind: 'wave' }
  if (y >= layout.dubTop && y < layout.dubTop + layout.dubH) return skipEdge() ?? { kind: 'dub' }
  const lane = layout.speakerLanes.find((l) => y >= l.top && y < l.top + l.height)
  if (!lane) return { kind: 'none' }

  if (y >= lane.top + 2 && y <= lane.top + lane.height - 2) {
    const segs = project.segments
    for (let i = segs.length - 1; i >= 0; i--) {
      const seg = segs[i]
      if (seg.speaker_id !== lane.speakerId) continue
      const ov = hc.override && hc.override.segmentId === seg.id ? hc.override : null
      const x0 = px(ov ? ov.start : seg.start)
      const x1 = px(ov ? ov.end : seg.end)
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
  }
  return skipEdge() ?? skipBody() ?? { kind: 'lane' }
}
