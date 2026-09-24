import { describe, expect, it } from 'vitest'
import { demoProject } from '../test/fixtures'
import type { Anchor, Filmstrip, Project } from '../types'
import {
  CLIP_LANE_H, FILM_H, LANES_TOP, RULER_H, WAVE_H,
  anchorSpanAt, computeLayout, filmstripTile, hitTest, overrunRatio, resolveSnap, snapCandidates,
  tickStep, visibleAnchors, withAlpha,
  type HitContext,
} from './timelineDraw'

const speaker2 = { id: 'spk_2', name: 'Speaker 2', color: '#4fb286', reference_path: null }
const project: Project = {
  ...demoProject,
  speakers: [...demoProject.speakers, speaker2],
  segments: [
    ...demoProject.segments, // seg_019e618e: 1.0–3.2 on speaker 1
    {
      ...demoProject.segments[0], id: 'seg_2', start: 5, end: 8, speaker_id: 'spk_2',
    },
  ],
  skip_ranges: [{ id: 'rng_1', start: 10, end: 14, label: 'song' }],
}

const hc = (over: Partial<HitContext> = {}): HitContext => ({
  zoom: 10, scrollX: 0, range: null, override: null, skipOverride: null, ...over,
})

describe('tickStep', () => {
  it('picks the smallest step whose major spacing is at least 70px', () => {
    expect(tickStep(100)).toBe(1) // 1s → 100px
    expect(tickStep(30)).toBe(5) // 2s → 60px too tight, 5s → 150px
    expect(tickStep(2)).toBe(60) // 30s → 60px, 60s → 120px
    expect(tickStep(0.01)).toBe(300)
  })
})

describe('computeLayout', () => {
  it('stacks ruler, scene strip, filmstrip, original, speakers, dub', () => {
    const l = computeLayout(800, 400, project.speakers, 'M')
    expect(l.sceneTop).toBe(RULER_H)
    expect(l.filmTop).toBe(LANES_TOP)
    expect(l.waveTop).toBe(LANES_TOP + FILM_H)
    expect(l.speakerLanes).toHaveLength(2)
    expect(l.speakerLanes[0].top).toBe(LANES_TOP + FILM_H + WAVE_H)
    expect(l.speakerLanes[0].height).toBe(CLIP_LANE_H.M)
    expect(l.speakerLanes[1].top).toBe(l.speakerLanes[0].top + CLIP_LANE_H.M)
    expect(l.dubTop).toBe(l.speakerLanes[1].top + CLIP_LANE_H.M)
    expect(l.bottom).toBe(l.dubTop + l.dubH)
  })

  it('honours the clip height setting', () => {
    expect(computeLayout(800, 400, project.speakers, 'S').speakerLanes[0].height).toBe(24)
    expect(computeLayout(800, 400, project.speakers, 'L').speakerLanes[0].height).toBe(56)
  })

  it('shrinks speaker lanes (never below 20px) when they would push the DUB lane off-canvas', () => {
    const many = Array.from({ length: 8 }, (_, i) => ({ ...speaker2, id: `s${i}` }))
    const l = computeLayout(800, 240, many, 'L')
    expect(l.speakerLanes.every((lane) => lane.height === 20)).toBe(true)
  })
})

describe('hitTest', () => {
  const layout = computeLayout(800, 400, project.speakers, 'M')
  const laneY = (i: number) => layout.speakerLanes[i].top + layout.speakerLanes[i].height / 2

  it('maps the fixed rows', () => {
    expect(hitTest(layout, project, 50, 5, hc()).kind).toBe('ruler')
    expect(hitTest(layout, project, 50, RULER_H + 3, hc()).kind).toBe('scene')
    expect(hitTest(layout, project, 50, layout.filmTop + 5, hc()).kind).toBe('film')
    expect(hitTest(layout, project, 50, layout.waveTop + 5, hc()).kind).toBe('wave')
    expect(hitTest(layout, project, 50, layout.dubTop + 5, hc()).kind).toBe('dub')
    expect(hitTest(layout, project, 50, layout.bottom + 5, hc()).kind).toBe('none')
  })

  it('finds segments, their edges and empty lane space in the right speaker lane', () => {
    // seg_019e618e spans 1.0–3.2s → 10–32px at zoom 10
    expect(hitTest(layout, project, 20, laneY(0), hc())).toEqual({ kind: 'segment', segmentId: 'seg_019e618e' })
    expect(hitTest(layout, project, 11, laneY(0), hc())).toEqual({ kind: 'edge', segmentId: 'seg_019e618e', side: 'start' })
    expect(hitTest(layout, project, 33, laneY(0), hc())).toEqual({ kind: 'edge', segmentId: 'seg_019e618e', side: 'end' })
    expect(hitTest(layout, project, 20, laneY(1), hc()).kind).toBe('lane') // other speaker's lane
    expect(hitTest(layout, project, 60, laneY(1), hc())).toEqual({ kind: 'segment', segmentId: 'seg_2' })
  })

  it('respects a live trim override', () => {
    const override = { segmentId: 'seg_019e618e', start: 1.0, end: 6.0 }
    expect(hitTest(layout, project, 50, laneY(0), hc({ override }))).toEqual({ kind: 'segment', segmentId: 'seg_019e618e' })
  })

  it('finds range handles in the ruler', () => {
    const range = { start: 2, end: 6 } // 20px..60px
    expect(hitTest(layout, project, 22, 5, hc({ range }))).toEqual({ kind: 'rangeHandle', side: 'start' })
    expect(hitTest(layout, project, 57, 5, hc({ range }))).toEqual({ kind: 'rangeHandle', side: 'end' })
    expect(hitTest(layout, project, 40, 5, hc({ range })).kind).toBe('ruler')
  })

  it('finds skip regions and their edges (edges in every lane, body only in empty lane space)', () => {
    // rng_1 spans 10–14s → 100–140px
    expect(hitTest(layout, project, 120, laneY(0), hc())).toEqual({ kind: 'skip', rangeId: 'rng_1' })
    expect(hitTest(layout, project, 101, laneY(1), hc())).toEqual({ kind: 'skipEdge', rangeId: 'rng_1', side: 'start' })
    expect(hitTest(layout, project, 139, layout.waveTop + 5, hc())).toEqual({ kind: 'skipEdge', rangeId: 'rng_1', side: 'end' })
    expect(hitTest(layout, project, 120, layout.waveTop + 5, hc()).kind).toBe('wave') // waveform still scrubs
  })

  it('prefers a segment over the skip region it sits in', () => {
    const p: Project = { ...project, skip_ranges: [{ id: 'rng_2', start: 0, end: 20, label: '' }] }
    expect(hitTest(layout, p, 20, laneY(0), hc())).toEqual({ kind: 'segment', segmentId: 'seg_019e618e' })
  })
})

describe('snapping', () => {
  const anchors: Anchor[] = [
    { t: 4, kind: 'scene', confidence: 0.9, start: 4, end: 4 },
    { t: 9.5, kind: 'silence', confidence: 0.5, start: 9, end: 10 },
  ]

  it('collects anchors, segment edges, skip edges and the playhead, minus the dragged item', () => {
    const c = snapCandidates(project, anchors, 2.5, { segmentId: 'seg_2' })
    expect(c).toContain(2.5) // playhead
    expect(c).toContain(4) // scene anchor
    expect(c).toContain(9) // silence extent
    expect(c).toContain(10) // silence extent (and skip start)
    expect(c).toContain(1.0) // seg_019e618e start
    expect(c).not.toContain(5) // seg_2 excluded
    expect(c).toContain(14) // skip end
    expect([...c].sort((a, b) => a - b)).toEqual(c)
  })

  it('snaps within 6px at the current zoom and not beyond', () => {
    const c = [4, 10]
    expect(resolveSnap(4.05, c, 100)).toEqual({ t: 4, snapped: true }) // 5px away
    expect(resolveSnap(4.07, c, 100)).toEqual({ t: 4.07, snapped: false }) // 7px away
    expect(resolveSnap(4.5, c, 10)).toEqual({ t: 4, snapped: true }) // 5px at zoom 10
    expect(resolveSnap(7, c, 10)).toEqual({ t: 7, snapped: false })
  })

  it('picks the nearest candidate', () => {
    expect(resolveSnap(6.9, [6.5, 7.2], 20).t).toBe(7.2)
  })
})

describe('anchors', () => {
  const anchors: Anchor[] = [
    { t: 4, kind: 'scene', confidence: 0.9, start: 4, end: 4 },
    { t: 6, kind: 'scene', confidence: 0.2, start: 6, end: 6 },
    { t: 8, kind: 'segment', confidence: 1, start: 8, end: 8 },
  ]

  it('filters by threshold and hides segment ticks when zoomed out', () => {
    expect(visibleAnchors(anchors, 0.35, 30).map((a) => a.t)).toEqual([4, 8])
    expect(visibleAnchors(anchors, 0.35, 10).map((a) => a.t)).toEqual([4])
    expect(visibleAnchors(anchors, 0, 30)).toHaveLength(3)
  })

  it('returns the span between the surrounding anchors, bounded by the media', () => {
    expect(anchorSpanAt(anchors, 5, 29.4)).toEqual({ start: 4, end: 6 })
    expect(anchorSpanAt(anchors, 1, 29.4)).toEqual({ start: 0, end: 4 })
    expect(anchorSpanAt(anchors, 20, 29.4)).toEqual({ start: 8, end: 29.4 })
  })
})

describe('filmstripTile', () => {
  const fs: Filmstrip = { interval: 2, cols: 4, rows: 3, tile_w: 160, tile_h: 90, count: 12, url: 'filmstrip.jpg' }

  it('maps a time to its tile and the tile to its sheet offset', () => {
    expect(filmstripTile(fs, 0)).toEqual({ index: 0, sx: 0, sy: 0 })
    expect(filmstripTile(fs, 3.9)).toEqual({ index: 1, sx: 160, sy: 0 })
    expect(filmstripTile(fs, 9)).toEqual({ index: 4, sx: 0, sy: 90 })
  })

  it('clamps past the last tile', () => {
    expect(filmstripTile(fs, 500).index).toBe(11)
    expect(filmstripTile(fs, -1).index).toBe(0)
  })
})

describe('overrunRatio', () => {
  const take = (id: string, duration: number) => ({
    id, path: '', duration, provider_id: 'tts.mock', rate_factor: 1, created_at: '', lang: 'en',
  })

  it('uses the active take, else the newest, and null without takes', () => {
    const seg = { ...project.segments[0], takes: [take('a', 1), take('b', 3)], active_take_id: 'a' }
    expect(overrunRatio(seg, 0, 2)).toBe(0.5)
    expect(overrunRatio({ ...seg, active_take_id: null }, 0, 2)).toBe(1.5)
    expect(overrunRatio({ ...seg, takes: [] }, 0, 2)).toBeNull()
  })
})

describe('withAlpha', () => {
  it('expands hex colours and leaves others alone', () => {
    expect(withAlpha('#fff', 0.5)).toBe('rgba(255, 255, 255, 0.5)')
    expect(withAlpha('#3d7eff', 0.35)).toBe('rgba(61, 126, 255, 0.35)')
    expect(withAlpha('var(--x)', 0.5)).toBe('var(--x)')
  })
})
