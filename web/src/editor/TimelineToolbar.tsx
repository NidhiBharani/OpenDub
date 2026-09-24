// TimelineToolbar — the 32px strip above the track headers + canvas.
//   left:   Select/Range tool · snapping · clip height
//   centre: Exclude from dub · Clear range · anchor threshold
//   right:  zoom −/+ · zoom slider (log 2..500 px/s) · Fit
import type { CSSProperties } from 'react'
import { Icon } from '../components/Icon'
import { Button, IconButton, MOD, Segmented } from '../components/primitives'
import { useStore } from '../state/store'
import type { ClipHeight } from '../state/store'

export type TimelineTool = 'select' | 'range'

const ZOOM_MIN = 2
const ZOOM_MAX = 500
const ZOOM_FACTOR = 1.3

/** zoom (px/s) ↔ slider position 0..1 on a log scale */
export function zoomToSlider(zoom: number): number {
  const z = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom))
  return Math.log(z / ZOOM_MIN) / Math.log(ZOOM_MAX / ZOOM_MIN)
}
export function sliderToZoom(v: number): number {
  const u = Math.max(0, Math.min(1, v))
  return ZOOM_MIN * Math.pow(ZOOM_MAX / ZOOM_MIN, u)
}

const sliderStyle: CSSProperties = {
  width: 88, height: 24, margin: 0, padding: 0, accentColor: 'var(--text-dim)', cursor: 'pointer',
}

const groupStyle: CSSProperties = { display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }

function Divider() {
  return <div aria-hidden="true" style={{ width: 1, height: 16, background: 'var(--border)', margin: '0 4px', flexShrink: 0 }} />
}

export function TimelineToolbar({ tool, onToolChange, onFit }: {
  tool: TimelineTool
  onToolChange: (t: TimelineTool) => void
  onFit: () => void
}) {
  const snapping = useStore((s) => s.snapping)
  const setSnapping = useStore((s) => s.setSnapping)
  const clipHeight = useStore((s) => s.clipHeight)
  const setClipHeight = useStore((s) => s.setClipHeight)
  const range = useStore((s) => s.range)
  const setRange = useStore((s) => s.setRange)
  const addSkipRange = useStore((s) => s.addSkipRange)
  const anchorThreshold = useStore((s) => s.anchorThreshold)
  const setAnchorThreshold = useStore((s) => s.setAnchorThreshold)
  const hasAnchors = useStore((s) => (s.anchors?.anchors.length ?? 0) > 0)
  const zoom = useStore((s) => s.zoom)
  const setZoom = useStore((s) => s.setZoom)
  const toast = useStore((s) => s.toast)

  const exclude = (): void => {
    addSkipRange().catch((err: unknown) => {
      toast('error', err instanceof Error ? err.message : 'Failed to exclude range')
    })
  }

  return (
    <div
      role="toolbar"
      aria-label="Timeline tools"
      style={{
        height: 'var(--toolbar-h)', flexShrink: 0, display: 'flex', alignItems: 'center', gap: 8,
        padding: '0 8px', background: 'var(--bg-raised)', borderBottom: '1px solid var(--border)',
        fontSize: 11, color: 'var(--text-dim)', userSelect: 'none', overflow: 'hidden',
      }}
    >
      <div style={groupStyle}>
        <Segmented<TimelineTool>
          ariaLabel="Tool"
          value={tool}
          onChange={onToolChange}
          options={[
            { value: 'select', label: 'Select', title: 'Select tool: click clips, drag edges to trim' },
            { value: 'range', label: 'Range', title: 'Range tool: drag on the lanes to mark a range' },
          ]}
        />
        <IconButton
          name="magnet"
          title={`Snapping (N) — ${snapping ? 'on' : 'off'}`}
          active={snapping}
          onClick={() => setSnapping(!snapping)}
        />
        <Segmented<ClipHeight>
          ariaLabel="Clip height"
          value={clipHeight}
          onChange={setClipHeight}
          options={[
            { value: 'S', label: 'S', title: 'Small clips' },
            { value: 'M', label: 'M', title: 'Medium clips' },
            { value: 'L', label: 'L', title: 'Large clips' },
          ]}
        />
      </div>

      <Divider />

      <div style={{ ...groupStyle, flex: 1, minWidth: 0 }}>
        <Button
          size="sm"
          disabled={!range}
          onClick={exclude}
          title={`Exclude the range from the dub — keeps the original audio (${MOD}E)`}
          style={{ height: 24 }}
        >
          <Icon name="ban" size={13} />
          Exclude from dub
        </Button>
        <Button
          size="sm"
          variant="ghost"
          disabled={!range}
          onClick={() => setRange(null)}
          title="Clear the range (Alt+X)"
          style={{ height: 24 }}
        >
          Clear range
        </Button>
        <div style={{ flex: 1 }} />
        <label
          title="Hide scene/silence anchors below this confidence"
          style={{ display: 'flex', alignItems: 'center', gap: 6, opacity: hasAnchors ? 1 : 0.5 }}
        >
          <span>anchors</span>
          <input
            type="range"
            aria-label="Anchor threshold"
            min={0}
            max={1}
            step={0.05}
            value={anchorThreshold}
            onChange={(e) => setAnchorThreshold(Number(e.target.value))}
            style={{ ...sliderStyle, width: 72 }}
          />
        </label>
      </div>

      <Divider />

      <div style={groupStyle}>
        <IconButton name="minus" title="Zoom out (−)" onClick={() => setZoom(zoom / ZOOM_FACTOR)} />
        <input
          type="range"
          aria-label="Zoom"
          min={0}
          max={1}
          step={0.001}
          value={zoomToSlider(zoom)}
          onChange={(e) => setZoom(sliderToZoom(Number(e.target.value)))}
          style={sliderStyle}
        />
        <IconButton name="plus" title="Zoom in (+)" onClick={() => setZoom(zoom * ZOOM_FACTOR)} />
        <IconButton name="fit" title="Fit whole timeline (Shift+Z)" onClick={onFit} />
      </div>
    </div>
  )
}
