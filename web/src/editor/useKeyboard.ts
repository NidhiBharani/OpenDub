// Global keyboard shortcuts for the Editor view (docs/plans/editor-redesign.md §7).
import { useEffect } from 'react'
import { stageStatus, useStore } from '../state/store'
import { visibleAnchors } from './timelineDraw'

const ZOOM_FACTOR = 1.3
const FORWARD_RATES = [1, 1.5, 2, 4]
const REVERSE_RATES = [0.5, 0.25] // <video> cannot play backwards: J slows down instead

/** Where the timeline skimmer is (seconds), or null when the pointer is not over the canvas.
 *  Written by Timeline, read by `I` / `O` so in/out points land under the pointer. */
export const timelinePointer: { time: number | null } = { time: null }

/** Window event the Timeline listens to for "Fit" (Shift+Z) — zoom so the whole media fits. */
export const FIT_EVENT = 'opendub:fit'

export function requestFit(): void {
  window.dispatchEvent(new CustomEvent(FIT_EVENT))
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true
  return target.isContentEditable
}

/** The segment the keyboard acts on: the selection, else the one under the playhead. */
function currentSegment(state: ReturnType<typeof useStore.getState>) {
  const segs = state.project?.segments ?? []
  if (state.selection) {
    const sel = segs.find((s) => s.id === state.selection)
    if (sel) return sel
  }
  const t = state.playhead
  return segs.find((s) => s.start <= t && t < s.end) ?? null
}

/** Mounts a window-level keydown listener implementing the editor's keyboard shortcuts. */
export function useEditorKeyboard(): void {
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (isEditableTarget(e.target)) return
      const state = useStore.getState()
      if (state.editing) return
      const mod = e.metaKey || e.ctrlKey
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key

      // ---- modifier chords ----
      if (mod) {
        if (key === 'b' && !e.shiftKey && !e.altKey) {
          // split the current segment at the playhead
          const seg = currentSegment(state)
          if (seg && state.playhead > seg.start + 0.05 && state.playhead < seg.end - 0.05) {
            e.preventDefault()
            void state.splitSegment(seg.id, state.playhead)
          }
          return
        }
        if (key === 'e' && !e.shiftKey && !e.altKey) {
          if (state.range) {
            e.preventDefault()
            void state.addSkipRange()
          }
          return
        }
        if (key === 'd' && e.shiftKey && !e.altKey) {
          e.preventDefault()
          void state.snapshotVersion()
          return
        }
        return // other ⌘/Ctrl chords belong to the browser
      }

      // ---- alt chords ----
      if (e.altKey) {
        if (key === 'x') {
          e.preventDefault()
          state.setRange(null)
          return
        }
        if (e.code === 'BracketLeft' || e.code === 'BracketRight') {
          e.preventDefault()
          const anchors = visibleAnchors(state.anchors?.anchors ?? [], state.anchorThreshold, Infinity)
          const t = state.playhead
          if (e.code === 'BracketLeft') {
            const prev = anchors.filter((a) => a.t < t - 0.01).map((a) => a.t)
            state.seek(prev.length ? Math.max(...prev) : 0)
          } else {
            const next = anchors.filter((a) => a.t > t + 0.01).map((a) => a.t)
            state.seek(next.length ? Math.min(...next) : (state.project?.media?.duration ?? t))
          }
          return
        }
        return
      }

      switch (e.key) {
        case ' ': {
          e.preventDefault()
          state.setRate(1)
          state.setPlaying(!state.playing)
          break
        }
        case 'k':
        case 'K': {
          e.preventDefault()
          state.setRate(1)
          state.setPlaying(false)
          break
        }
        case 'l':
        case 'L': {
          e.preventDefault()
          if (!state.playing || state.rate < 1) {
            state.setRate(1)
            state.setPlaying(true)
          } else {
            const i = FORWARD_RATES.indexOf(state.rate)
            state.setRate(FORWARD_RATES[Math.min(FORWARD_RATES.length - 1, i + 1)])
          }
          break
        }
        case 'j':
        case 'J': {
          e.preventDefault()
          if (!state.playing || state.rate >= 1) {
            state.setRate(REVERSE_RATES[0])
            state.setPlaying(true)
          } else {
            const i = REVERSE_RATES.indexOf(state.rate)
            state.setRate(REVERSE_RATES[Math.min(REVERSE_RATES.length - 1, i + 1)])
          }
          break
        }
        case 'ArrowLeft': {
          e.preventDefault()
          state.seek(state.playhead - (e.shiftKey ? 5 : 1))
          break
        }
        case 'ArrowRight': {
          e.preventDefault()
          state.seek(state.playhead + (e.shiftKey ? 5 : 1))
          break
        }
        case 'Home': {
          e.preventDefault()
          state.seek(0)
          break
        }
        case 'End': {
          e.preventDefault()
          state.seek(state.project?.media?.duration ?? state.playhead)
          break
        }
        case 'ArrowUp':
        case 'ArrowDown': {
          e.preventDefault()
          const segs = [...(state.project?.segments ?? [])].sort((a, b) => a.start - b.start)
          if (segs.length === 0) break
          const idx = segs.findIndex((s) => s.id === state.selection)
          const nextIdx = e.key === 'ArrowDown'
            ? (idx === -1 ? 0 : Math.min(segs.length - 1, idx + 1))
            : (idx === -1 ? segs.length - 1 : Math.max(0, idx - 1))
          state.selectSegment(segs[nextIdx].id, true)
          break
        }
        case '+':
        case '=': {
          e.preventDefault()
          state.setZoom(state.zoom * ZOOM_FACTOR)
          break
        }
        case '-': {
          e.preventDefault()
          state.setZoom(state.zoom / ZOOM_FACTOR)
          break
        }
        case 'Z': {
          if (e.shiftKey) {
            e.preventDefault()
            requestFit()
          }
          break
        }
        case 'i':
        case 'I': {
          e.preventDefault()
          state.setRangeIn(timelinePointer.time ?? state.playhead)
          break
        }
        case 'o':
        case 'O': {
          e.preventDefault()
          state.setRangeOut(timelinePointer.time ?? state.playhead)
          break
        }
        case 'x':
        case 'X': {
          e.preventDefault()
          const seg = currentSegment(state)
          if (seg) state.setRange({ start: seg.start, end: seg.end })
          break
        }
        case 'n':
        case 'N': {
          e.preventDefault()
          state.setSnapping(!state.snapping)
          break
        }
        case '1': {
          state.setAudioTrack('original')
          break
        }
        case '2': {
          // mirror the header button: the dub track only exists once mix is done
          if (stageStatus(state.project, 'mix') === 'done') state.setAudioTrack('dub')
          break
        }
        case 'Delete':
        case 'Backspace': {
          // only skip regions: deleting a line is a deliberate menu action elsewhere
          if (state.skipSelection) {
            e.preventDefault()
            void state.removeSkipRange(state.skipSelection)
          }
          break
        }
        case 'Escape': {
          if (state.range) state.setRange(null)
          else if (state.skipSelection) state.selectSkipRange(null)
          else state.selectSegment(null)
          break
        }
        default:
          break
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])
}
