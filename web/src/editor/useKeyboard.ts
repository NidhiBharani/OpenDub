// Global keyboard shortcuts for the Editor view.
import { useEffect } from 'react'
import { stageStatus, useStore } from '../state/store'

const ZOOM_FACTOR = 1.3

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true
  return target.isContentEditable
}

/** Mounts a window-level keydown listener implementing the editor's keyboard shortcuts. */
export function useEditorKeyboard(): void {
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.metaKey) return
      if (isEditableTarget(e.target)) return

      const state = useStore.getState()

      switch (e.key) {
        case ' ': {
          e.preventDefault()
          state.setPlaying(!state.playing)
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
        case '1': {
          state.setAudioTrack('original')
          break
        }
        case '2': {
          // mirror the header button: the dub track only exists once mix is done
          if (stageStatus(state.project, 'mix') === 'done') state.setAudioTrack('dub')
          break
        }
        case 'Escape': {
          state.selectSegment(null)
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
