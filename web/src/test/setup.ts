import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

// jsdom lacks a few browser APIs the app touches; stub the ones components use.
afterEach(() => cleanup())

// EventSource (SSE) — the store opens one when a project is opened.
class MockEventSource {
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  addEventListener() {}
  removeEventListener() {}
  close() {}
}
// @ts-expect-error assigning a stub onto the jsdom global
globalThis.EventSource = MockEventSource

// matchMedia is referenced by some layout code paths.
if (!globalThis.matchMedia) {
  // @ts-expect-error minimal stub
  globalThis.matchMedia = () => ({
    matches: false,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })
}

// ResizeObserver — the canvas Timeline observes its container size.
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver

// jsdom implements neither canvas 2D contexts nor media playback.
HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext
Object.assign(HTMLMediaElement.prototype, {
  play: vi.fn().mockResolvedValue(undefined),
  pause: vi.fn(),
  load: vi.fn(),
})
Element.prototype.scrollIntoView = vi.fn()

// Font Loading API — Timeline re-renders once the web font is ready.
if (!document.fonts) {
  // @ts-expect-error minimal stub
  document.fonts = { ready: Promise.resolve(), addEventListener: vi.fn() }
}
