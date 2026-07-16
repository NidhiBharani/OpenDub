import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// Headless component testing: React renders into jsdom (no browser, no screen).
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
