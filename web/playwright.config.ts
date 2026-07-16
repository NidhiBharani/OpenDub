import { defineConfig, devices } from '@playwright/test'

// Real-browser (headless Chromium) end-to-end tests. Drives the actual app —
// FastAPI backend + Vite frontend — and captures screenshots. No display needed.
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],

  // Auto-start both servers; reuse them if they're already running (they are, in dev).
  webServer: [
    {
      command: 'cd ../server && .venv/bin/uvicorn app.main:app --port 8000',
      url: 'http://localhost:8000/api/health',
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
    {
      command: 'npm run dev',
      url: 'http://localhost:5173',
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
  ],
})
