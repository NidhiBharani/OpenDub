import { expect, test } from '@playwright/test'
import { mkdirSync } from 'node:fs'

const SHOTS = 'e2e/screenshots'
test.beforeAll(() => mkdirSync(SHOTS, { recursive: true }))

test.describe('OpenDub — real browser rendering', () => {
  test('library renders with the demo project and a screenshot is captured', async ({ page }) => {
    await page.goto('/')

    // The project-browser toolbar is painted by the real browser.
    await expect(page.getByRole('button', { name: /New project/ })).toBeVisible()
    await expect(page.getByRole('textbox', { name: 'Search projects' })).toBeVisible()

    // The backend's demo project renders as a card (target lang + segment count).
    const card = page.getByRole('button', { name: 'Demo Episode' })
    await expect(card).toBeVisible()
    await expect(card.getByText(/→ EN/)).toBeVisible()
    await expect(card.getByText(/\d+ lines?/)).toBeVisible()

    await page.screenshot({ path: `${SHOTS}/library.png`, fullPage: true })
  })

  test('opening the project renders the editor (player + timeline canvas)', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Demo Episode' }).click()

    // Editor mounts: a canvas timeline and a video element are present and laid out.
    const canvas = page.locator('canvas').first()
    await expect(canvas).toBeVisible()
    const box = await canvas.boundingBox()
    expect(box?.width ?? 0).toBeGreaterThan(100) // actually laid out, not 0×0

    await page.screenshot({ path: `${SHOTS}/editor.png`, fullPage: true })
  })

  test('settings view renders the simple pipeline panel and the capability map', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Settings' }).click()

    // Simple mode is the default: runtime, preset, and the two feature toggles.
    await expect(page.getByRole('radio', { name: /Balanced/ })).toBeVisible()
    await expect(page.getByRole('checkbox', { name: /Lip sync/ })).toBeVisible()
    await page.screenshot({ path: `${SHOTS}/settings.png`, fullPage: true })

    // Advanced mode opens the 41-capability map, phase rail first.
    await page.getByRole('button', { name: 'Advanced' }).click()
    await expect(page.getByRole('navigation', { name: 'Capability phases' })).toBeVisible()
    await expect(page.getByRole('article', { name: /^A1 / })).toBeVisible()
    await page.screenshot({ path: `${SHOTS}/settings-advanced.png`, fullPage: true })
  })
})
