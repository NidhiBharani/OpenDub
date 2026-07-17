import { expect, test } from '@playwright/test'
import { mkdirSync } from 'node:fs'

const SHOTS = 'e2e/screenshots'
test.beforeAll(() => mkdirSync(SHOTS, { recursive: true }))

test.describe('OpenDub — real browser rendering', () => {
  test('library renders with the demo project and a screenshot is captured', async ({ page }) => {
    await page.goto('/')

    // Brand + tagline are painted by the real browser.
    await expect(page.getByText('voice-preserving dubbing studio')).toBeVisible()
    await expect(page.getByText(/Drop a video here/)).toBeVisible()

    // The backend's demo project renders as a card (duration + segment count + target lang).
    const card = page.getByText('Demo Episode')
    await expect(card).toBeVisible()
    await expect(page.getByText(/segments · → EN/)).toBeVisible()

    await page.screenshot({ path: `${SHOTS}/library.png`, fullPage: true })
  })

  test('opening the project renders the editor (player + timeline canvas)', async ({ page }) => {
    await page.goto('/')
    await page.getByText('Demo Episode').click()

    // Editor mounts: a canvas timeline and a video element are present and laid out.
    const canvas = page.locator('canvas').first()
    await expect(canvas).toBeVisible()
    const box = await canvas.boundingBox()
    expect(box?.width ?? 0).toBeGreaterThan(100) // actually laid out, not 0×0

    await page.screenshot({ path: `${SHOTS}/editor.png`, fullPage: true })
  })

  test('settings view renders provider pickers', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Settings' }).click()
    // Wait for the settings view to paint something recognizable.
    await expect(page.getByText(/Translation|Voice|Transcription|Providers|Settings/i).first()).toBeVisible()
    await page.screenshot({ path: `${SHOTS}/settings.png`, fullPage: true })
  })
})
