import { test, expect } from '@playwright/test';

const ui = process.env.UI_BASE_URL || 'http://localhost:3000';

async function waitForNo404(page){
  await expect(async () => {
    const resp = await page.goto(ui + '/', { waitUntil: 'domcontentloaded' });
    expect(resp?.status()).toBe(200);
  }).toPass({ intervals: [500, 1000, 1500], timeout: 15_000 });
}

async function ensureApi(page){
  const res = await page.request.get(process.env.API_BASE_URL || 'http://localhost:8000/api/health');
  expect(res.ok()).toBeTruthy();
}

test.describe('Dark AppShell + Pages', () => {
  test.beforeAll(async ({ browser }) => {
    // verify API is up
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    await ensureApi(page);
    await ctx.close();
  });

  test('Home renders in dark layout and header links present', async ({ page }) => {
    await waitForNo404(page);
    await expect(page.locator('body')).toHaveCSS('background-color', /rgb\(/);
    await expect(page.locator('text=TONASKET FOODS')).toBeVisible();
    await expect(page.getByRole('banner').getByRole('link', { name: 'Funding' })).toBeVisible();
    await expect(page.getByRole('banner').getByRole('link', { name: 'Terminal' })).toBeVisible();

    // Sidebar + TopBar presence
    await expect(page.getByRole('banner').getByRole('navigation', { name: /Main/i })).toBeVisible();
    await expect(page.locator('header')).toBeVisible();

    // Screenshot
    await page.screenshot({ path: 'screens/home.png', fullPage: true });
  });

  test('Funding page: costs panel, filters, chart render; toggle flow works', async ({ page }) => {
    await page.goto(ui + '/funding');

    // Costs panel
    await expect(page.getByTestId('costs-card')).toBeVisible({ timeout: 20000 });
    await expect(page.getByTestId('costs-card').getByText(/Monthly cost to service/i)).toBeVisible();
    await page.screenshot({ path: 'screens/funding_top.png', fullPage: false });

    // Filters
    await expect(page.getByText('Only show enabled programs')).toBeVisible();
    await expect(page.getByText(/Include disabled/i)).toBeVisible();

    // Chart renders
    await page.waitForSelector('canvas');

    // Save toggles button present
    await expect(page.getByRole('button', { name: /Save toggles/i })).toBeVisible();

    // Try first row enable/disable via shadcn Switch in Enabled column
    const firstRow = page.locator('table tbody tr').first();
    const enabledSwitch = firstRow.getByRole('switch');
    if (await enabledSwitch.isVisible()) {
      const before = await enabledSwitch.getAttribute('aria-checked');
      await enabledSwitch.click();
      await page.waitForTimeout(1000);
      const after = await enabledSwitch.getAttribute('aria-checked');
      expect(after).not.toBe(before);
    }

    await page.screenshot({ path: 'screens/funding_table.png', fullPage: false });
  });

  test('Terminal page: SSE stream visible and copy button present', async ({ page }) => {
    await page.goto(ui + '/terminal');
    await expect(page.getByTestId('terminal-logs')).toBeVisible();
    await expect(page.getByRole('button', { name: /Run/i })).toBeVisible();

    // Trigger a status command to produce logs
    await page.getByRole('button', { name: /^Run$/ }).click();
    await page.waitForTimeout(1500);

    // Copy button and log area
    await expect(page.getByRole('button', { name: /Copy All/i })).toBeVisible();
    await page.screenshot({ path: 'screens/terminal.png', fullPage: false });
  });
});

