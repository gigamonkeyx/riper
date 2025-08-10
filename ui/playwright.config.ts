import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 90_000,
  expect: { timeout: 10_000 },
  fullyParallel: true,
  reporter: [['list']],
  use: {
    baseURL: process.env.UI_BASE_URL || 'http://localhost:3000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: [
    {
      command: process.platform === 'win32' ? 'py -3 -u ui_api.py' : 'python3 ui_api.py',
      url: 'http://localhost:8000/api/health',
      timeout: 120000,
      reuseExistingServer: true,
      cwd: '..',
    },
    {
      command: 'npm run dev -- -p 3000',
      url: 'http://localhost:3000',
      timeout: 180000,
      reuseExistingServer: true,
      cwd: '.',
    }
  ],
  projects: [
    { name: 'Chromium', use: { ...devices['Desktop Chrome'] } },
  ],
});

