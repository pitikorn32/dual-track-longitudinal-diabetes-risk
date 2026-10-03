import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readdir } from 'node:fs/promises';

const published = [
  ['1 year', 'CatBoost', '0.2085', 'EBM', '0.2170', '3.86%'],
  ['2 years', 'Logistic', '0.3169', 'CatBoost', '0.3326', '7.07%'],
  ['3 years', 'XGBoost', '0.3964', 'XGBoost', '0.4087', '10.27%'],
  ['4 years', 'Logistic', '0.4727', 'CatBoost', '0.4758', '13.46%'],
  ['5 years', 'GEE', '0.5282', 'CatBoost', '0.5226', '16.49%'],
];

test('published results remain available without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4173/dist/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Dual-Track Longitudinal Modeling of Diabetes Risk');
  const rows = page.locator('#results-table tbody tr');
  for (let i = 0; i < published.length; i += 1) {
    await expect(rows.nth(i).locator('th, td')).toHaveText(published[i]);
  }
  await expect(page.getByRole('link', { name: 'Read the paper' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Download BibTeX' })).toBeVisible();
  await expect(page.locator('.result-explorer')).toBeHidden();
  await context.close();
});

test('horizon selection preserves the published numbers and rounding', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('./');
  for (let i = 0; i < published.length; i += 1) {
    await page.getByRole('button', { name: published[i][0], exact: true }).click();
    await expect(page.locator('#screening-model')).toHaveText(published[i][1]);
    await expect(page.locator('#screening-ap')).toHaveText(published[i][2]);
    await expect(page.locator('#monotonic-model')).toHaveText(published[i][3]);
    await expect(page.locator('#monotonic-ap')).toHaveText(published[i][4]);
    await expect(page.locator('#prevalence')).toHaveText(published[i][5]);
  }
  await expect(page.locator('#delta-ap')).toHaveText('−0.0056');
  await page.getByRole('button', { name: '3 years', exact: true }).click();
  await expect(page.locator('#delta-ap')).toHaveText('+0.0122');
  await expect(page.locator('#roc')).toHaveText('0.8157');
  await expect(page.locator('#brier')).toHaveText('0.0761');
  expect(errors).toEqual([]);
});

test('figures expand and return keyboard focus when closed', async ({ page }) => {
  await page.goto('./');
  const expand = page.locator('[data-figure="pipeline"]');
  await expand.click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('img')).toHaveAttribute('src', /\/dist\/assets\/images\/study-pipeline.png$/);
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(expand).toBeFocused();
  await page.getByText('View the original paper comparison figure', { exact: true }).click();
  await page.locator('[data-figure="comparison"]').click();
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('img')).toHaveAttribute('src', /track-comparison.png$/);
  await page.getByRole('button', { name: 'Close figure' }).click();
  await expect(dialog).toBeHidden();
});

test('citation copies and handles denied clipboard access', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('./');
  await page.getByRole('button', { name: 'Copy BibTeX' }).click();
  await expect(page.getByRole('status')).toHaveText('BibTeX copied to clipboard.');
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied).toContain('@inproceedings{khlaisamniang2026dualtrack');
  expect(copied).toContain('year = {2026}');
  await page.evaluate(() => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: async () => { throw new Error('Permission denied'); } },
    });
  });
  await page.getByRole('button', { name: 'Copy BibTeX' }).click();
  await expect(page.getByRole('status')).toContainText('download the BibTeX file');
});

test('assets and downloads work from a project subdirectory', async ({ page, request }) => {
  const externalAssets = [];
  page.on('request', (req) => {
    if (new URL(req.url()).hostname !== '127.0.0.1') externalAssets.push(req.url());
  });
  await page.goto('./');
  const assets = await page.locator('img[src], link[rel="stylesheet"], script[src], a[href^="assets/"]').evaluateAll((nodes) => [...new Set(nodes.map((node) => node.src || node.href))]);
  for (const url of assets) {
    expect(new URL(url).pathname).toMatch(/^\/dist\/assets\/|^\/dist\/(styles.css|site.js)$/);
    const response = await request.get(url);
    expect(response.ok(), url).toBeTruthy();
    if (url.endsWith('.pdf')) expect((await response.body()).subarray(0, 5).toString()).toBe('%PDF-');
  }
  expect(externalAssets).toEqual([]);
  expect((await readdir(new URL('../dist/', import.meta.url))).sort()).toEqual(['assets', 'index.html', 'site.js', 'styles.css']);
  const files = await readdir(new URL('../dist/assets/', import.meta.url), { recursive: true });
  expect(files.some((file) => /highlight|internal|datasets|joblib|pkl/.test(file))).toBeFalsy();
});

test('layout and interactions have no automated accessibility violations', async ({ page }) => {
  await page.goto('./');
  for (const width of [360, 390, 768, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
    const scan = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
    expect(scan.violations).toEqual([]);
  }
  await page.locator('[data-figure="pipeline"]').click();
  const dialogScan = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
  expect(dialogScan.violations).toEqual([]);
});
