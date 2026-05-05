import { chromium } from "playwright";
import { mkdir } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(__dirname, "../../docs/PBT/lab 5/screenshots");

async function snap(page, name) {
  const fp = path.join(OUT_DIR, name);
  await page.screenshot({ path: fp, fullPage: true });
  console.log("Saved:", fp);
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
  const page = await context.newPage();

  // 1 — Pipeline Phase 3 (F2: reconstruction with DUA-CFG)
  console.log("1. Pipeline — select trial and run through to Phase 3...");
  await page.goto("http://localhost:4173/pipeline", { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);
  const cards = page.locator("div.group.relative.cursor-pointer");
  await cards.first().waitFor({ state: "visible", timeout: 10000 });
  const box = await cards.first().boundingBox();
  if (box) await page.mouse.click(box.x + box.width / 2, box.y + 60);
  await page.waitForTimeout(1000);
  const startBtn = page.getByRole("button", { name: /start decoding pipeline/i });
  await startBtn.scrollIntoViewIfNeeded();
  await startBtn.click();

  // Skip encoding
  await page.waitForTimeout(1500);
  const skipBtn = page.getByRole("button", { name: /skip/i });
  try { await skipBtn.click(); } catch {}
  await page.waitForTimeout(1000);
  const proceedBtn = page.getByRole("button", { name: /proceed to reconstruction/i });
  try {
    await proceedBtn.waitFor({ state: "visible", timeout: 8000 });
    await proceedBtn.click();
  } catch {}
  await page.waitForTimeout(6000);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(500);
  await snap(page, "f2_screen_01_dua_cfg.png");

  // 2 — Scroll to reveal + metrics
  await page.waitForTimeout(4000);
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(500);
  await snap(page, "f2_screen_02_reconstruction_reveal.png");

  // 3 — Explorer: Reconstruction tab
  console.log("2. Explorer — Reconstruction tab...");
  await page.goto("http://localhost:4173/explorer", { waitUntil: "networkidle" });
  await page.waitForTimeout(2000);
  const reconTab = page.getByText("Reconstruction", { exact: false }).first();
  try {
    await reconTab.click();
    await page.waitForTimeout(1500);
  } catch {}
  await snap(page, "f2_screen_03_explorer_reconstruction.png");

  // 4 — Explorer: Uncertainty tab
  console.log("3. Explorer — Uncertainty tab...");
  const uncTab = page.locator('button, a, [role="tab"]').filter({ hasText: /Uncertainty/i }).first();
  try {
    await uncTab.click();
    await page.waitForTimeout(1500);
  } catch {}
  await snap(page, "f2_screen_04_explorer_uncertainty.png");

  await browser.close();
  console.log("All F2 screenshots captured.");
}

main().catch((e) => { console.error(e); process.exit(1); });
