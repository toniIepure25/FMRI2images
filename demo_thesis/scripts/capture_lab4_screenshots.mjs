import { chromium } from "playwright";
import { mkdir } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(__dirname, "../../docs/PBT/lab 4/screenshots");

async function snap(page, name, fullPage = true) {
  const fp = path.join(OUT_DIR, name);
  await page.screenshot({ path: fp, fullPage });
  console.log("Saved:", fp);
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1400, height: 900 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  // 1 — Home
  console.log("1. Home page...");
  await page.goto("http://localhost:4173/", { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);
  await snap(page, "f1_screen_01_home.png");

  // 2 — Pipeline: trial selection
  console.log("2. Pipeline — trial selection...");
  await page.goto("http://localhost:4173/pipeline", { waitUntil: "networkidle" });
  await page.waitForTimeout(3000);
  await snap(page, "f1_screen_02_pipeline_selection.png");

  // 3 — Select first trial card
  console.log("3. Select first trial card...");
  const cards = page.locator("div.group.relative.cursor-pointer");
  await cards.first().waitFor({ state: "visible", timeout: 10000 });
  const box = await cards.first().boundingBox();
  if (box) {
    await page.mouse.click(box.x + box.width / 2, box.y + 60);
  }
  await page.waitForTimeout(1500);
  await snap(page, "f1_screen_03_pipeline_selected.png");

  // 4 — Click "Start Decoding Pipeline"
  console.log("4. Start Decoding Pipeline...");
  const startBtn = page.getByRole("button", { name: /start decoding pipeline/i });
  await startBtn.scrollIntoViewIfNeeded();
  await startBtn.click();
  await page.waitForTimeout(4000);
  await snap(page, "f1_screen_04_pipeline_encoding.png");

  // 5 — Encoding in progress (more time)
  console.log("5. Encoding progress...");
  await page.waitForTimeout(5000);
  await snap(page, "f1_screen_05_pipeline_retrieval.png");

  // 6 — Scroll to Top-K results
  console.log("6. Top-K candidates...");
  await page.waitForTimeout(5000);
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(1000);
  await snap(page, "f1_screen_06_pipeline_topk.png");

  // 7 — Click "Proceed to Reconstruction"
  console.log("7. Proceed to Reconstruction...");
  const proceedBtn = page.getByRole("button", { name: /proceed to reconstruction/i });
  try {
    await proceedBtn.waitFor({ state: "visible", timeout: 10000 });
    await proceedBtn.click();
  } catch {
    console.log("   Proceed button not found, trying skip...");
    const skipBtn = page.getByRole("button", { name: /skip/i });
    try {
      await skipBtn.click();
      await page.waitForTimeout(1000);
      const proceedBtn2 = page.getByRole("button", { name: /proceed to reconstruction/i });
      await proceedBtn2.waitFor({ state: "visible", timeout: 5000 });
      await proceedBtn2.click();
    } catch {
      console.log("   Could not proceed, taking screenshot anyway");
    }
  }
  await page.waitForTimeout(3000);
  await snap(page, "f1_screen_07_pipeline_reconstruction.png");

  // 8 — Grand reveal
  console.log("8. Grand Reveal...");
  await page.waitForTimeout(5000);
  await page.evaluate(() => window.scrollTo(0, 400));
  await page.waitForTimeout(500);
  await snap(page, "f1_screen_08_pipeline_reveal.png");

  // 9 — Metrics
  console.log("9. Metrics...");
  await page.waitForTimeout(3000);
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(500);
  await snap(page, "f1_screen_09_pipeline_metrics.png");

  // 10 — Explorer
  console.log("10. Explorer...");
  await page.goto("http://localhost:4173/explorer", { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);
  await snap(page, "f1_screen_10_explorer.png");

  // 11 — Challenge
  console.log("11. Challenge...");
  await page.goto("http://localhost:4173/challenge", { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);
  await snap(page, "f1_screen_11_challenge.png");

  await browser.close();
  console.log("All screenshots captured.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
