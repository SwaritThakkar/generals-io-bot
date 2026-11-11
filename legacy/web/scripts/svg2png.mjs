/** Render every SVG in a directory to a 2x PNG using headless Chrome. */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, basename } from "node:path";
import puppeteer from "puppeteer-core";

const dir = process.argv[2];
const scale = Number(process.argv[3] || 2);
const only = process.argv[4];

const browser = await puppeteer.launch({
  headless: "new",
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  args: ["--no-sandbox", "--force-device-scale-factor=" + scale],
});
const page = await browser.newPage();

const files = readdirSync(dir).filter(f => f.endsWith(".svg") && (!only || f.includes(only)));
for (const f of files) {
  const svg = readFileSync(join(dir, f), "utf8");
  const m = svg.match(/width="(\d+)" height="(\d+)"/);
  const w = Number(m[1]), h = Number(m[2]);
  await page.setViewport({ width: w, height: h, deviceScaleFactor: scale });
  await page.setContent(
    `<html><body style="margin:0;background:#fff">${svg}</body></html>`,
    { waitUntil: "domcontentloaded" }
  );
  const out = join(dir, basename(f, ".svg") + ".png");
  await page.screenshot({ path: out, clip: { x: 0, y: 0, width: w, height: h } });
  console.log(`  ${basename(out)}  ${w}x${h} @${scale}x`);
}
await browser.close();
