/**
 * Browser smoke test for the fully-local (vs bot) game.
 *
 * The engine, the map, the opponent and the clock all live in the tab, so the
 * things that can break are integration-shaped and invisible to vitest: the
 * canvas module failing to load, the clock not starting, input not reaching the
 * match, or a stray request to the server. This drives a real Chrome and checks
 * all four.
 *
 *   PORT=3012 npm run start &
 *   node scripts/smoke-local.mjs
 */

import puppeteer from "puppeteer-core";

const BASE = process.env.SMOKE_BASE ?? "http://127.0.0.1:3012";
const CHROME =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const fail = [];
const check = (label, ok, detail = "") => {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  ${detail}` : ""}`);
  if (!ok) fail.push(label);
};

const browser = await puppeteer.launch({
  headless: "new",
  executablePath: CHROME,
  args: ["--no-sandbox"],
});
const page = await browser.newPage();

const errors = [];
const failedRequests = [];
const apiCalls = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(`console: ${m.text()}`);
});
page.on("requestfailed", (r) =>
  failedRequests.push(`${r.url()} (${r.failure()?.errorText})`),
);
page.on("response", (r) => {
  if (r.status() === 404) failedRequests.push(`404 ${r.url()}`);
  if (r.url().includes("/api/")) apiCalls.push(r.url());
});

console.log(`\ndriving ${BASE}/local …\n`);
await page.goto(`${BASE}/local?bot=expander&size=13&tick=120&spawn=6&seed=42`, {
  waitUntil: "networkidle0",
});
await page.waitForFunction(() => !!window.__genmod?.match, { timeout: 10_000 });

const read = () =>
  page.evaluate(() => ({
    tick: Number(document.querySelector(".clock b")?.textContent ?? -1),
    rows: [...document.querySelectorAll(".srow")].map((el) => ({
      name: el.querySelector(".sname")?.textContent?.trim(),
      land: Number(el.querySelectorAll(".snum")[0]?.textContent?.replace("land", "").trim()),
      army: Number(el.querySelectorAll(".snum")[1]?.textContent?.replace("army", "").trim()),
    })),
    canvas: (() => {
      const c = document.querySelector("canvas");
      return c ? { w: c.width, h: c.height } : null;
    })(),
  }));

const first = await read();
check("canvas rendered", !!first.canvas && first.canvas.w > 0, JSON.stringify(first.canvas));

await new Promise((r) => setTimeout(r, 2500));
const later = await read();
check("clock advances", later.tick > first.tick, `${first.tick} -> ${later.tick}`);
check("opponent is playing", later.rows[1]?.land > 1, `land ${later.rows[1]?.land}`);

// --- human input: click our general, then queue a path with the arrow keys ---
const geo = await page.evaluate(() => {
  const m = window.__genmod.match;
  const c = document.querySelector("canvas");
  const r = c.getBoundingClientRect();
  const g = m.board.generalIdx[0];
  const w = m.board.w;
  const cell = r.width / w;
  const dirs = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];
  const row = Math.floor(g / w);
  const col = g % w;
  const dir = dirs.findIndex(([dr, dc]) => {
    const rr = row + dr;
    const cc = col + dc;
    return rr >= 0 && rr < m.board.h && cc >= 0 && cc < w && !m.board.mountain[rr * w + cc];
  });
  return {
    x: r.x + (col + 0.5) * cell,
    y: r.y + (row + 0.5) * cell,
    dir,
    land: m.scoreboard().land[0],
    army: m.board.army[g],
    general: g,
  };
});

// Let the general build up enough army to be worth moving.
await page.waitForFunction(
  () => window.__genmod.match.board.army[window.__genmod.match.board.generalIdx[0]] >= 6,
  { timeout: 15_000 },
);

await page.mouse.click(geo.x, geo.y);
const KEY = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"][geo.dir];
await page.keyboard.press(KEY);
const queued = await page.evaluate(() => window.__genmod.match.queueLength(0));
check("keypress reaches the match", queued >= 1, `queue=${queued}`);

// The cursor must travel with the path, so a second key chains off the first.
await page.keyboard.press(KEY);
const chain = await page.evaluate(() => {
  const q = window.__genmod.match.pendingMoves(0);
  const w = window.__genmod.match.board.w;
  const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  if (q.length < 2) return { len: q.length, chained: false };
  const [a, b] = q;
  const [dr, dc] = dirs[a.dir];
  const dest = (Math.floor(a.from / w) + dr) * w + ((a.from % w) + dc);
  return { len: q.length, chained: b.from === dest, dest, second: b.from };
});
check(
  "arrow keys chain into a path",
  chain.len === 2 && chain.chained,
  `queue=${chain.len} second.from=${chain.second} expected=${chain.dest}`,
);

await new Promise((r) => setTimeout(r, 1200));
const afterMove = await page.evaluate(() => ({
  land: window.__genmod.match.scoreboard().land[0],
  queue: window.__genmod.match.queueLength(0),
}));
check("queued path executed", afterMove.land > geo.land, `land ${geo.land} -> ${afterMove.land}`);
check("queue drained", afterMove.queue === 0, `queue=${afterMove.queue}`);

// Draining the queue must not snap the cursor back to where we started: the
// next key has to act from where the army ended up.
const dirs4 = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"];
let afterDrain = { from: -1 };
for (const key of [KEY, ...dirs4]) {
  await page.keyboard.press(key);
  afterDrain = await page.evaluate(() => {
    const q = window.__genmod.match.pendingMoves(0);
    return { from: q[0]?.from ?? -1 };
  });
  if (afterDrain.from !== -1) break;
}
const owned = await page.evaluate(
  (cell) => cell >= 0 && window.__genmod.match.board.owner[cell] === 0,
  afterDrain.from,
);
check(
  "cursor stays with the army after the queue drains",
  afterDrain.from !== -1 && afterDrain.from !== geo.general && owned,
  `next move starts at ${afterDrain.from}, general is ${geo.general}, owned=${owned}`,
);

check("no server requests at all", apiCalls.length === 0, apiCalls.slice(0, 3).join(", "));
check("no failed requests", failedRequests.length === 0, failedRequests.slice(0, 3).join(", "));
check("no page errors", errors.length === 0, errors.slice(0, 3).join(" | "));

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed\n` : "\nall checks passed\n");
process.exit(fail.length ? 1 : 0);
