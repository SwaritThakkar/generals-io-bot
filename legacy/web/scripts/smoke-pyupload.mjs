/**
 * Browser smoke test for loading a .py agent from the local server's page.
 *
 * Covers the part vitest and the API tests cannot see: the file picker wiring,
 * the error surface for a bad file, and that starting a new game actually seats
 * the uploaded agent and runs it.
 *
 *   python server/local_server.py --p0 human --p1 expander --tick 150 &
 *   node scripts/smoke-pyupload.mjs
 */

import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import puppeteer from "puppeteer-core";

const BASE = process.env.SMOKE_BASE ?? "http://127.0.0.1:8000";
const CHROME =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const fail = [];
const check = (label, ok, detail = "") => {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  ${detail}` : ""}`);
  if (!ok) fail.push(label);
};

const dir = mkdtempSync(join(tmpdir(), "genmod-agents-"));
const goodPath = join(dir, "my_agent.py");
writeFileSync(
  goodPath,
  `import numpy as np

def act(state, player):
    """Walk the biggest stack at the first legal neighbour."""
    legal = state.legal_actions(player)
    if not legal:
        return 0
    owned = state.army * (state.owner == player)
    best = int(np.argmax(owned))
    for a in legal:
        if a // 9 == best:
            return a
    return legal[0]
`,
);
const badPath = join(dir, "oops.py");
writeFileSync(badPath, "def act(state, player):\n    return 1/0\n");

const browser = await puppeteer.launch({
  headless: "new",
  executablePath: CHROME,
  args: ["--no-sandbox"],
});
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

console.log(`\ndriving ${BASE} …\n`);
// The SSE stream keeps a connection open, so the page is never "network idle".
await page.goto(BASE, { waitUntil: "domcontentloaded" });
await page.waitForSelector("#drop");

check("setup panel present", (await page.$("#sel0")) !== null);

// --- a broken file must report its own traceback, not vanish ---------------
await (await page.$("#file")).uploadFile(badPath);
await page.waitForFunction(() => document.querySelector("#loadmsg .err"), { timeout: 30_000 });
const errText = await page.$eval("#loadmsg", (el) => el.textContent.trim());
check("a broken agent reports its error", /raised when called|ZeroDivision/i.test(errText),
  errText.slice(0, 70).replace(/\s+/g, " "));

// --- a good file loads and can be seated -----------------------------------
await (await page.$("#file")).uploadFile(goodPath);
await page.waitForFunction(() => document.querySelector("#loadmsg .ok"), { timeout: 30_000 });
const okText = await page.$eval("#loadmsg", (el) => el.textContent.trim());
check("uploaded agent loads", /my_agent\.py:act/.test(okText), okText.slice(0, 60).replace(/\s+/g, " "));

const seated = await page.$eval("#sel1", (el) => el.selectedOptions[0].textContent.trim());
check("uploaded agent is seated as Blue", /my_agent\.py:act/.test(seated), seated);

// --- start a game and confirm the agent is really playing ------------------
await page.select("#sel0", "builtin:expander");
await page.$eval("#tick", (el) => (el.value = "100"));
await page.$eval("#size", (el) => (el.value = "13"));
await page.$eval("#spawn", (el) => (el.value = "6"));
await page.click("#start");

// Wait for the match to actually swap before judging it: a previous game may
// still be running, and its tick counter would satisfy any naive check.
await page.waitForFunction(
  () => document.querySelector("#title")?.textContent?.includes("my_agent.py:act"),
  { timeout: 40_000 },
);
await page.waitForFunction(
  () => Number(document.querySelector(".clock b")?.textContent ?? 0) > 12,
  { timeout: 40_000 },
);
// Poll for the property itself rather than sampling at an arbitrary tick: how
// fast the agent takes its first tile depends on the map it was dealt.
let tookLand = true;
try {
  await page.waitForFunction(
    () => {
      const rows = [...document.querySelectorAll(".srow")];
      const land = Number(rows[1]?.querySelectorAll(".snum")[0]?.textContent?.replace("land", "").trim());
      return land > 1;
    },
    { timeout: 30_000, polling: 250 },
  );
} catch {
  tookLand = false;
}

const state = await page.evaluate(() => ({
  tick: Number(document.querySelector(".clock b")?.textContent ?? 0),
  names: [...document.querySelectorAll(".sname")].map((e) => e.textContent.trim()),
  land: [...document.querySelectorAll(".srow")].map((el) =>
    Number(el.querySelectorAll(".snum")[0]?.textContent?.replace("land", "").trim()),
  ),
  title: document.querySelector("#title")?.textContent ?? "",
}));
check("new game started with both agents", /my_agent\.py:act/.test(state.title), state.title);
check("clock is running", state.tick > 12, `tick ${state.tick}`);
check("uploaded agent is taking land", tookLand, `land ${JSON.stringify(state.land)}`);
check("no page errors", errors.length === 0, errors.slice(0, 2).join(" | "));

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed\n` : "\nall checks passed\n");
process.exit(fail.length ? 1 : 0);
