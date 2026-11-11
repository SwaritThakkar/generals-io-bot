/**
 * The built-in opponents must be legal, decisive, and beat a random player.
 *
 * They are the floor a trained agent should clear, so if these ever start
 * losing to random something is wrong with the engine, not the heuristics.
 */

import { describe, expect, it } from "vitest";

import { isLegal, score, stepInPlace } from "../lib/engine";
import { LOCAL_BOTS, expander, rusher, safeAct } from "../lib/localbots";
import { generateMap } from "../lib/mapgen";
import { Board, DEFAULT_CONFIG, GameConfig, PASS } from "../lib/types";

const cfg: GameConfig = { ...DEFAULT_CONFIG, height: 15, width: 15, minSpawnDist: 8, maxTicks: 700 };

function randomAction(b: Board, player: number, rand: () => number): number {
  const legal: number[] = [];
  for (let cell = 0; cell < b.h * b.w; cell++) {
    if (b.owner[cell] !== player || b.army[cell] < 2) continue;
    for (let d = 0; d < 4; d++) {
      const a = cell * 9 + 1 + d;
      if (isLegal(b, player, a)) legal.push(a);
    }
  }
  return legal.length ? legal[Math.floor(rand() * legal.length)] : PASS;
}

function mulberry(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Returns the winner, or -1 for a draw. */
function playOut(seed: number, bot: typeof expander, botSeat: number): number {
  const b = generateMap(cfg, seed);
  const rand = mulberry(seed * 7919 + 13);
  while (!b.done) {
    const actions = [0, 1].map((p) =>
      p === botSeat ? safeAct(bot, b, p, cfg) : randomAction(b, p, rand),
    );
    for (const [p, a] of actions.entries())
      if (a !== PASS) expect(isLegal(b, p, a), `illegal action from seat ${p}`).toBe(true);
    stepInPlace(b, actions, cfg);
  }
  return b.winner;
}

describe("built-in opponents", () => {
  it("offers exactly the two sample opponents", () => {
    expect(LOCAL_BOTS.map((b) => b.id)).toEqual(["expander", "rusher"]);
    for (const b of LOCAL_BOTS) {
      expect(b.name.length).toBeGreaterThan(0);
      expect(b.description.length).toBeGreaterThan(0);
    }
  });

  it("only ever emits legal actions, from either seat", () => {
    for (const bot of LOCAL_BOTS) {
      for (let seat = 0; seat < 2; seat++) {
        const b = generateMap(cfg, 4242 + seat);
        for (let t = 0; t < 200 && !b.done; t++) {
          const a = safeAct(bot, b, seat, cfg);
          if (a !== PASS) expect(isLegal(b, seat, a)).toBe(true);
          const actions = [0, 0];
          actions[seat] = a;
          stepInPlace(b, actions, cfg);
        }
      }
    }
  });

  it("actually expands rather than shuffling in place", () => {
    const b = generateMap(cfg, 99);
    for (let t = 0; t < 120 && !b.done; t++) {
      stepInPlace(b, [safeAct(expander, b, 0, cfg), PASS], cfg);
    }
    expect(score(b).land[0]).toBeGreaterThan(8);
  });

  it("beats a random player from either seat", () => {
    for (const bot of [expander, rusher]) {
      let wins = 0;
      const games = 8;
      for (let g = 0; g < games; g++) {
        const seat = g % 2;
        if (playOut(500 + g, bot, seat) === seat) wins++;
      }
      expect(wins, `${bot.id} won only ${wins}/${games}`).toBeGreaterThanOrEqual(games - 1);
    }
  });

  it("rusher converts a lead faster than expander against a passive base", () => {
    // The whole point of a rusher: it ends the game, rather than out-landing you.
    const ticksToWin = (bot: typeof expander, seed: number): number => {
      const b = generateMap(cfg, seed);
      while (!b.done) stepInPlace(b, [safeAct(bot, b, 0, cfg), PASS], cfg);
      return b.winner === 0 ? b.tick : Infinity;
    };
    let rusherFaster = 0;
    const seeds = [11, 22, 33, 44, 55];
    for (const seed of seeds) {
      const r = ticksToWin(rusher, seed);
      const e = ticksToWin(expander, seed);
      expect(r, `rusher failed to win seed ${seed}`).toBeLessThan(Infinity);
      if (r < e) rusherFaster++;
    }
    expect(rusherFaster, "rusher should close out most maps sooner").toBeGreaterThanOrEqual(3);
  });
});
