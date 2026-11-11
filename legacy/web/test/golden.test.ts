/**
 * Proves the TypeScript engine is equivalent to the JAX simulator.
 *
 * Vectors come from `python tools/export_golden.py`, which pins an explicit
 * starting board and action sequence and records the JAX board after every
 * tick. Any divergence in combat, production, capture or termination fails here.
 */

import { describe, expect, it } from "vitest";

import golden from "./golden.json";
import { score, stepInPlace } from "../lib/engine";
import { Board, GameConfig } from "../lib/types";

interface Snapshot {
  tick: number;
  done: boolean;
  winner: number;
  owner: number[];
  army: number[];
  castle: number[];
  general: number[];
  generalIdx: number[];
  alive: number[];
}

interface Case {
  name: string;
  config: GameConfig;
  initial: Snapshot & { h: number; w: number; n: number; mountain: number[] };
  actions: number[][];
  ticks: Snapshot[];
}

function boardFrom(init: Case["initial"]): Board {
  return {
    h: init.h,
    w: init.w,
    n: init.n,
    army: Int32Array.from(init.army),
    owner: Int8Array.from(init.owner),
    mountain: Uint8Array.from(init.mountain),
    castle: Uint8Array.from(init.castle),
    general: Uint8Array.from(init.general),
    generalIdx: Int32Array.from(init.generalIdx),
    alive: Uint8Array.from(init.alive),
    tick: init.tick,
    done: init.done,
    winner: init.winner,
  };
}

const cases = (golden as { cases: Case[] }).cases;

describe("TypeScript engine matches the JAX simulator", () => {
  it("ships a non-trivial vector set", () => {
    expect(cases.length).toBeGreaterThanOrEqual(15);
    expect(cases.reduce((a, c) => a + c.ticks.length, 0)).toBeGreaterThan(500);
  });

  for (const c of cases) {
    it(c.name, () => {
      const b = boardFrom(c.initial);
      for (let t = 0; t < c.actions.length; t++) {
        stepInPlace(b, c.actions[t], c.config);
        const want = c.ticks[t];
        const where = `${c.name} @ tick ${t + 1}`;

        expect(b.tick, `${where}: tick`).toBe(want.tick);
        expect(b.done, `${where}: done`).toBe(want.done);
        expect(b.winner, `${where}: winner`).toBe(want.winner);
        expect(Array.from(b.owner), `${where}: owner`).toEqual(want.owner);
        expect(Array.from(b.army), `${where}: army`).toEqual(want.army);
        expect(Array.from(b.castle), `${where}: castle`).toEqual(want.castle);
        expect(Array.from(b.general), `${where}: general`).toEqual(want.general);
        expect(Array.from(b.generalIdx), `${where}: generalIdx`).toEqual(want.generalIdx);
        expect(Array.from(b.alive), `${where}: alive`).toEqual(want.alive);
      }
    });
  }

  it("keeps the scoreboard in sync with the board", () => {
    for (const c of cases) {
      const b = boardFrom(c.initial);
      for (let t = 0; t < c.actions.length; t++) stepInPlace(b, c.actions[t], c.config);
      const sb = score(b);
      let land = 0;
      for (let i = 0; i < b.h * b.w; i++) if (b.owner[i] >= 0) land++;
      expect(sb.land.reduce((a, x) => a + x, 0)).toBe(land);
    }
  });
});
