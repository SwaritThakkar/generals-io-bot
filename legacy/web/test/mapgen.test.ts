/**
 * Map generation is checked by constraint, not by byte-equality with JAX: the
 * two implementations sample the same distribution from different PRNGs.
 */

import { describe, expect, it } from "vitest";

import { bfs, generateMap } from "../lib/mapgen";
import { DEFAULT_CONFIG, GameConfig } from "../lib/types";

const cfg: GameConfig = { ...DEFAULT_CONFIG };

describe("map generation", () => {
  it("satisfies every constraint across many seeds", () => {
    for (let seed = 0; seed < 40; seed++) {
      const b = generateMap(cfg, seed);
      const hw = b.h * b.w;

      const generals = Array.from(b.generalIdx);
      expect(new Set(generals).size).toBe(cfg.numPlayers);
      for (const g of generals) {
        expect(b.mountain[g]).toBe(0);
        expect(b.army[g]).toBe(1);
      }

      let castles = 0;
      for (let i = 0; i < hw; i++) {
        if (!b.castle[i]) continue;
        castles++;
        expect(b.army[i]).toBeGreaterThanOrEqual(cfg.castleArmyMin);
        expect(b.army[i]).toBeLessThanOrEqual(cfg.castleArmyMax);
        expect(b.general[i]).toBe(0);
      }
      expect(castles).toBeGreaterThanOrEqual(cfg.minCastles);
      expect(castles).toBeLessThanOrEqual(cfg.maxCastles);

      const d0 = bfs(b.mountain, b.h, b.w, generals[0]);
      expect(d0[generals[1]]).toBeGreaterThanOrEqual(cfg.minSpawnDist);

      for (let p = 0; p < cfg.numPlayers; p++) {
        const d = bfs(b.mountain, b.h, b.w, generals[p]);
        let near = false;
        for (let i = 0; i < hw; i++)
          if (b.castle[i] && d[i] <= cfg.guaranteedCastleRadius) near = true;
        expect(near, `player ${p} needs a castle within radius`).toBe(true);
      }
    }
  });

  it("is reproducible from its seed", () => {
    const a = generateMap(cfg, 12345);
    const b = generateMap(cfg, 12345);
    expect(Array.from(a.owner)).toEqual(Array.from(b.owner));
    expect(Array.from(a.army)).toEqual(Array.from(b.army));
    expect(Array.from(a.mountain)).toEqual(Array.from(b.mountain));
  });

  it("honours a spawn-distance cap for the curriculum", () => {
    const near: GameConfig = { ...cfg, minSpawnDist: 4, maxSpawnDist: 8 };
    for (let seed = 0; seed < 15; seed++) {
      const b = generateMap(near, seed);
      const d = bfs(b.mountain, b.h, b.w, b.generalIdx[0]);
      expect(d[b.generalIdx[1]]).toBeGreaterThanOrEqual(4);
      expect(d[b.generalIdx[1]]).toBeLessThanOrEqual(8);
    }
  });

  it("mountain density lands near the configured fraction", () => {
    let total = 0;
    let mtn = 0;
    for (let seed = 0; seed < 20; seed++) {
      const b = generateMap(cfg, seed);
      total += b.h * b.w;
      for (let i = 0; i < b.h * b.w; i++) mtn += b.mountain[i];
    }
    expect(mtn / total).toBeGreaterThan(cfg.mountainDensity - 0.03);
    expect(mtn / total).toBeLessThan(cfg.mountainDensity + 0.03);
  });
});
