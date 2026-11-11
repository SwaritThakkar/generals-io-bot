/**
 * Map generation, matching the constraints in `genmod/mapgen.py`.
 *
 * This deliberately does *not* reproduce JAX's Threefry stream bit-for-bit --
 * mirroring that in TypeScript would be fragile for no benefit. What matters is
 * that both generators draw from the same distribution, which
 * `web/test/mapgen.test.ts` checks constraint by constraint. Rules-level
 * equivalence is proven separately, against explicitly supplied boards.
 */

import { Rng } from "./rng";
import { Board, DIRS, GameConfig } from "./types";

const UNREACHABLE = 1 << 28;

/** 4-connected BFS over passable cells, from a single source. */
export function bfs(
  mountain: Uint8Array,
  h: number,
  w: number,
  source: number,
): Int32Array {
  const dist = new Int32Array(h * w).fill(UNREACHABLE);
  if (mountain[source]) return dist;
  dist[source] = 0;
  const queue = new Int32Array(h * w);
  queue[0] = source;
  let head = 0;
  let tail = 1;
  while (head < tail) {
    const cur = queue[head++];
    const r = Math.floor(cur / w);
    const c = cur % w;
    const d = dist[cur] + 1;
    for (const [dr, dc] of DIRS) {
      const rr = r + dr;
      const cc = c + dc;
      if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue;
      const ni = rr * w + cc;
      if (mountain[ni] || dist[ni] <= d) continue;
      dist[ni] = d;
      queue[tail++] = ni;
    }
  }
  return dist;
}

/** BFS distance to the nearest source cell, over passable cells. */
export function bfsMulti(
  mountain: Uint8Array,
  h: number,
  w: number,
  isSource: (i: number) => boolean,
): Int32Array {
  const dist = new Int32Array(h * w).fill(UNREACHABLE);
  const queue = new Int32Array(h * w);
  let head = 0;
  let tail = 0;
  for (let i = 0; i < h * w; i++) {
    if (mountain[i] || !isSource(i)) continue;
    dist[i] = 0;
    queue[tail++] = i;
  }
  while (head < tail) {
    const cur = queue[head++];
    const r = Math.floor(cur / w);
    const c = cur % w;
    const d = dist[cur] + 1;
    for (const [dr, dc] of DIRS) {
      const rr = r + dr;
      const cc = c + dc;
      if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue;
      const ni = rr * w + cc;
      if (mountain[ni] || dist[ni] <= d) continue;
      dist[ni] = d;
      queue[tail++] = ni;
    }
  }
  return dist;
}

export const UNREACHABLE_DIST = UNREACHABLE;

function emptyBoard(cfg: GameConfig): Board {
  const hw = cfg.height * cfg.width;
  return {
    h: cfg.height,
    w: cfg.width,
    n: cfg.numPlayers,
    army: new Int32Array(hw),
    owner: new Int8Array(hw).fill(-1),
    mountain: new Uint8Array(hw),
    castle: new Uint8Array(hw),
    general: new Uint8Array(hw),
    generalIdx: new Int32Array(cfg.numPlayers).fill(-1),
    alive: new Uint8Array(cfg.numPlayers).fill(1),
    tick: 0,
    done: false,
    winner: -1,
  };
}

/**
 * Sample a map satisfying every constraint, or throw after `attempts` tries.
 * Unlike the JAX generator (which must be branch-free to stay vectorisable),
 * this can simply reject and retry, so the constraints hold exactly rather than
 * falling back to a nearest-valid placement.
 */
export function generateMap(cfg: GameConfig, seed: number, attempts = 200): Board {
  const rng = new Rng(seed);
  const { height: h, width: w, numPlayers: n } = cfg;
  const hw = h * w;

  for (let attempt = 0; attempt < attempts; attempt++) {
    const b = emptyBoard(cfg);
    for (let i = 0; i < hw; i++) b.mountain[i] = rng.next() < cfg.mountainDensity ? 1 : 0;

    const passable: number[] = [];
    for (let i = 0; i < hw; i++) if (!b.mountain[i]) passable.push(i);
    if (passable.length < hw * 0.5) continue;

    // Generals: the first anywhere, each next one inside the spawn-distance band.
    const generals: number[] = [rng.pick(passable)];
    const dists: Int32Array[] = [bfs(b.mountain, h, w, generals[0])];
    let ok = true;
    for (let p = 1; p < n; p++) {
      const prev = dists[p - 1];
      const band = passable.filter(
        (i) =>
          prev[i] >= cfg.minSpawnDist &&
          prev[i] <= cfg.maxSpawnDist &&
          !generals.includes(i),
      );
      if (!band.length) {
        ok = false;
        break;
      }
      const g = rng.pick(band);
      generals.push(g);
      dists.push(bfs(b.mountain, h, w, g));
    }
    if (!ok) continue;

    // Castles: one guaranteed near each general, the rest anywhere free.
    const target = rng.int(cfg.minCastles, cfg.maxCastles + 1);
    const castles = new Set<number>();
    for (let p = 0; p < n; p++) {
      const near = passable.filter(
        (i) =>
          !generals.includes(i) &&
          !castles.has(i) &&
          dists[p][i] >= 1 &&
          dists[p][i] <= cfg.guaranteedCastleRadius,
      );
      if (!near.length) {
        ok = false;
        break;
      }
      castles.add(rng.pick(near));
    }
    if (!ok) continue;

    const free = rng.shuffle(
      passable.filter((i) => !generals.includes(i) && !castles.has(i)),
    );
    for (const i of free) {
      if (castles.size >= target) break;
      castles.add(i);
    }
    if (castles.size < Math.min(target, cfg.minCastles)) continue;

    for (const i of castles) {
      b.castle[i] = 1;
      b.army[i] = rng.int(cfg.castleArmyMin, cfg.castleArmyMax + 1);
    }
    for (let p = 0; p < n; p++) {
      const g = generals[p];
      b.general[g] = 1;
      b.owner[g] = p;
      b.army[g] = 1;
      b.generalIdx[p] = g;
    }
    return b;
  }
  throw new Error(
    `could not generate a ${h}x${w} map for ${n} players in ${attempts} attempts; ` +
      `try lowering minSpawnDist (${cfg.minSpawnDist}) or mountainDensity`,
  );
}
