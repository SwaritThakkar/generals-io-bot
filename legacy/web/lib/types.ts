/**
 * Shared types for the perfect-information Generals.io engine.
 *
 * This is a hand-verified mirror of `genmod/env.py`. The dynamics must stay
 * bit-identical to the JAX simulator that agents train against -- `web/test/
 * golden.test.ts` replays JAX-generated games through this engine and asserts
 * equality turn by turn. Change one, change both, and regenerate the vectors
 * with `python tools/export_golden.py`.
 */

/** Direction order, fixed everywhere: 0=up, 1=down, 2=left, 3=right. */
export const DIRS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
] as const;
export const DIR_NAMES = ["up", "down", "left", "right"] as const;
export type DirName = (typeof DIR_NAMES)[number];

/** Per-cell action count: pass, plus (all-but-one | half) x 4 directions. */
export const CELL_ACTIONS = 9;
export const PASS = 0;

export const NEUTRAL = -1;

export interface GameConfig {
  height: number;
  width: number;
  numPlayers: number;
  /** Fraction of cells filled with mountains. */
  mountainDensity: number;
  minCastles: number;
  maxCastles: number;
  castleArmyMin: number;
  castleArmyMax: number;
  /** Minimum BFS distance between generals. */
  minSpawnDist: number;
  /** Maximum BFS distance between generals (the paper's spawn curriculum). */
  maxSpawnDist: number;
  /** Every player is guaranteed a neutral castle within this BFS radius. */
  guaranteedCastleRadius: number;
  /** Generals and owned castles gain +1 every this many ticks. */
  generalGrowthInterval: number;
  /** Every owned cell gains +1 every this many ticks. */
  landGrowthInterval: number;
  /** Ticks before the game is decided on land, then army, then a draw. */
  maxTicks: number;
  /** Wall-clock milliseconds per tick. 500 matches the browser game. */
  tickMs: number;
}

export const DEFAULT_CONFIG: GameConfig = {
  height: 23,
  width: 23,
  numPlayers: 2,
  mountainDensity: 0.2,
  minCastles: 9,
  maxCastles: 11,
  castleArmyMin: 40,
  castleArmyMax: 50,
  minSpawnDist: 17,
  maxSpawnDist: 1e9,
  guaranteedCastleRadius: 6,
  generalGrowthInterval: 2,
  landGrowthInterval: 50,
  maxTicks: 2000,
  tickMs: 500,
};

/**
 * A complete board. Flat typed arrays indexed by `row * width + col`, which is
 * the same layout the Python side uses, so wire payloads need no reshaping.
 */
export interface Board {
  h: number;
  w: number;
  n: number;
  army: Int32Array;
  owner: Int8Array;
  mountain: Uint8Array;
  castle: Uint8Array;
  general: Uint8Array;
  generalIdx: Int32Array;
  alive: Uint8Array;
  tick: number;
  done: boolean;
  /** -1 while unresolved or on a draw. */
  winner: number;
}

export interface Scoreboard {
  land: number[];
  army: number[];
}

/** A move expressed the way the UI thinks about it. */
export interface Move {
  from: number;
  dir: number;
  half: boolean;
}

export function idx(b: { w: number }, r: number, c: number): number {
  return r * b.w + c;
}

export function rowOf(b: { w: number }, i: number): number {
  return Math.floor(i / b.w);
}

export function colOf(b: { w: number }, i: number): number {
  return i % b.w;
}

export function encodeAction(cell: number, dir: number, half: boolean): number {
  return cell * CELL_ACTIONS + 1 + dir + (half ? 4 : 0);
}

export function decodeAction(a: number): Move | null {
  const k = a % CELL_ACTIONS;
  if (k === 0) return null;
  const m = k - 1;
  return { from: Math.floor(a / CELL_ACTIONS), dir: m % 4, half: m >= 4 };
}

export function cloneBoard(b: Board): Board {
  return {
    ...b,
    army: new Int32Array(b.army),
    owner: new Int8Array(b.owner),
    mountain: new Uint8Array(b.mountain),
    castle: new Uint8Array(b.castle),
    general: new Uint8Array(b.general),
    generalIdx: new Int32Array(b.generalIdx),
    alive: new Uint8Array(b.alive),
  };
}
