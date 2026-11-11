/**
 * Dynamics for perfect-information Generals.io.
 *
 * A faithful mirror of `genmod/env.py`. The JAX version resolves every cell on
 * every tick because that is what a GPU wants; here only the handful of cells
 * that actually receive an incoming stack can change, so the same rules cost a
 * few microseconds per tick. The two are proven equivalent by the golden tests.
 *
 * Simultaneous resolution, in order:
 *   1. Departures  -- every legal mover's units leave first; a source keeps >= 1
 *                     unit and keeps its owner.
 *   2. Arrivals    -- each player's force at a destination is what they sent in
 *                     plus any garrison they already held there.
 *   3. Resolution  -- the largest force survives with (largest - second largest).
 *                     A zero remainder leaves the cell with its previous owner.
 *                     A neutral garrison then absorbs the remainder; only a
 *                     strict surplus captures the cell.
 */

import {
  Board,
  CELL_ACTIONS,
  DIRS,
  GameConfig,
  Scoreboard,
  cloneBoard,
} from "./types";

export interface StepResult {
  /** Whose general fell on this tick. */
  captured: number[];
  /** True on the tick the game ended. */
  justFinished: boolean;
}

interface Scratch {
  sent: Int32Array;
  src: Int32Array;
  dst: Int32Array;
  force: Float64Array;
}

function scratchFor(n: number): Scratch {
  return {
    sent: new Int32Array(n),
    src: new Int32Array(n),
    dst: new Int32Array(n),
    force: new Float64Array(n),
  };
}

const scratchCache = new Map<number, Scratch>();
function getScratch(n: number): Scratch {
  let s = scratchCache.get(n);
  if (!s) {
    s = scratchFor(n);
    scratchCache.set(n, s);
  }
  return s;
}

/** Is `action` a move this player may legally make right now? */
export function isLegal(b: Board, player: number, action: number): boolean {
  if (b.done || !b.alive[player]) return false;
  const k = action % CELL_ACTIONS;
  if (k === 0) return false;
  const cell = Math.floor(action / CELL_ACTIONS);
  if (cell < 0 || cell >= b.h * b.w) return false;
  if (b.owner[cell] !== player || b.army[cell] < 2) return false;
  const m = k - 1;
  const [dr, dc] = DIRS[m % 4];
  const r = Math.floor(cell / b.w) + dr;
  const c = (cell % b.w) + dc;
  if (r < 0 || r >= b.h || c < 0 || c >= b.w) return false;
  return !b.mountain[r * b.w + c];
}

/** Every legal action for `player`, as flat action indices. */
export function legalActions(b: Board, player: number): number[] {
  const out: number[] = [];
  if (b.done || !b.alive[player]) return out;
  for (let cell = 0; cell < b.h * b.w; cell++) {
    if (b.owner[cell] !== player || b.army[cell] < 2) continue;
    const r = Math.floor(cell / b.w);
    const c = cell % b.w;
    for (let d = 0; d < 4; d++) {
      const rr = r + DIRS[d][0];
      const cc = c + DIRS[d][1];
      if (rr < 0 || rr >= b.h || cc < 0 || cc >= b.w) continue;
      if (b.mountain[rr * b.w + cc]) continue;
      out.push(cell * CELL_ACTIONS + 1 + d);
      out.push(cell * CELL_ACTIONS + 5 + d);
    }
  }
  return out;
}

/** Advance one tick, mutating `b`. `actions[p]` is a flat action index. */
export function stepInPlace(
  b: Board,
  actions: ArrayLike<number>,
  cfg: GameConfig,
): StepResult {
  const empty: StepResult = { captured: [], justFinished: false };
  if (b.done) return empty;

  const { h, w, n } = b;
  const hw = h * w;
  const s = getScratch(n);

  // --- 1. departures ------------------------------------------------------
  for (let p = 0; p < n; p++) {
    s.sent[p] = 0;
    s.src[p] = -1;
    s.dst[p] = -1;
    const a = actions[p] | 0;
    if (!isLegal(b, p, a)) continue;
    const cell = Math.floor(a / CELL_ACTIONS);
    const m = (a % CELL_ACTIONS) - 1;
    const [dr, dc] = DIRS[m % 4];
    const dst = (Math.floor(cell / w) + dr) * w + ((cell % w) + dc);
    const army = b.army[cell];
    const sent = m >= 4 ? Math.floor(army / 2) : army - 1;
    s.src[p] = cell;
    s.dst[p] = dst;
    s.sent[p] = sent;
    b.army[cell] -= sent;
  }

  // --- 2 & 3. arrivals and combat, only where something landed ------------
  for (let p = 0; p < n; p++) {
    const cell = s.dst[p];
    if (cell < 0) continue;
    // Resolve each destination once, even when several players target it.
    let first = true;
    for (let q = 0; q < p; q++) if (s.dst[q] === cell) first = false;
    if (!first) continue;

    const owner = b.owner[cell];
    const garrison = b.army[cell];
    for (let q = 0; q < n; q++) {
      s.force[q] = (owner === q ? garrison : 0) + (s.dst[q] === cell ? s.sent[q] : 0);
    }

    let best = 0;
    let bestVal = s.force[0];
    let secondVal = -Infinity;
    for (let q = 1; q < n; q++) {
      const v = s.force[q];
      if (v > bestVal) {
        secondVal = bestVal;
        bestVal = v;
        best = q;
      } else if (v > secondVal) {
        secondVal = v;
      }
    }
    if (n === 1) secondVal = 0;
    const strength = bestVal - secondVal;

    if (owner < 0) {
      // A neutral garrison absorbs the remainder; only a surplus captures.
      const after = strength - garrison;
      b.owner[cell] = after > 0 ? best : -1;
      b.army[cell] = Math.abs(after);
    } else {
      b.owner[cell] = strength > 0 ? best : owner;
      b.army[cell] = strength;
    }
  }

  // --- general capture ----------------------------------------------------
  const captured: number[] = [];
  const remap = new Int8Array(n);
  for (let p = 0; p < n; p++) remap[p] = p;
  for (let p = 0; p < n; p++) {
    if (!b.alive[p]) continue;
    const gi = b.generalIdx[p];
    if (gi < 0) continue;
    const now = b.owner[gi];
    if (now === p) continue;
    captured.push(p);
    remap[p] = now;
  }
  if (captured.length) {
    // The captor inherits every cell of the fallen player, at half strength.
    for (let i = 0; i < hw; i++) {
      const o = b.owner[i];
      if (o < 0 || remap[o] === o) continue;
      b.army[i] = Math.ceil(b.army[i] / 2);
      b.owner[i] = remap[o];
    }
    for (const p of captured) {
      const gi = b.generalIdx[p];
      b.general[gi] = 0;
      b.castle[gi] = 1; // a fallen general becomes a castle
      b.alive[p] = 0;
      b.generalIdx[p] = -1;
    }
  }

  // --- production ---------------------------------------------------------
  b.tick += 1;
  const fast = b.tick % cfg.generalGrowthInterval === 0;
  const land = b.tick % cfg.landGrowthInterval === 0;
  if (fast || land) {
    for (let i = 0; i < hw; i++) {
      if (b.owner[i] < 0) continue;
      if (fast && (b.general[i] || b.castle[i])) b.army[i] += 1;
      if (land) b.army[i] += 1;
    }
  }

  // --- termination --------------------------------------------------------
  let nAlive = 0;
  let lastStanding = -1;
  for (let p = 0; p < n; p++)
    if (b.alive[p]) {
      nAlive++;
      lastStanding = p;
    }
  const timeout = b.tick >= cfg.maxTicks;
  if (nAlive <= 1 || timeout) {
    b.done = true;
    if (nAlive === 1) b.winner = lastStanding;
    else if (nAlive > 1 && timeout) {
      // Lexicographic (land, army); a perfect tie stays a draw.
      const sb = score(b);
      let bestLand = -1;
      let bestArmy = -1;
      let who = -1;
      let ties = 0;
      for (let p = 0; p < n; p++) {
        if (!b.alive[p]) continue;
        const l = sb.land[p];
        const a = sb.army[p];
        if (l > bestLand || (l === bestLand && a > bestArmy)) {
          bestLand = l;
          bestArmy = a;
          who = p;
          ties = 1;
        } else if (l === bestLand && a === bestArmy) ties++;
      }
      b.winner = ties === 1 ? who : -1;
    } else b.winner = -1;
  }

  return { captured, justFinished: b.done };
}

/** Advance one tick, returning a new board and leaving `b` untouched. */
export function step(
  b: Board,
  actions: ArrayLike<number>,
  cfg: GameConfig,
): { board: Board; result: StepResult } {
  const next = cloneBoard(b);
  const result = stepInPlace(next, actions, cfg);
  return { board: next, result };
}

/** Owned-cell and total-army counts per player. */
export function score(b: Board): Scoreboard {
  const land = new Array(b.n).fill(0);
  const army = new Array(b.n).fill(0);
  for (let i = 0; i < b.h * b.w; i++) {
    const o = b.owner[i];
    if (o < 0) continue;
    land[o] += 1;
    army[o] += b.army[i];
  }
  return { land, army };
}
