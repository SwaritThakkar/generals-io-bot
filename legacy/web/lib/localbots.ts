/**
 * Scripted opponents that run inside the app itself.
 *
 * These exist so a fresh deployment always has something to play against, with
 * no endpoint to register and no round trip per tick. They are deliberately
 * simple heuristics, not learned policies -- they are the floor your agent
 * should clear comfortably, and a way to check the app works.
 *
 * Anything stronger belongs behind the `genmod-bot/1` protocol, where it can be
 * a real JAX policy (see `server/bot_protocol.py`).
 */

import { isLegal } from "./engine";
import { UNREACHABLE_DIST, bfs, bfsMulti } from "./mapgen";
import { Board, CELL_ACTIONS, DIRS, GameConfig, PASS } from "./types";

export interface LocalBot {
  id: string;
  name: string;
  description: string;
  act(board: Board, player: number, cfg: GameConfig): number;
}

const NO_MOVE = -Infinity;

/** Walk every legal (cell, direction) pair, keeping the best by `rank`. */
function bestMove(
  b: Board,
  player: number,
  rank: (cell: number, dir: number, target: number) => number,
): number {
  let best = PASS;
  let bestScore = NO_MOVE;
  for (let cell = 0; cell < b.h * b.w; cell++) {
    if (b.owner[cell] !== player || b.army[cell] < 2) continue;
    const r = Math.floor(cell / b.w);
    const c = cell % b.w;
    for (let d = 0; d < 4; d++) {
      const rr = r + DIRS[d][0];
      const cc = c + DIRS[d][1];
      if (rr < 0 || rr >= b.h || cc < 0 || cc >= b.w) continue;
      const target = rr * b.w + cc;
      if (b.mountain[target]) continue;
      const score = rank(cell, d, target);
      if (score > bestScore) {
        bestScore = score;
        best = cell * CELL_ACTIONS + 1 + d;
      }
    }
  }
  return bestScore === NO_MOVE ? PASS : best;
}

/** A capture we can afford right now, scored by what is left over. */
function captureScore(b: Board, player: number, cell: number, target: number): number {
  if (b.owner[target] === player) return NO_MOVE;
  const surplus = b.army[cell] - 1 - b.army[target];
  return surplus > 0 ? surplus : NO_MOVE;
}

/**
 * Take any adjacent cell we can afford; otherwise march the biggest stack down
 * a BFS field towards the nearest cell we do not own. A port of
 * `genmod.bots.expander_bot`, so local and hosted play meet the same baseline.
 */
export const expander: LocalBot = {
  id: "expander",
  name: "Expander",
  description: "Grabs whatever it can reach, then marches its biggest stack at the frontier.",
  act(b, player) {
    const capture = bestMove(b, player, (cell, _d, target) =>
      captureScore(b, player, cell, target),
    );
    if (capture !== PASS) return capture;

    const frontier = bfsMulti(b.mountain, b.h, b.w, (i) => b.owner[i] !== player);
    return bestMove(b, player, (cell, _d, target) => {
      if (frontier[target] >= frontier[cell]) return NO_MOVE;
      // Bigger stacks move first; ties break towards the closer frontier.
      return b.army[cell] * 8 - frontier[target];
    });
  },
};

/**
 * Expands, gathers, then commits.
 *
 * Expander spreads its army thin, which is exactly why it can never threaten a
 * defended base. Rusher runs the loop a human runs: expand through most of the
 * production cycle, pull army back to the general in the last stretch of it, and
 * once the pile is big enough to survive the walk and the fight, march it
 * straight at the enemy general -- which, with no fog, it can see from turn one.
 *
 * The result is a genuinely different opponent: it takes less land than
 * Expander, but it punishes a base you left undefended.
 */
export const rusher: LocalBot = {
  id: "rusher",
  name: "Rusher",
  description: "Gathers army at its base, then walks the whole pile at your general.",
  act(b, player, cfg) {
    const target = enemyGeneral(b, player);
    if (target < 0) return expander.act(b, player, cfg);

    const toEnemy = bfs(b.mountain, b.h, b.w, target);
    const strike = strikeStack(b, player, target, toEnemy);
    if (strike >= 0) {
      const move = bestMove(b, player, (cell, _d, dest) =>
        cell === strike && toEnemy[dest] < toEnemy[cell] ? -toEnemy[dest] : NO_MOVE,
      );
      if (move !== PASS) return move;
    }

    // Consolidate through the tail of each production cycle, expand the rest.
    const cycle = cfg.landGrowthInterval;
    const phase = b.tick % cycle;
    const gathering = b.tick >= cycle && phase >= cycle - Math.max(8, Math.floor(cycle * 0.3));
    if (gathering) {
      const rally = b.generalIdx[player];
      if (rally >= 0) {
        const toRally = bfs(b.mountain, b.h, b.w, rally);
        const move = bestMove(b, player, (cell, _d, dest) => {
          if (cell === rally) return NO_MOVE;
          if (toRally[dest] >= toRally[cell]) return NO_MOVE;
          // Move the fattest stack first; it is the one worth walking home.
          return b.army[cell] * 8 - toRally[dest];
        });
        if (move !== PASS) return move;
      }
    }
    return expander.act(b, player, cfg);
  },
};

function enemyGeneral(b: Board, player: number): number {
  for (let p = 0; p < b.n; p++) {
    if (p === player || !b.alive[p]) continue;
    if (b.generalIdx[p] >= 0) return b.generalIdx[p];
  }
  return -1;
}

/**
 * The stack to commit with, or -1 while we should still be building one.
 *
 * It has to survive the walk (enemy tiles on the way cost army) and still beat
 * the garrison waiting at the general.
 */
function strikeStack(
  b: Board,
  player: number,
  target: number,
  toEnemy: Int32Array,
): number {
  let best = -1;
  let bestArmy = 0;
  for (let i = 0; i < b.h * b.w; i++) {
    if (b.owner[i] !== player || b.army[i] < 2) continue;
    if (b.army[i] > bestArmy) {
      bestArmy = b.army[i];
      best = i;
    }
  }
  if (best < 0) return -1;

  const distance = toEnemy[best];
  if (distance >= UNREACHABLE_DIST) return -1;
  // One unit stays behind on every hop, and the garrison still has to be beaten.
  return bestArmy - 1 > b.army[target] + distance ? best : -1;
}

export const LOCAL_BOTS: LocalBot[] = [expander, rusher];

export function findLocalBot(id: string): LocalBot | null {
  return LOCAL_BOTS.find((b) => b.id === id) ?? null;
}

/** Never hand the engine an illegal action, whatever the heuristic produced. */
export function safeAct(bot: LocalBot, b: Board, player: number, cfg: GameConfig): number {
  const action = bot.act(b, player, cfg);
  return action !== PASS && isLegal(b, player, action) ? action : PASS;
}
