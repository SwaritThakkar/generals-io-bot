/**
 * A whole match, run inside the browser tab.
 *
 * Playing a bot needs no server: the engine, the map generator and the built-in
 * opponents are all TypeScript, so the clock is a `setInterval` and a keypress
 * takes effect on the very next tick. Going through the serverless API instead
 * costs ~300ms of function round trip plus ~1s of Redis per request, which is
 * hopeless for a 500ms tick.
 *
 * Only friend matches genuinely need the server, because two browsers have to
 * agree on the board.
 */

import { isLegal, score, stepInPlace } from "./engine";
import { LocalBot, findLocalBot, safeAct } from "./localbots";
import { generateMap } from "./mapgen";
import { Board, GameConfig, Move, PASS, encodeAction } from "./types";
import { Snapshot, botStateOf, toSnapshot } from "./wire";

export type OpponentKind = "builtin" | "endpoint";

export interface Opponent {
  name: string;
  kind: OpponentKind;
  /** For `builtin`. */
  bot?: LocalBot | null;
  /** For `endpoint`: a URL speaking `genmod-bot/1`. */
  url?: string;
}

export interface SeatInfo {
  name: string;
  kind: "human" | OpponentKind;
  note: string | null;
}

const DIRS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
] as const;

export class LocalMatch {
  readonly cfg: GameConfig;
  readonly seed: number;
  readonly seats: SeatInfo[];
  board: Board;

  private queues: Move[][];
  private lastMoves: (Move | null)[];
  private opponents: (Opponent | null)[];
  private inflight: boolean[];

  /**
   * @param humanSeat which seat the person at the keyboard holds
   * @param opponent  who fills the other seat
   */
  constructor(cfg: GameConfig, seed: number, opponent: Opponent, humanSeat = 0) {
    this.cfg = cfg;
    this.seed = seed;
    this.board = generateMap(cfg, seed);
    this.queues = [[], []];
    this.lastMoves = [null, null];
    this.inflight = [false, false];
    this.opponents = [null, null];
    this.opponents[1 - humanSeat] = opponent;
    this.seats = [0, 1].map((p) =>
      p === humanSeat
        ? { name: "You", kind: "human" as const, note: null }
        : { name: opponent.name, kind: opponent.kind, note: null },
    );
  }

  get done(): boolean {
    return this.board.done;
  }

  snapshot(): Snapshot {
    return toSnapshot(this.board, this.lastMoves);
  }

  queueLength(player: number): number {
    return this.queues[player].length;
  }

  /** The moves still waiting to be played, for drawing the pending path. */
  pendingMoves(player: number): Move[] {
    return this.queues[player].slice();
  }

  queueMove(player: number, move: Move): void {
    if (this.queues[player].length < 64) this.queues[player].push(move);
  }

  clearQueue(player: number): void {
    this.queues[player] = [];
  }

  /** Pop the next action for a seat, dropping a path that has gone stale. */
  private popAction(player: number): number {
    const q = this.queues[player];
    while (q.length) {
      const move = q.shift()!;
      const action = encodeAction(move.from, move.dir, move.half);
      if (isLegal(this.board, player, action)) {
        this.lastMoves[player] = move;
        return action;
      }
      // The source changed hands, so the rest of the path is meaningless.
      q.length = 0;
    }
    this.lastMoves[player] = null;
    return PASS;
  }

  /**
   * Kick off any remote-endpoint request whose queue has run dry.
   *
   * Deliberately not awaited by `tick`: a slow agent must not stall the clock,
   * it just misses the tick and plays on the next one.
   */
  pump(): void {
    for (let p = 0; p < 2; p++) {
      const opp = this.opponents[p];
      if (!opp || opp.kind !== "endpoint" || !opp.url) continue;
      if (this.queues[p].length || this.inflight[p] || !this.board.alive[p]) continue;
      this.inflight[p] = true;
      void this.askEndpoint(p, opp.url).finally(() => {
        this.inflight[p] = false;
      });
    }
  }

  private async askEndpoint(player: number, url: string): Promise<void> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.max(400, this.cfg.tickMs * 2));
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          protocol: "genmod-bot/1",
          player,
          deadlineMs: this.cfg.tickMs,
          state: botStateOf(this.board, this.cfg, player),
        }),
      });
      if (!res.ok) {
        this.seats[player].note = `HTTP ${res.status} from the endpoint`;
        return;
      }
      const data = (await res.json()) as { actions?: number[]; action?: number };
      const raw = Array.isArray(data.actions)
        ? data.actions
        : typeof data.action === "number"
          ? [data.action]
          : [];
      const moves: Move[] = [];
      for (const a of raw.slice(0, 32)) {
        const k = a % 9;
        if (k === 0) continue;
        const m = k - 1;
        moves.push({ from: Math.floor(a / 9), dir: m % 4, half: m >= 4 });
      }
      this.queues[player] = moves;
      this.seats[player].note = null;
    } catch (e) {
      const aborted = e instanceof Error && e.name === "AbortError";
      this.seats[player].note = aborted
        ? "the endpoint did not answer in time"
        : `could not reach the endpoint — is it running, and does it allow this origin? (${String(e)})`;
    } finally {
      clearTimeout(timer);
    }
  }

  /** Advance exactly one tick. Synchronous, so input feels immediate. */
  tick(): void {
    if (this.board.done) return;
    const actions = [0, 0];
    for (let p = 0; p < 2; p++) {
      const opp = this.opponents[p];
      if (opp?.kind === "builtin" && opp.bot) {
        // Pure function, microseconds -- no reason to queue it.
        const action = safeAct(opp.bot, this.board, p, this.cfg);
        actions[p] = action;
        this.lastMoves[p] = action === PASS ? null : decode(action);
      } else {
        actions[p] = this.popAction(p);
      }
    }
    stepInPlace(this.board, actions, this.cfg);
  }

  scoreboard() {
    return score(this.board);
  }

  /** Where a queued path leaves the cursor -- used to preview the next move. */
  cursorAfterQueue(player: number, selection: number): number {
    const q = this.queues[player];
    if (!q.length) return selection;
    const last = q[q.length - 1];
    const [dr, dc] = DIRS[last.dir];
    return (
      (Math.floor(last.from / this.board.w) + dr) * this.board.w +
      ((last.from % this.board.w) + dc)
    );
  }
}

function decode(action: number): Move {
  const m = (action % 9) - 1;
  return { from: Math.floor(action / 9), dir: m % 4, half: m >= 4 };
}

export function builtinOpponent(id: string): Opponent | null {
  const bot = findLocalBot(id);
  return bot ? { name: bot.name, kind: "builtin", bot } : null;
}

export function endpointOpponent(url: string, name?: string): Opponent {
  return { name: name?.trim() || "Your endpoint", kind: "endpoint", url };
}
