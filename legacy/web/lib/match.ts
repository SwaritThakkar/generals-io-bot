/**
 * Match lifecycle: creation, seating, and the authoritative clock.
 *
 * Vercel has no long-lived process to run a 0.5s timer, so the clock is derived
 * rather than driven: the match records when it started, the target tick is
 * `floor((now - startedAt) / tickMs)`, and any request advances the stored board
 * up to that tick. Whoever touches the match next does the work, under a lock,
 * so the result is identical no matter who asks or how often.
 */

import { BotRef, askBot } from "./bots";
import { isLegal, stepInPlace } from "./engine";
import { generateMap } from "./mapgen";
import * as store from "./store";
import { WireBoard, packBoard, toSnapshot, unpackBoard } from "./wire";
import {
  Board,
  DEFAULT_CONFIG,
  GameConfig,
  Move,
  PASS,
  encodeAction,
} from "./types";

export type MatchMode = "friend" | "bot";
export type MatchStatus = "waiting" | "running" | "finished";

export interface Seat {
  /** Display name. */
  name: string;
  /** Secret held by the browser that owns this seat; never sent to other players. */
  token: string | null;
  kind: "human" | "bot" | "open";
  bot: BotRef | null;
  joinedAt: number;
  /** Last bot error, surfaced in the UI so a broken endpoint is obvious. */
  note: string | null;
}

export interface Match {
  id: string;
  mode: MatchMode;
  status: MatchStatus;
  cfg: GameConfig;
  seed: number;
  createdAt: number;
  /** Epoch ms of tick 0. Null until every seat is filled. */
  startedAt: number | null;
  seats: Seat[];
  board: WireBoard;
  queues: Move[][];
  queueSeq: number[];
  lastMoves: (Move | null)[];
}

const key = (id: string) => `match:${id}`;
/** Ticks a single advance call will catch up at most, so a stale match can't hang a request. */
const MAX_CATCHUP = 240;

export function newId(): string {
  const alphabet = "abcdefghjkmnpqrstuvwxyz23456789";
  let s = "";
  for (let i = 0; i < 6; i++) s += alphabet[Math.floor(Math.random() * alphabet.length)];
  return s;
}

export function newToken(): string {
  return Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2);
}

export interface CreateOptions {
  mode: MatchMode;
  cfg?: Partial<GameConfig>;
  seed?: number;
  hostName?: string;
  /** For `mode: "bot"`, the opponent to seat. */
  bot?: BotRef | null;
}

export async function createMatch(opts: CreateOptions): Promise<{ match: Match; token: string }> {
  const cfg: GameConfig = { ...DEFAULT_CONFIG, ...(opts.cfg ?? {}) };
  const seed = opts.seed ?? (Math.random() * 0xffffffff) >>> 0;
  const board = generateMap(cfg, seed);
  const token = newToken();

  const seats: Seat[] = [
    {
      name: opts.hostName?.trim() || "Player 1",
      token,
      kind: "human",
      bot: null,
      joinedAt: Date.now(),
      note: null,
    },
  ];
  for (let p = 1; p < cfg.numPlayers; p++) {
    const bot = opts.mode === "bot" ? (opts.bot ?? null) : null;
    seats.push({
      name: bot ? bot.name : "Waiting…",
      token: null,
      kind: bot ? "bot" : "open",
      bot,
      joinedAt: 0,
      note: null,
    });
  }

  const match: Match = {
    id: newId(),
    mode: opts.mode,
    status: "waiting",
    cfg,
    seed,
    createdAt: Date.now(),
    startedAt: null,
    seats,
    board: packBoard(board),
    queues: seats.map(() => []),
    queueSeq: seats.map(() => 0),
    lastMoves: seats.map(() => null),
  };
  maybeStart(match);
  await store.set(key(match.id), match);
  return { match, token };
}

export async function loadMatch(id: string): Promise<Match | null> {
  return store.get<Match>(key(id));
}

function maybeStart(m: Match): void {
  if (m.status !== "waiting") return;
  if (m.seats.some((s) => s.kind === "open")) return;
  m.status = "running";
  // A moment of grace so both browsers have painted before the clock runs.
  m.startedAt = Date.now() + 1500;
}

/** Seat a second human. Returns their token, or an existing token on reconnect. */
export async function joinMatch(
  id: string,
  name: string,
  token: string | null,
): Promise<{ match: Match; token: string; player: number } | { error: string }> {
  return store.lockedUpdate<Match, { match: Match; token: string; player: number } | { error: string }>(
    key(id),
    async (m) => {
      if (!m) return { result: { error: "no such match" } };

      if (token) {
        const seat = m.seats.findIndex((s) => s.token === token);
        if (seat >= 0) return { result: { match: m, token, player: seat } };
      }
      const open = m.seats.findIndex((s) => s.kind === "open");
      if (open < 0) return { result: { error: "this match is full" } };

      const fresh = newToken();
      m.seats[open] = {
        name: name.trim() || `Player ${open + 1}`,
        token: fresh,
        kind: "human",
        bot: null,
        joinedAt: Date.now(),
        note: null,
      };
      maybeStart(m);
      return { next: m, result: { match: m, token: fresh, player: open } };
    },
  );
}

export function seatOf(m: Match, token: string | null): number {
  if (!token) return -1;
  return m.seats.findIndex((s) => s.token === token);
}

/** Ticks that should have elapsed by `now`. */
export function targetTick(m: Match, now = Date.now()): number {
  if (m.status !== "running" || m.startedAt === null) return m.board.tick;
  if (now < m.startedAt) return 0;
  return Math.floor((now - m.startedAt) / m.cfg.tickMs);
}

function popAction(m: Match, board: Board, player: number): number {
  const q = m.queues[player];
  while (q.length) {
    const move = q.shift()!;
    const action = encodeAction(move.from, move.dir, move.half);
    if (isLegal(board, player, action)) {
      m.lastMoves[player] = move;
      return action;
    }
    // The browser game drops the whole path when a queued move goes stale --
    // usually the source tile changed hands and the rest is meaningless.
    q.length = 0;
  }
  m.lastMoves[player] = null;
  return PASS;
}

/**
 * Bring the stored board up to the current wall-clock tick.
 *
 * Bot seats are asked for a move whenever their queue runs dry. The request is
 * bounded by the tick budget, and a bot that misses it simply passes that tick.
 */
export async function advance(m: Match, now = Date.now()): Promise<boolean> {
  if (m.status !== "running") return false;
  const target = Math.min(targetTick(m, now), m.cfg.maxTicks);
  if (target <= m.board.tick) return false;

  const board = unpackBoard(m.board);
  const budget = Math.max(1, Math.min(target - board.tick, MAX_CATCHUP));
  let changed = false;

  for (let i = 0; i < budget && !board.done; i++) {
    // Top up any bot queue that has run dry. Built-in opponents are pure
    // functions and cost nothing, so they always answer; a remote endpoint is
    // only worth a round trip while we are keeping pace with the clock.
    const keepingPace = target - board.tick <= 2;
    for (let p = 0; p < m.seats.length; p++) {
      const seat = m.seats[p];
      if (seat.kind !== "bot" || !seat.bot || m.queues[p].length || !board.alive[p]) continue;
      if (!seat.bot.builtin && !keepingPace) continue;
      const { moves, error } = await askBot(
        seat.bot,
        board,
        m.cfg,
        p,
        Math.floor(m.cfg.tickMs * 0.7),
      );
      m.queues[p] = moves;
      seat.note = error ?? null;
    }

    const actions = m.seats.map((_, p) => popAction(m, board, p));
    stepInPlace(board, actions, m.cfg);
    changed = true;
  }

  m.board = packBoard(board);
  if (board.done) {
    m.status = "finished";
  }
  return changed;
}

/**
 * Bring a match up to the current tick and persist it.
 *
 * Taking the lock costs three Redis round trips, so a poll with nothing to
 * advance skips it and pays for one GET. That is the common case: a stream polls
 * several times per tick.
 */
export async function tickMatch(id: string): Promise<Match | null> {
  const current = await loadMatch(id);
  if (!current) return null;
  if (current.status !== "running" || targetTick(current) <= current.board.tick) {
    return current;
  }
  return store.lockedUpdate<Match, Match | null>(key(id), async (m) => {
    if (!m) return { result: null };
    // Re-read inside the lock: another request may have advanced it already.
    const changed = await advance(m);
    return changed ? { next: m, result: m } : { result: m };
  });
}

export async function appendMoves(
  id: string,
  token: string,
  moves: (Move & { seq: number })[],
): Promise<Match | { error: string }> {
  return store.lockedUpdate<Match, Match | { error: string }>(key(id), async (m) => {
    if (!m) return { result: { error: "no such match" } };
    const p = seatOf(m, token);
    if (p < 0) return { result: { error: "not a player in this match" } };
    await advance(m);
    for (const mv of moves) {
      if (mv.seq <= m.queueSeq[p]) continue; // already applied; a retry
      if (m.queues[p].length >= 64) break;
      m.queues[p].push({ from: mv.from, dir: mv.dir, half: !!mv.half });
      m.queueSeq[p] = mv.seq;
    }
    return { next: m, result: m };
  });
}

export async function clearQueue(id: string, token: string): Promise<Match | { error: string }> {
  return store.lockedUpdate<Match, Match | { error: string }>(key(id), async (m) => {
    if (!m) return { result: { error: "no such match" } };
    const p = seatOf(m, token);
    if (p < 0) return { result: { error: "not a player in this match" } };
    await advance(m);
    m.queues[p] = [];
    return { next: m, result: m };
  });
}

/** What a client is allowed to see. Seat tokens never leave the server. */
export function publicView(m: Match, token: string | null) {
  const player = seatOf(m, token);
  const board = unpackBoard(m.board);
  return {
    id: m.id,
    mode: m.mode,
    status: m.status,
    cfg: m.cfg,
    player,
    startedAt: m.startedAt,
    serverNow: Date.now(),
    seats: m.seats.map((s) => ({
      name: s.name,
      kind: s.kind,
      bot: s.bot ? { id: s.bot.id, name: s.bot.name } : null,
      note: s.note,
    })),
    queue: player >= 0 ? m.queues[player] : [],
    queueSeq: player >= 0 ? m.queueSeq[player] : 0,
    snapshot: toSnapshot(board, m.lastMoves),
  };
}

export type PublicMatch = ReturnType<typeof publicView>;
