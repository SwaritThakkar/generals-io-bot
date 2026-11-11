/**
 * Client for the `genmod-bot/1` HTTP protocol.
 *
 * A bot is any URL that accepts a POST of the game state and answers with an
 * action. `server/bot_protocol.py` turns a Python agent into one in a few lines,
 * so a JAX policy needs no export step to become a playable opponent.
 *
 * Registered bots come from the `GENMOD_BOTS` environment variable:
 *
 *   GENMOD_BOTS=[{"id":"expander","name":"Expander","url":"https://…/act"}]
 *
 * A match may also carry an ad-hoc URL, so you can point the app at a laptop
 * over a tunnel without redeploying.
 */

import { Board, GameConfig, Move, PASS, decodeAction, encodeAction } from "./types";
import { isLegal } from "./engine";
import { LOCAL_BOTS, findLocalBot, safeAct } from "./localbots";
import { botStateOf } from "./wire";

export interface BotRef {
  id: string;
  name: string;
  description?: string;
  /** Set for opponents that run inside this app; `url` is then unused. */
  builtin?: string;
  url?: string;
}

/** Opponents from the environment: real agents behind a `genmod-bot/1` endpoint. */
export function remoteBots(): BotRef[] {
  const raw = process.env.GENMOD_BOTS;
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((b) => b && typeof b.url === "string")
      .map((b, i) => ({
        id: String(b.id ?? `bot${i}`),
        name: String(b.name ?? b.id ?? `Bot ${i}`),
        description: typeof b.description === "string" ? b.description : "Remote endpoint.",
        url: String(b.url),
      }));
  } catch {
    console.warn("GENMOD_BOTS is not valid JSON; no remote bots registered");
    return [];
  }
}

/** Everything the lobby may offer: the built-in samples first, then your agents. */
export function registeredBots(): BotRef[] {
  const builtin: BotRef[] = LOCAL_BOTS.map((b) => ({
    id: b.id,
    name: b.name,
    description: b.description,
    builtin: b.id,
  }));
  const remote = remoteBots().filter((r) => !builtin.some((b) => b.id === r.id));
  return [...builtin, ...remote];
}

interface BotReply {
  action?: number | { r: number; c: number; dir: number | string; half?: boolean };
  actions?: number[];
}

const DIR_INDEX: Record<string, number> = { up: 0, down: 1, left: 2, right: 3 };

function normalise(reply: BotReply, b: Board): number[] {
  if (Array.isArray(reply.actions)) return reply.actions.map((a) => a | 0);
  const a = reply.action;
  if (typeof a === "number") return [a | 0];
  if (a && typeof a === "object") {
    const dir = typeof a.dir === "string" ? DIR_INDEX[a.dir] : a.dir;
    if (dir === undefined || dir < 0 || dir > 3) return [PASS];
    return [encodeAction(a.r * b.w + a.c, dir, !!a.half)];
  }
  return [PASS];
}

/**
 * Ask a bot for its next move(s). Never throws: a bot that errors, times out or
 * answers with nonsense simply passes, so one flaky endpoint cannot wedge a match.
 */
export async function askBot(
  bot: BotRef,
  b: Board,
  cfg: GameConfig,
  player: number,
  deadlineMs: number,
): Promise<{ moves: Move[]; error?: string }> {
  if (bot.builtin) {
    const local = findLocalBot(bot.builtin);
    if (!local) return { moves: [], error: `unknown built-in bot ${bot.builtin}` };
    const action = safeAct(local, b, player, cfg);
    const move = action === PASS ? null : decodeAction(action);
    return { moves: move ? [move] : [] };
  }
  if (!bot.url) return { moves: [], error: `bot ${bot.id} has no endpoint` };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), deadlineMs);
  try {
    const res = await fetch(bot.url!, {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        protocol: "genmod-bot/1",
        player,
        deadlineMs,
        state: botStateOf(b, cfg, player),
      }),
    });
    if (!res.ok) return { moves: [], error: `bot ${bot.id}: HTTP ${res.status}` };
    const actions = normalise((await res.json()) as BotReply, b);
    const moves: Move[] = [];
    for (const a of actions) {
      if (a === PASS) continue;
      const m = decodeAction(a);
      if (m) moves.push(m);
      if (moves.length >= 32) break; // a queue, not a firehose
    }
    // Only the first move is validated here; later ones are checked as they pop.
    if (moves.length && !isLegal(b, player, encodeAction(moves[0].from, moves[0].dir, moves[0].half)))
      return { moves: [], error: `bot ${bot.id}: illegal move` };
    return { moves };
  } catch (e) {
    const msg = e instanceof Error && e.name === "AbortError" ? "timeout" : String(e);
    return { moves: [], error: `bot ${bot.id}: ${msg}` };
  } finally {
    clearTimeout(timer);
  }
}
