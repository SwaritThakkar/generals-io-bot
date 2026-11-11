/**
 * The in-browser match driver: this is the whole vs-bot game path, so it needs
 * to hold up without a server anywhere in sight.
 */

import { describe, expect, it, vi } from "vitest";

import { LocalMatch, builtinOpponent, endpointOpponent } from "../lib/localmatch";
import { encodeAction } from "../lib/types";
import { DEFAULT_CONFIG, GameConfig } from "../lib/types";

const cfg: GameConfig = {
  ...DEFAULT_CONFIG,
  height: 13,
  width: 13,
  minSpawnDist: 6,
  maxTicks: 900,
};

function newMatch(seed = 7, bot = "expander") {
  return new LocalMatch(cfg, seed, builtinOpponent(bot)!, 0);
}

describe("local (in-tab) matches", () => {
  it("seats a human and the chosen opponent", () => {
    const m = newMatch(1, "rusher");
    expect(m.seats[0].kind).toBe("human");
    expect(m.seats[1].kind).toBe("builtin");
    expect(m.seats[1].name).toBe("Rusher");
  });

  it("plays to a decisive finish against a passive human", () => {
    for (const bot of ["expander", "rusher"]) {
      const m = newMatch(11, bot);
      let guard = 0;
      while (!m.done && guard++ < cfg.maxTicks + 5) m.tick();
      expect(m.done, `${bot} never finished`).toBe(true);
      expect(m.board.winner, `${bot} should beat a player who never moves`).toBe(1);
    }
  });

  it("plays a queued path one move per tick", () => {
    const m = newMatch(3);
    const start = m.board.generalIdx[0];
    // Wait for the general to accumulate enough to move twice.
    while (m.board.army[start] < 6) m.tick();

    const w = m.board.w;
    const dirs = [
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ];
    const r = Math.floor(start / w);
    const c = start % w;
    const d = dirs.findIndex(([dr, dc]) => {
      const rr = r + dr;
      const cc = c + dc;
      return (
        rr >= 0 && rr < m.board.h && cc >= 0 && cc < w && !m.board.mountain[rr * w + cc]
      );
    });
    expect(d).toBeGreaterThanOrEqual(0);

    const before = m.scoreboard().land[0];
    m.queueMove(0, { from: start, dir: d, half: false });
    expect(m.queueLength(0)).toBe(1);
    expect(m.pendingMoves(0)).toHaveLength(1);

    m.tick();
    expect(m.queueLength(0), "one move should be consumed per tick").toBe(0);
    expect(m.scoreboard().land[0]).toBeGreaterThan(before);
  });

  it("drops a stale path instead of playing nonsense", () => {
    const m = newMatch(5);
    // A move from a cell we do not own can never be legal.
    const foreign = m.board.generalIdx[1];
    m.queueMove(0, { from: foreign, dir: 0, half: false });
    m.queueMove(0, { from: foreign, dir: 1, half: false });
    m.tick();
    expect(m.queueLength(0)).toBe(0);
  });

  it("clearQueue empties the pending path", () => {
    const m = newMatch(9);
    const g = m.board.generalIdx[0];
    m.queueMove(0, { from: g, dir: 3, half: false });
    m.queueMove(0, { from: g, dir: 3, half: true });
    expect(m.queueLength(0)).toBe(2);
    m.clearQueue(0);
    expect(m.queueLength(0)).toBe(0);
  });

  it("is reproducible from its seed", () => {
    const a = newMatch(4242);
    const b = newMatch(4242);
    for (let i = 0; i < 120; i++) {
      a.tick();
      b.tick();
    }
    expect(Array.from(a.board.owner)).toEqual(Array.from(b.board.owner));
    expect(Array.from(a.board.army)).toEqual(Array.from(b.board.army));
  });

  it("keeps ticking when a remote endpoint is slow or broken", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("connection refused"));
    vi.stubGlobal("fetch", fetchMock);
    try {
      const m = new LocalMatch(cfg, 21, endpointOpponent("https://nope.example/act"), 0);
      for (let i = 0; i < 12; i++) {
        m.pump();
        m.tick();
      }
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());
      // The clock is never blocked on the opponent, and the failure is surfaced.
      expect(m.board.tick).toBe(12);
      await vi.waitFor(() => expect(m.seats[1].note).toMatch(/could not reach the endpoint/));
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("plays the moves a working endpoint returns", async () => {
    const m = new LocalMatch(cfg, 33, endpointOpponent("https://ok.example/act"), 0);
    const theirGeneral = m.board.generalIdx[1];
    const w = m.board.w;
    const r = Math.floor(theirGeneral / w);
    const c = theirGeneral % w;
    const dirs = [
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ];
    const d = dirs.findIndex(([dr, dc]) => {
      const rr = r + dr;
      const cc = c + dc;
      return rr >= 0 && rr < m.board.h && cc >= 0 && cc < w && !m.board.mountain[rr * w + cc];
    });
    const action = encodeAction(theirGeneral, d, false);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({ actions: [action] }) }),
    );
    try {
      while (m.board.army[theirGeneral] < 4) m.tick();
      const before = m.scoreboard().land[1];
      m.pump();
      await vi.waitFor(() => expect(m.queueLength(1)).toBe(1));
      m.tick();
      expect(m.scoreboard().land[1]).toBeGreaterThan(before);
      expect(m.seats[1].note).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
