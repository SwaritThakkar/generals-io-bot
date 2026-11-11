"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

import { LocalMatch, Opponent, builtinOpponent, endpointOpponent } from "@/lib/localmatch";
import { DEFAULT_CONFIG, GameConfig, Move } from "@/lib/types";
import { Snapshot } from "@/lib/wire";

interface BoardApi {
  update(s: Snapshot): void;
  setPlayer(p: number): void;
  setServerQueue(seq: number, moves: Move[]): void;
  clearQueue(): void;
  destroy(): void;
}

const PLAYER_VAR = ["--p0", "--p1"];
const ME = 0;

function num(v: string | null, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function LocalGame() {
  const params = useSearchParams();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const boardRef = useRef<BoardApi | null>(null);
  const matchRef = useRef<LocalMatch | null>(null);
  /** Highest keypress sequence handed to the match, so the board can drop its echo. */
  const lastSeqRef = useRef(0);

  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [seats, setSeats] = useState<{ name: string; note: string | null }[]>([]);
  const [paused, setPaused] = useState(false);
  const [ready, setReady] = useState(false);
  const [fatal, setFatal] = useState<string | null>(null);

  const size = Math.min(32, Math.max(5, num(params.get("size"), 23)));
  const tickMs = Math.min(5000, Math.max(50, num(params.get("tick"), 500)));
  const spawn = Math.max(2, num(params.get("spawn"), 17));
  const botId = params.get("bot") ?? "expander";
  const url = params.get("url");
  const [seed, setSeed] = useState(() => num(params.get("seed"), (Math.random() * 4294967295) >>> 0));

  const paint = useCallback(() => {
    const m = matchRef.current;
    if (!m) return;
    const s = m.snapshot();
    setSnap(s);
    setSeats(m.seats.map((x) => ({ name: x.name, note: x.note })));
    const b = boardRef.current;
    if (b) {
      b.update(s);
      // The match owns the queue; the board's optimistic echo is only there to
      // make the arrow appear before the next tick paints.
      b.setServerQueue(lastSeqRef.current, m.pendingMoves(ME));
    }
  }, []);

  // ---- build the match, then the canvas -----------------------------------
  useEffect(() => {
    let board: BoardApi | null = null;
    let cancelled = false;

    (async () => {
      const cfg: GameConfig = {
        ...DEFAULT_CONFIG,
        height: size,
        width: size,
        tickMs,
        minSpawnDist: Math.min(spawn, size - 4),
      };
      const opponent: Opponent | null = url
        ? endpointOpponent(url, params.get("name") ?? undefined)
        : builtinOpponent(botId);
      if (!opponent) {
        setFatal(`Unknown opponent "${botId}".`);
        return;
      }

      let match: LocalMatch;
      try {
        match = new LocalMatch(cfg, seed, opponent, ME);
      } catch (e) {
        setFatal(e instanceof Error ? e.message : String(e));
        return;
      }
      matchRef.current = match;
      lastSeqRef.current = 0;
      // A handle for debugging in the console, and for scripts/smoke-local.mjs.
      (window as unknown as { __genmod?: unknown }).__genmod = { match };

      const mod = await import(
        /* webpackIgnore: true */ /* turbopackIgnore: true */ `${location.origin}/board.mjs`
      );
      if (cancelled || !canvasRef.current) return;
      board = mod.createBoard(canvasRef.current, {
        onAppend: (m: Move & { seq: number }) => {
          lastSeqRef.current = m.seq;
          matchRef.current?.queueMove(ME, { from: m.from, dir: m.dir, half: m.half });
        },
        onClear: () => matchRef.current?.clearQueue(ME),
      }) as BoardApi;
      boardRef.current = board;
      board.setPlayer(ME);
      setReady(true);
      paint();
    })();

    return () => {
      cancelled = true;
      board?.destroy();
      boardRef.current = null;
    };
  }, [size, tickMs, spawn, botId, url, seed, params, paint]);

  // ---- the clock ----------------------------------------------------------
  useEffect(() => {
    if (!ready || paused) return;
    const handle = setInterval(() => {
      const m = matchRef.current;
      if (!m || m.done) return;
      m.pump(); // fire any endpoint request; never awaited
      m.tick();
      paint();
    }, tickMs);
    return () => clearInterval(handle);
  }, [ready, paused, tickMs, paint]);

  const restart = () => {
    setReady(false);
    setSnap(null);
    setSeed((Math.random() * 4294967295) >>> 0);
  };

  const done = snap?.done ?? false;
  const winner = snap?.winner ?? -1;

  return (
    <main className="wrap">
      <header className="masthead">
        <div>
          <div className="eyebrow">Runs entirely in this tab · no fog</div>
          <h1>Fogless Generals</h1>
          <p className="muted" style={{ marginTop: 4 }}>
            You vs {seats[1]?.name ?? "…"} — the engine, the map and the opponent all run
            here, so your keypresses land on the very next tick.
          </p>
        </div>
        <div className="clock">
          tick <b>{snap?.tick ?? 0}</b> · {tickMs}ms
        </div>
      </header>

      {fatal && <div className="err">{fatal}</div>}

      <div className="stage">
        <section className="panel" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="board-hold">
            <canvas ref={canvasRef} />
          </div>
          <div className="row">
            <button onClick={() => setPaused((p) => !p)} disabled={done}>
              {paused ? "Resume" : "Pause"}
            </button>
            <button onClick={restart}>New game</button>
            <span className="tiny" style={{ marginLeft: "auto" }}>
              {done
                ? "Game over."
                : paused
                  ? "Paused."
                  : `Queued: ${matchRef.current?.queueLength(ME) ?? 0}`}
            </span>
          </div>
        </section>

        <aside className="stack">
          {done && (
            <div className="panel verdict">
              <div
                className="who"
                style={{ color: winner >= 0 ? `var(${PLAYER_VAR[winner]})` : undefined }}
              >
                {winner < 0 ? "Draw" : winner === ME ? "You win" : `${seats[1]?.name} wins`}
              </div>
              <div className="tiny">
                {winner >= 0 && snap && snap.land[1 - winner] === 0
                  ? `General captured on tick ${snap.tick}`
                  : "Decided on land at the tick limit"}
              </div>
            </div>
          )}

          <div className="panel">
            <h2>Scoreboard</h2>
            <div className="stack" style={{ gap: 7 }}>
              {seats.map((s, p) => (
                <div className="srow" key={p}>
                  <span className="swatch" style={{ background: `var(${PLAYER_VAR[p]})` }} />
                  <span className={`sname${snap && !snap.alive[p] ? " dead" : ""}`}>
                    {s.name}
                  </span>
                  <span className="snum">
                    {snap?.land[p] ?? 0}
                    <small>land</small>
                  </span>
                  <span className="snum">
                    {snap?.armyTotal[p] ?? 0}
                    <small>army</small>
                  </span>
                </div>
              ))}
            </div>
            {seats.map(
              (s, p) =>
                s.note && (
                  <div className="err" style={{ marginTop: 9 }} key={p}>
                    {s.name}: {s.note}
                  </div>
                ),
            )}
          </div>

          <div className="panel">
            <h2>Controls</h2>
            <p className="tiny">
              Click one of your tiles, then <kbd>↑</kbd> <kbd>↓</kbd> <kbd>←</kbd>{" "}
              <kbd>→</kbd> to queue a path. <kbd>shift</kbd> + direction sends half
              instead of all-but-one. <kbd>Q</kbd> clears the queue.
            </p>
          </div>

          <div className="panel">
            <Link href="/">← New match</Link>
          </div>
        </aside>
      </div>
    </main>
  );
}

export default function LocalPage() {
  return (
    <Suspense fallback={<main className="wrap"><p className="muted">Loading…</p></main>}>
      <LocalGame />
    </Suspense>
  );
}
