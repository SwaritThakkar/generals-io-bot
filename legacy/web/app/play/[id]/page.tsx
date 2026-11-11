"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";

interface Move {
  from: number;
  dir: number;
  half: boolean;
}

interface Snapshot {
  h: number; w: number; n: number; tick: number; done: boolean; winner: number;
  owner: number[]; army: number[]; mountain: number[]; castle: number[];
  general: number[]; alive: number[]; land: number[]; armyTotal: number[];
  lastMoves: (Move | null)[];
}

interface Seat {
  name: string;
  kind: "human" | "bot" | "open";
  bot: { id: string; name: string } | null;
  note: string | null;
}

interface MatchView {
  id: string;
  mode: "friend" | "bot";
  status: "waiting" | "running" | "finished";
  cfg: { tickMs: number; maxTicks: number; height: number; width: number };
  player: number;
  startedAt: number | null;
  serverNow: number;
  seats: Seat[];
  queue: Move[];
  queueSeq: number;
  snapshot: Snapshot;
}

interface BoardApi {
  update(s: Snapshot): void;
  setPlayer(p: number): void;
  setServerQueue(seq: number, moves: Move[]): void;
  clearQueue(): void;
  queueLength(): number;
  destroy(): void;
  relayout(): void;
}

const PLAYER_VAR = ["--p0", "--p1", "--p2", "--p3"];

export default function Play({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const boardRef = useRef<BoardApi | null>(null);
  const tokenRef = useRef<string | null>(null);
  const pendingRef = useRef<(Move & { seq: number })[]>([]);
  const flushingRef = useRef(false);

  const [view, setView] = useState<MatchView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  /** Set once the join settles, either with a seat or as a genuine spectator. */
  const [joined, setJoined] = useState(false);
  const [copied, setCopied] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);

  // ---- send queued moves, coalescing bursts of keypresses into one request --
  const flush = useCallback(async () => {
    if (flushingRef.current || !tokenRef.current) return;
    const batch = pendingRef.current.splice(0, pendingRef.current.length);
    if (!batch.length) return;
    flushingRef.current = true;
    try {
      const res = await fetch(`/api/match/${id}/move`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ token: tokenRef.current, moves: batch }),
      });
      if (res.ok) {
        const v: MatchView = await res.json();
        boardRef.current?.setServerQueue(v.queueSeq, v.queue);
      }
    } catch {
      // The server owns the queue; the next tick's snapshot re-syncs us.
    } finally {
      flushingRef.current = false;
      if (pendingRef.current.length) void flush();
    }
  }, [id]);

  const clear = useCallback(async () => {
    if (!tokenRef.current) return;
    pendingRef.current = [];
    await fetch(`/api/match/${id}/move`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ token: tokenRef.current, clear: true }),
    }).catch(() => {});
  }, [id]);

  // ---- join, then load the shared canvas module ---------------------------
  useEffect(() => {
    let board: BoardApi | null = null;
    let cancelled = false;

    (async () => {
      const stored = localStorage.getItem(`genmod.token.${id}`);
      const name = localStorage.getItem("genmod.name") ?? "";
      let joined: MatchView | null = null;
      try {
        const res = await fetch(`/api/match/${id}/join`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ name, token: stored }),
        });
        const data = await res.json();
        if (res.ok) {
          tokenRef.current = data.token;
          localStorage.setItem(`genmod.token.${id}`, data.token);
          joined = data.match;
        } else if (stored) {
          tokenRef.current = stored;
        } else {
          // Full match, or none: fall through to spectating.
          setError(data.error ?? null);
        }
      } catch (e) {
        setError(String(e));
      }
      if (cancelled) return;
      // Only now may the stream open. Connecting first would hand the server no
      // token, and publicView would seat us as a spectator until the stream
      // recycled 45s later -- which is exactly what used to happen.
      setJoined(true);

      const mod = await import(
        /* webpackIgnore: true */ /* turbopackIgnore: true */ `${location.origin}/board.mjs`
      );
      if (cancelled || !canvasRef.current) return;
      board = mod.createBoard(canvasRef.current, {
        onAppend: (m: Move & { seq: number }) => {
          pendingRef.current.push(m);
          void flush();
        },
        onClear: () => void clear(),
      }) as BoardApi;
      boardRef.current = board;
      if (joined) {
        board.setPlayer(joined.player);
        board.update(joined.snapshot);
        board.setServerQueue(joined.queueSeq, joined.queue);
        setView(joined);
      }
    })();

    return () => {
      cancelled = true;
      board?.destroy();
      boardRef.current = null;
    };
  }, [id, flush, clear]);

  // ---- authoritative stream ----------------------------------------------
  useEffect(() => {
    if (!joined) return;
    let source: EventSource | null = null;
    let stopped = false;

    const connect = () => {
      if (stopped) return;
      const q = tokenRef.current ? `?token=${encodeURIComponent(tokenRef.current)}` : "";
      source = new EventSource(`/api/match/${id}/stream${q}`);
      source.addEventListener("open", () => setConnected(true));
      source.addEventListener("state", (ev) => {
        const v: MatchView = JSON.parse((ev as MessageEvent).data);
        setView(v);
        setError(null);
        const b = boardRef.current;
        if (b) {
          b.setPlayer(v.player);
          b.update(v.snapshot);
          b.setServerQueue(v.queueSeq, v.queue);
        }
      });
      source.addEventListener("gone", () => {
        setError("This match has expired or never existed.");
        stopped = true;
        source?.close();
      });
      source.addEventListener("bye", () => {
        // Expected: the function reached its lifetime. Reconnect promptly.
        source?.close();
        setConnected(false);
        setTimeout(connect, 60);
      });
      source.onerror = () => {
        setConnected(false);
        source?.close();
        if (!stopped) setTimeout(connect, 1200);
      };
    };

    connect();
    return () => {
      stopped = true;
      source?.close();
    };
  }, [id, joined]);

  // ---- pre-game countdown -------------------------------------------------
  useEffect(() => {
    if (!view || view.status !== "running" || view.startedAt === null) {
      setCountdown(null);
      return;
    }
    const skew = view.serverNow - Date.now();
    const tick = () => {
      const left = view.startedAt! - (Date.now() + skew);
      setCountdown(left > 0 ? Math.ceil(left / 1000) : null);
    };
    tick();
    const h = setInterval(tick, 100);
    return () => clearInterval(h);
  }, [view]);

  const snap = view?.snapshot;
  const me = view?.player ?? -1;
  const link = typeof window !== "undefined" ? `${location.origin}/play/${id}` : "";

  const status = (() => {
    if (error) return error;
    if (!view) return "Connecting…";
    if (view.status === "waiting") return "Waiting for an opponent to open the link…";
    if (countdown !== null) return `Starting in ${countdown}…`;
    if (view.status === "finished") {
      if (snap!.winner < 0) return "Draw — no general fell before the turn limit.";
      return `${view.seats[snap!.winner].name} wins.`;
    }
    return me >= 0 ? "Your move." : "Spectating.";
  })();

  return (
    <main className="wrap">
      <header className="masthead">
        <div>
          <div className="eyebrow">
            {view?.mode === "bot" ? "Human vs model" : "Head to head"} · no fog
          </div>
          <h1>Fogless Generals</h1>
        </div>
        <div className="clock">
          tick <b>{snap?.tick ?? 0}</b> / {view?.cfg.maxTicks ?? "—"} ·{" "}
          {connected ? "live" : "reconnecting…"}
        </div>
      </header>

      <div className="stage">
        <section className="panel" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="board-hold">
            <canvas ref={canvasRef} />
          </div>
          <div className={error ? "err" : "muted"}>{status}</div>
        </section>

        <aside className="stack">
          {view?.status === "finished" && snap && (
            <div className="panel verdict">
              <div
                className="who"
                style={{ color: snap.winner >= 0 ? `var(${PLAYER_VAR[snap.winner]})` : undefined }}
              >
                {snap.winner < 0 ? "Draw" : `${view.seats[snap.winner].name} wins`}
              </div>
              <div className="tiny">
                {snap.winner >= 0 && snap.land[1 - snap.winner] === 0
                  ? `General captured on tick ${snap.tick}`
                  : `Decided on land at the turn limit`}
              </div>
            </div>
          )}

          <div className="panel">
            <h2>Scoreboard</h2>
            <div className="stack" style={{ gap: 7 }}>
              {view?.seats.map((s, p) => (
                <div className="srow" key={p}>
                  <span className="swatch" style={{ background: `var(${PLAYER_VAR[p]})` }} />
                  <span className={`sname${snap && !snap.alive[p] ? " dead" : ""}`}>
                    {s.name}
                    {p === me && " (you)"}
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
            {view?.seats.map(
              (s, p) =>
                s.note && (
                  <div className="err" style={{ marginTop: 9 }} key={p}>
                    {s.name}: {s.note}
                  </div>
                ),
            )}
          </div>

          {view?.mode === "friend" && view.status === "waiting" && (
            <div className="panel">
              <h2>Invite</h2>
              <div className="share">
                <input readOnly value={link} onFocus={(e) => e.currentTarget.select()} />
                <button
                  onClick={() => {
                    navigator.clipboard?.writeText(link);
                    setCopied(true);
                    setTimeout(() => setCopied(false), 1600);
                  }}
                >
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
              <p className="tiny" style={{ marginTop: 8 }}>
                Send this to your friend. The clock starts a moment after they arrive.
              </p>
            </div>
          )}

          <div className="panel">
            <h2>Controls</h2>
            <p className="tiny">
              Click one of your tiles, then <kbd>↑</kbd> <kbd>↓</kbd> <kbd>←</kbd>{" "}
              <kbd>→</kbd> to queue a path. <kbd>shift</kbd> + direction sends half
              instead of all-but-one. <kbd>Q</kbd> clears the queue. Amber arrows are
              your pending path; the coloured arrows are what each side committed last
              tick.
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
