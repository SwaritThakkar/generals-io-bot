"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

interface BotInfo {
  id: string;
  name: string;
  description: string;
  builtin: boolean;
}

const SIZES = [9, 11, 15, 19, 23, 27];
const SPEEDS = [
  { ms: 1000, label: "1.0s — relaxed" },
  { ms: 500, label: "0.5s — generals.io" },
  { ms: 250, label: "0.25s — fast" },
];

export default function Lobby() {
  const router = useRouter();
  const [bots, setBots] = useState<BotInfo[]>([]);
  const [persistent, setPersistent] = useState(true);
  const [misconfigured, setMisconfigured] = useState(false);
  const [name, setName] = useState("");
  const [size, setSize] = useState(23);
  const [tickMs, setTickMs] = useState(500);
  const [minSpawn, setMinSpawn] = useState(17);
  const [botId, setBotId] = useState("");
  const [botUrl, setBotUrl] = useState("");
  const [joinCode, setJoinCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chosen = bots.find((b) => b.id === botId) ?? null;

  useEffect(() => {
    setName(localStorage.getItem("genmod.name") ?? "");
    fetch("/api/bots")
      .then((r) => r.json())
      .then((d) => {
        setBots(d.bots ?? []);
        setPersistent(d.persistent !== false);
        setMisconfigured(!!d.misconfigured);
        if (d.bots?.length) setBotId(d.bots[0].id);
      })
      .catch(() => setBots([]));
  }, []);

  /** Bot matches run entirely in the tab -- no round trip, no Redis, no lag. */
  function playLocally() {
    localStorage.setItem("genmod.name", name);
    const q = new URLSearchParams({
      size: String(size),
      tick: String(tickMs),
      spawn: String(Math.min(minSpawn, size - 4)),
    });
    if (botUrl.trim()) q.set("url", botUrl.trim());
    else q.set("bot", botId);
    router.push(`/local?${q}`);
  }

  async function create(mode: "friend" | "bot") {
    setBusy(true);
    setError(null);
    localStorage.setItem("genmod.name", name);
    try {
      const res = await fetch("/api/match", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          mode,
          name,
          size,
          tickMs,
          minSpawnDist: Math.min(minSpawn, size - 4),
          botId: botId || undefined,
          botUrl: mode === "bot" && botUrl.trim() ? botUrl.trim() : undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "could not create the match");
      localStorage.setItem(`genmod.token.${data.match.id}`, data.token);
      router.push(`/play/${data.match.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  function join() {
    const code = joinCode.trim().toLowerCase().replace(/.*\//, "");
    if (!code) return;
    localStorage.setItem("genmod.name", name);
    router.push(`/play/${code}`);
  }

  return (
    <main className="wrap">
      <header className="masthead">
        <div>
          <div className="eyebrow">Perfect-information variant</div>
          <h1>Fogless Generals</h1>
          <p className="muted" style={{ marginTop: 4 }}>
            Generals.io with the fog of war removed. Both players see the whole board —
            the only thing you don&apos;t know is what your opponent is doing{" "}
            <em>this</em> tick.
          </p>
        </div>
      </header>

      {misconfigured ? (
        <div className="err">
          <b>This deployment has no storage, so matches will not work.</b> Every request
          can land on a different instance, and the in-memory fallback is not shared
          between them — a match you create may be gone by the next tick. Add Redis in
          the Vercel dashboard (Storage → Redis), which injects{" "}
          <code>KV_REST_API_URL</code> and <code>KV_REST_API_TOKEN</code>, then redeploy.
        </div>
      ) : (
        !persistent && (
          <div className="warn">
            <b>Storage is in-memory.</b> Fine for local development — one process holds
            every match. Add a Vercel KV / Upstash Redis integration before deploying.
          </div>
        )
      )}
      {error && <div className="err">{error}</div>}

      <section className="panel">
        <h2>Match settings</h2>
        <div className="fields">
          <label>
            Your name
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Player 1"
              maxLength={24}
            />
          </label>
          <label>
            Board
            <select value={size} onChange={(e) => setSize(Number(e.target.value))}>
              {SIZES.map((s) => (
                <option key={s} value={s}>
                  {s} × {s}
                </option>
              ))}
            </select>
          </label>
          <label>
            Tick
            <select value={tickMs} onChange={(e) => setTickMs(Number(e.target.value))}>
              {SPEEDS.map((s) => (
                <option key={s.ms} value={s.ms}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Spawn distance
            <input
              type="number"
              min={2}
              max={40}
              value={minSpawn}
              onChange={(e) => setMinSpawn(Number(e.target.value))}
            />
          </label>
        </div>
        <p className="tiny" style={{ marginTop: 8 }}>
          Spawn distance is the minimum BFS gap between the two generals — 17 is the
          official value, and lowering it is the curriculum knob from the paper.
        </p>
      </section>

      <div className="grid2">
        <section className="panel stack">
          <div>
            <h2>Play a friend</h2>
            <p className="muted">
              Creates a match and gives you a link. Whoever opens it takes the second
              seat, and the clock starts once you are both in.
            </p>
          </div>
          <button className="primary" onClick={() => create("friend")} disabled={busy}>
            Create a match link
          </button>
          <div>
            <label style={{ marginBottom: 6 }}>Or join a code</label>
            <div className="share">
              <input
                value={joinCode}
                onChange={(e) => setJoinCode(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && join()}
                placeholder="abc123 or a pasted link"
              />
              <button onClick={join} disabled={busy}>
                Join
              </button>
            </div>
          </div>
        </section>

        <section className="panel stack">
          <div>
            <h2>Play a model</h2>
            <p className="muted">
              Two scripted samples ship with the app so there is always something to
              play. For your own agent, point this at any endpoint speaking{" "}
              <code>genmod-bot/1</code> — your browser calls it directly, so serve it
              over HTTPS (a tunnel) or from localhost.
            </p>
          </div>
          <label>
            Opponent
            <select value={botId} onChange={(e) => setBotId(e.target.value)}>
              {bots.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.name}
                  {b.builtin ? "" : " · your endpoint"}
                </option>
              ))}
            </select>
          </label>
          {chosen && <p className="tiny">{chosen.description}</p>}
          <label>
            Or an endpoint URL (overrides the choice above)
            <input
              value={botUrl}
              onChange={(e) => setBotUrl(e.target.value)}
              placeholder="https://your-tunnel.example/act"
            />
          </label>
          <button
            className="primary"
            onClick={playLocally}
            disabled={!botId && !botUrl.trim()}
          >
            Play this opponent
          </button>
          <p className="tiny">
            Runs entirely in your browser — nothing round-trips to the server, so
            keypresses land on the very next tick.
          </p>
        </section>
      </div>

      <section className="panel">
        <h2>Controls</h2>
        <p className="muted">
          Click one of your tiles to select it, then <kbd>↑</kbd> <kbd>↓</kbd>{" "}
          <kbd>←</kbd> <kbd>→</kbd> (or <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd>{" "}
          <kbd>D</kbd>) to queue a path — each key sends all but one unit.{" "}
          <kbd>shift</kbd> + a direction sends half instead. <kbd>Q</kbd> or{" "}
          <kbd>esc</kbd> clears the queue. Moves are queued on the server, so lag
          delays them but never drops them.
        </p>
      </section>
    </main>
  );
}
