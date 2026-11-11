#!/usr/bin/env python3
"""Run and watch matches locally, on the JAX engine.

This is the canonical rules path: no TypeScript involved, and your real Python
environment, so JAX and a trained checkpoint work exactly as they do in training.

    # play against a built-in, and load your own .py from the page
    python server/local_server.py --open

    # watch two of your agents, as fast as they can go
    python server/local_server.py --p0 my_agent.py:act --p1 other.py:act --tick 0

    # 100-game series, no browser
    python server/local_server.py --p0 expander --p1 random --games 100 --headless

Agent specs on the command line: a builtin (`expander`, `random`, `pass`),
`human`, `module:function`, `file.py:function`, or an `http(s)://` endpoint.
From the page you can also just pick a `.py` file; it is imported in this
process, so it gets the same interpreter your training code runs in.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from genmod import EnvConfig, reset, step                        # noqa: E402
from genmod.config import DIRS, NUM_CELL_ACTIONS                 # noqa: E402
from genmod.state import army_counts, land_counts                # noqa: E402
from server.agents_registry import (                             # noqa: E402
    Agent,
    AgentLoadError,
    builtin_agents,
    load_from_source,
    validate,
)
from server.bot_protocol import BotState, load_agent, normalise   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
UI = ROOT / "server" / "local_ui.html"
BOARD_JS = ROOT / "shared" / "board.mjs"
MAX_UPLOAD_BYTES = 2_000_000


# --------------------------------------------------------------------------- #
def to_bot_state(state, cfg: EnvConfig, player: int,
                 owner=None, army=None, mountain=None) -> BotState:
    n = cfg.num_players
    return BotState(
        h=cfg.height, w=cfg.width, n=n, tick=int(state.step),
        owner=np.asarray(state.owner) if owner is None else owner,
        army=np.asarray(state.army) if army is None else army,
        mountain=np.asarray(state.mountain) if mountain is None else mountain,
        castle=np.asarray(state.castle),
        general=np.asarray(state.general),
        general_idx=np.asarray(state.general_idx),
        alive=np.asarray(state.alive),
        land=np.asarray(land_counts(state, n)),
        army_total=np.asarray(army_counts(state, n)),
        config={
            "generalGrowthInterval": cfg.general_growth_interval,
            "landGrowthInterval": cfg.land_growth_interval,
            "maxTicks": cfg.max_steps,
            "minSpawnDist": cfg.min_spawn_dist,
            "tickMs": 500,
        },
    )


def snapshot(state, cfg: EnvConfig, last_moves) -> dict:
    n = cfg.num_players
    return {
        "h": cfg.height, "w": cfg.width, "n": n,
        "tick": int(state.step), "done": bool(state.done), "winner": int(state.winner),
        "owner": np.asarray(state.owner).reshape(-1).astype(int).tolist(),
        "army": np.asarray(state.army).reshape(-1).astype(int).tolist(),
        "mountain": np.asarray(state.mountain).reshape(-1).astype(int).tolist(),
        "castle": np.asarray(state.castle).reshape(-1).astype(int).tolist(),
        "general": np.asarray(state.general).reshape(-1).astype(int).tolist(),
        "alive": np.asarray(state.alive).astype(int).tolist(),
        "land": np.asarray(land_counts(state, n)).astype(int).tolist(),
        "armyTotal": np.asarray(army_counts(state, n)).astype(int).tolist(),
        "lastMoves": last_moves,
    }


def decode_move(action: int) -> dict | None:
    k = action % NUM_CELL_ACTIONS
    if k == 0:
        return None
    m = k - 1
    return {"from": action // NUM_CELL_ACTIONS, "dir": m % 4, "half": bool(m >= 4)}


# --------------------------------------------------------------------------- #
class Match:
    """One live game. `agents[p]` is None for the seat at the keyboard."""

    def __init__(self, cfg: EnvConfig, agents: list[Agent], seed: int, tick_ms: int):
        self.cfg = cfg
        self.tick_ms = tick_ms
        self.seed = seed
        self.lock = threading.Lock()
        # Keyboard/UI queue edits must not wait for an agent search. `advance`
        # holds `lock` while the synchronous rules/bot path runs; queue state
        # has its own short critical sections so a thinking bot cannot freeze
        # the human input thread.
        self.queue_lock = threading.Lock()
        self.agent_defs = agents
        self.agents: list[Any] = [None if a.kind == "human" else a.fn for a in agents]
        self.names = [a.label for a in agents]
        self.queues: list[deque[int]] = [deque() for _ in agents]
        self.notes: list[str | None] = [a.error for a in agents]
        self.state = reset(jax.random.PRNGKey(seed), cfg)
        self.last_moves: list[dict | None] = [None] * len(agents)
        self.version = 0
        self.started = time.time()

    @property
    def human_seats(self) -> list[int]:
        return [i for i, a in enumerate(self.agents) if a is None]

    # -- input ------------------------------------------------------------
    def queue_moves(self, player: int, moves: list[dict]) -> None:
        with self.queue_lock:
            q = self.queues[player]
            for m in moves[:64]:
                if len(q) >= 64:
                    break
                q.append(
                    int(m["from"]) * NUM_CELL_ACTIONS + 1 + int(m["dir"])
                    + (4 if m.get("half") else 0)
                )

    def clear_queue(self, player: int) -> None:
        with self.queue_lock:
            self.queues[player].clear()

    def queued_moves(self, player: int) -> list[int]:
        with self.queue_lock:
            return list(self.queues[player])

    def queue_lengths(self) -> list[int]:
        with self.queue_lock:
            return [len(q) for q in self.queues]

    # -- clock ------------------------------------------------------------
    def _legal(self, host: dict, player: int, action: int) -> bool:
        """Legality against host arrays, so a tick costs no device transfer."""
        if action <= 0 or action >= self.cfg.num_cells * NUM_CELL_ACTIONS:
            return False
        cell, k = divmod(action, NUM_CELL_ACTIONS)
        if k == 0 or host["owner"][cell] != player or host["army"][cell] < 2:
            return False
        r, c = divmod(cell, self.cfg.width)
        dr, dc = DIRS[(k - 1) % 4]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < self.cfg.height and 0 <= cc < self.cfg.width):
            return False
        return not host["mountain"][rr * self.cfg.width + cc]

    def _action_for(self, host: dict, bot_states: list, player: int) -> int:
        agent = self.agents[player]
        with self.queue_lock:
            needs_action = agent is not None and not self.queues[player]
        if needs_action:
            try:
                bs = bot_states[player]
                # Never hold queue_lock while running user code. A human may
                # queue or clear moves while the other seat is thinking.
                planned = [int(a) for a in normalise(agent(bs, player), bs)[:32]]
                with self.queue_lock:
                    if not self.queues[player]:
                        self.queues[player].extend(planned)
                self.notes[player] = None
            except Exception as exc:
                self.notes[player] = f"{type(exc).__name__}: {exc}"

        with self.queue_lock:
            while self.queues[player]:
                action = self.queues[player].popleft()
                if self._legal(host, player, action):
                    self.last_moves[player] = decode_move(action)
                    return action
                # A stale path is meaningless once its source changes hands.
                self.queues[player].clear()
        self.last_moves[player] = None
        return 0

    def advance(self) -> None:
        with self.lock:
            if bool(self.state.done):
                return
            n = self.cfg.num_players
            owner = np.asarray(self.state.owner)
            army = np.asarray(self.state.army)
            mountain = np.asarray(self.state.mountain)
            host = {
                "owner": owner.reshape(-1),
                "army": army.reshape(-1),
                "mountain": mountain.reshape(-1),
            }
            bot_states = [
                to_bot_state(self.state, self.cfg, p, owner, army, mountain)
                if self.agents[p] is not None else None
                for p in range(n)
            ]
            actions = [self._action_for(host, bot_states, p) for p in range(n)]
            self.state, _r, _d, _i = step(self.state, jnp.asarray(actions, jnp.int32), self.cfg)
            self.version += 1

    def view(self) -> dict:
        with self.lock:
            return {
                "version": self.version,
                "tickMs": self.tick_ms,
                "seed": self.seed,
                "names": self.names,
                "kinds": [a.kind for a in self.agent_defs],
                "notes": self.notes,
                "humanSeats": self.human_seats,
                "snapshot": snapshot(self.state, self.cfg, self.last_moves),
                "queueLengths": self.queue_lengths(),
            }

    def run(self, stop: threading.Event) -> None:
        """Used by the headless series runner; the server uses `Session` instead."""
        interval = self.tick_ms / 1000.0
        nxt = time.perf_counter()
        while not stop.is_set():
            self.advance()
            if bool(self.state.done):
                return
            if interval <= 0:
                continue
            nxt += interval
            time.sleep(max(0.0, nxt - time.perf_counter()))


# --------------------------------------------------------------------------- #
class Session:
    """Holds the agent registry and the match currently being played."""

    def __init__(self, cfg: EnvConfig, agents: list[Agent], seed: int, tick_ms: int,
                 allow_upload: bool):
        self.allow_upload = allow_upload
        self.lock = threading.Lock()
        self.registry: dict[str, Agent] = {a.id: a for a in builtin_agents()}
        for a in agents:
            if a.id not in self.registry:
                self.registry[a.id] = a
        self.default_cfg = cfg
        self.match = Match(cfg, agents, seed, tick_ms)
        self.epoch = 0

    # -- registry ---------------------------------------------------------
    def agents_json(self) -> list[dict]:
        return [a.to_json() for a in self.registry.values()]

    def add_uploaded(self, source: str, filename: str, func: str | None) -> Agent:
        agent = load_from_source(source, filename, func)
        sample = to_bot_state(self.match.state, self.match.cfg, 0)
        validate(agent, sample, 0)
        with self.lock:
            self.registry[agent.id] = agent
        return agent

    def resolve(self, agent_id: str) -> Agent:
        agent = self.registry.get(agent_id)
        if agent is None:
            raise AgentLoadError(f"unknown agent {agent_id!r}")
        return agent

    # -- matches ----------------------------------------------------------
    def new_match(self, p0: str, p1: str, cfg: EnvConfig, seed: int, tick_ms: int) -> None:
        agents = [self.resolve(p0), self.resolve(p1)]
        with self.lock:
            self.match = Match(cfg, agents, seed, tick_ms)
            self.epoch += 1

    def view(self) -> dict:
        with self.lock:
            match = self.match
            epoch = self.epoch
        v = match.view()
        v["epoch"] = epoch
        v["agents"] = self.agents_json()
        v["allowUpload"] = self.allow_upload
        v["seats"] = [a.id for a in match.agent_defs]
        v["config"] = {
            "size": match.cfg.height,
            "tick": match.tick_ms,
            "minSpawn": match.cfg.min_spawn_dist,
            "maxTicks": match.cfg.max_steps,
        }
        return v

    def run(self, stop: threading.Event) -> None:
        """Tick whichever match is current; a new one simply takes over."""
        nxt = time.perf_counter()
        while not stop.is_set():
            with self.lock:
                match = self.match
            match.advance()
            interval = match.tick_ms / 1000.0
            if interval <= 0:
                if bool(match.state.done):
                    time.sleep(0.05)
                continue
            nxt = max(nxt + interval, time.perf_counter() - interval)
            time.sleep(max(0.0, nxt - time.perf_counter()))


# --------------------------------------------------------------------------- #
def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _body(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code: int, payload: dict) -> None:
            self._body(code, json.dumps(payload).encode(), "application/json")

        def _read_json(self) -> dict:
            length = int(self.headers.get("content-length", 0))
            if length > MAX_UPLOAD_BYTES:
                raise AgentLoadError("that file is too large")
            return json.loads(self.rfile.read(length) or b"{}")

        # -- GET ----------------------------------------------------------
        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/":
                self._body(200, UI.read_bytes(), "text/html; charset=utf-8")
            elif path == "/board.mjs":
                self._body(200, BOARD_JS.read_bytes(), "text/javascript; charset=utf-8")
            elif path == "/api/state":
                self._json(200, session.view())
            elif path == "/api/stream":
                self._stream()
            else:
                self._json(404, {"error": "not found"})

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream; charset=utf-8")
            self.send_header("cache-control", "no-cache")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()

            def chunk(text: str) -> None:
                raw = text.encode()
                self.wfile.write(f"{len(raw):X}\r\n".encode() + raw + b"\r\n")
                self.wfile.flush()

            last = None
            try:
                while True:
                    view = session.view()
                    stamp = (view["epoch"], view["version"])
                    if stamp != last:
                        last = stamp
                        chunk(f"event: state\ndata: {json.dumps(view)}\n\n")
                    else:
                        chunk(": ping\n\n")
                    time.sleep(max(0.02, min(0.2, view["tickMs"] / 4000)))
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                except Exception:
                    pass
                self.close_connection = True

        # -- POST ---------------------------------------------------------
        def do_POST(self):  # noqa: N802
            path = self.path.split("?")[0]
            try:
                body = self._read_json()
            except AgentLoadError as exc:
                self._json(413, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(400, {"error": f"bad request: {exc}"})
                return

            if path == "/api/move":
                self._move(body)
            elif path == "/api/agent":
                self._upload(body)
            elif path == "/api/new":
                self._new(body)
            else:
                self._json(404, {"error": "not found"})

        def _move(self, body: dict) -> None:
            match = session.match
            player = int(body.get("player", -1))
            if player not in match.human_seats:
                self._json(403, {"error": "that seat is not yours"})
                return
            if body.get("clear"):
                match.clear_queue(player)
            else:
                match.queue_moves(player, body.get("moves", []))
            self._json(200, {"ok": True, "queue": len(match.queues[player])})

        def _upload(self, body: dict) -> None:
            if not session.allow_upload:
                self._json(
                    403,
                    {"error": "uploads are disabled when the server is not on localhost"},
                )
                return
            source = body.get("source")
            filename = str(body.get("filename") or "agent.py")
            if not isinstance(source, str) or not source.strip():
                self._json(400, {"error": "no file contents received"})
                return
            try:
                agent = session.add_uploaded(source, filename, body.get("func") or None)
            except AgentLoadError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(200, {"agent": agent.to_json(), "agents": session.agents_json()})

        def _new(self, body: dict) -> None:
            try:
                size = max(5, min(32, int(body.get("size", session.default_cfg.height))))
                tick = max(0, min(5000, int(body.get("tick", 500))))
                min_spawn = max(2, min(size - 4, int(body.get("minSpawn", 17))))
                max_ticks = max(50, min(20000, int(body.get("maxTicks", 2000))))
                seed = int(body.get("seed", int(time.time() * 1000) & 0x7FFFFFFF))
                cfg = EnvConfig(
                    height=size, width=size, min_spawn_dist=min_spawn,
                    max_spawn_dist=10**9, max_steps=max_ticks, bfs_iters=4 * size,
                )
                session.new_match(str(body.get("p0", "human")), str(body.get("p1")), cfg, seed, tick)
            except AgentLoadError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(200, session.view())

        def log_message(self, *args):
            pass

    return Handler


# --------------------------------------------------------------------------- #
def spec_to_agent(spec: str) -> Agent:
    """Turn a command-line agent spec into a registry entry.

    Accepts the registry ids the page shows (`builtin:expander`) as well as the
    short CLI form (`expander`), so a spec copied from either place works.
    """
    if spec in ("human", "you"):
        return Agent(id="human", label="You (keyboard)", kind="human")
    if spec.startswith("builtin:"):
        spec = spec.split(":", 1)[1]
    from genmod.bots import BOTS

    if spec in BOTS:
        return Agent(id=f"builtin:{spec}", label=f"{spec} (built-in)", kind="builtin",
                     fn=load_agent(spec), func_name=spec)
    return Agent(id=f"cli:{spec}", label=spec, kind="uploaded", fn=load_agent(spec))


def run_series(cfg: EnvConfig, specs: list[str], games: int, seed: int) -> None:
    """Score a headless series -- the fastest way to compare two agents."""
    agents = [spec_to_agent(s) for s in specs]
    if any(a.kind == "human" for a in agents):
        raise SystemExit("--headless needs an agent in every seat (no 'human')")
    wins = [0] * cfg.num_players
    draws = 0
    ticks = 0
    t0 = time.perf_counter()

    for g in range(games):
        match = Match(cfg, agents, seed + g, tick_ms=0)
        match.run(threading.Event())
        w = int(match.state.winner)
        if w < 0:
            draws += 1
        else:
            wins[w] += 1
        ticks += int(match.state.step)
        print(
            f"\rgame {g + 1}/{games}  "
            + "  ".join(f"{specs[p]}: {wins[p]}" for p in range(cfg.num_players))
            + f"  draws: {draws}",
            end="", flush=True,
        )
    dt = time.perf_counter() - t0
    print(f"\n{ticks} ticks in {dt:.1f}s  ({ticks / dt:,.0f} ticks/s)")
    for p in range(cfg.num_players):
        print(f"  {specs[p]:<28} {wins[p]:>4} wins  ({wins[p] / games:.1%})")
    print(f"  {'draws':<28} {draws:>4}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--p0", default="human", help="agent spec for seat 0")
    ap.add_argument("--p1", default="expander", help="agent spec for seat 1")
    ap.add_argument("--size", type=int, default=23)
    ap.add_argument("--tick", type=int, default=500, help="ms per tick; 0 = as fast as possible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-spawn", type=int, default=17)
    ap.add_argument("--max-ticks", type=int, default=2000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; uploads are refused off localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open a browser on start")
    ap.add_argument("--headless", action="store_true", help="no server; print the result")
    ap.add_argument("--games", type=int, default=1, help="with --headless, play a series")
    args = ap.parse_args()

    cfg = EnvConfig(
        height=args.size, width=args.size,
        min_spawn_dist=min(args.min_spawn, args.size - 4),
        max_spawn_dist=10**9, max_steps=args.max_ticks, bfs_iters=4 * args.size,
    )

    if args.headless:
        run_series(cfg, [args.p0, args.p1], args.games, args.seed)
        return

    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    session = Session(
        cfg, [spec_to_agent(args.p0), spec_to_agent(args.p1)],
        args.seed, max(0, args.tick), allow_upload=loopback,
    )
    stop = threading.Event()
    threading.Thread(target=session.run, args=(stop,), daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    url = f"http://{args.host}:{args.port}/"
    print(f"{args.p0}  vs  {args.p1}   ({cfg.height}x{cfg.width}, {args.tick}ms/tick)")
    print(f"play at {url}")
    if not loopback:
        print("note: bound off localhost, so loading .py files from the page is disabled")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
