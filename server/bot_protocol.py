"""The `genmod-bot/1` protocol: turn any Python agent into a playable opponent.

An agent is a callable ``(state: BotState, player: int) -> action``. The action
may be a flat integer index, a ``(row, col, direction, half)`` tuple, or a list
of those to queue a whole path.

Serving one is a single call::

    from server.bot_protocol import serve
    from genmod.bots import expander_bot

    serve(my_agent, port=8801, name="My agent")

The web app POSTs the board to that URL every tick and plays whatever comes
back, so a JAX policy needs no weight export to become an opponent. The same
callable also works directly in `server/local_server.py`, where it is called
in-process against the JAX engine.
"""

from __future__ import annotations

import json
import pathlib
import re
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Sequence

import numpy as np

PROTOCOL = "genmod-bot/1"
CELL_ACTIONS = 9
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_NAMES = ("up", "down", "left", "right")
PASS = 0


@dataclass
class BotState:
    """One tick of the board, exactly as the server sees it. Nothing is hidden."""

    h: int
    w: int
    n: int
    tick: int
    owner: np.ndarray      # (h, w) int8, -1 neutral
    army: np.ndarray       # (h, w) int32
    mountain: np.ndarray   # (h, w) bool
    castle: np.ndarray     # (h, w) bool
    general: np.ndarray    # (h, w) bool
    general_idx: np.ndarray  # (n,) int32, -1 once captured
    alive: np.ndarray      # (n,) bool
    land: np.ndarray       # (n,) int32
    army_total: np.ndarray  # (n,) int32
    config: dict

    @classmethod
    def from_json(cls, d: dict) -> "BotState":
        h, w = int(d["h"]), int(d["w"])
        grid = lambda k, t: np.asarray(d[k], t).reshape(h, w)  # noqa: E731
        return cls(
            h=h, w=w, n=int(d["n"]), tick=int(d["tick"]),
            owner=grid("owner", np.int8), army=grid("army", np.int32),
            mountain=grid("mountain", bool), castle=grid("castle", bool),
            general=grid("general", bool),
            general_idx=np.asarray(d["generalIdx"], np.int32),
            alive=np.asarray(d["alive"], bool),
            land=np.asarray(d["land"], np.int32),
            army_total=np.asarray(d["armyTotal"], np.int32),
            config=d.get("config", {}),
        )

    # -- convenience ---------------------------------------------------------
    def owned(self, player: int) -> np.ndarray:
        return self.owner == player

    def passable(self) -> np.ndarray:
        return ~self.mountain

    def legal_actions(self, player: int) -> list[int]:
        """Every legal flat action index for `player`."""
        out: list[int] = []
        owner, army = self.owner.reshape(-1), self.army.reshape(-1)
        mtn = self.mountain
        for cell in np.flatnonzero((owner == player) & (army >= 2)):
            r, c = divmod(int(cell), self.w)
            for d, (dr, dc) in enumerate(DIRS):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < self.h and 0 <= cc < self.w) or mtn[rr, cc]:
                    continue
                out.append(int(cell) * CELL_ACTIONS + 1 + d)
                out.append(int(cell) * CELL_ACTIONS + 5 + d)
        return out

    def to_genmod_state(self, cfg):
        """Convert to a `genmod.State` so a JAX policy can act on it directly."""
        from genmod.scenario import make_state

        return make_state(
            cfg, self.owner, self.army, mountain=self.mountain,
            castle=self.castle, general=self.general, tick=self.tick,
        )

    def env_config(self):
        """A `genmod.EnvConfig` matching this match's rules."""
        from genmod import EnvConfig

        c = self.config
        return EnvConfig(
            height=self.h, width=self.w, num_players=self.n,
            general_growth_interval=int(c.get("generalGrowthInterval", 2)),
            land_growth_interval=int(c.get("landGrowthInterval", 50)),
            max_steps=int(c.get("maxTicks", 2000)),
            min_spawn_dist=int(c.get("minSpawnDist", 2)),
            bfs_iters=4 * max(self.h, self.w),
        )


def encode(row: int, col: int, direction: int | str, half: bool = False, *, width: int) -> int:
    """(row, col, direction, half) -> flat action index."""
    d = DIR_NAMES.index(direction) if isinstance(direction, str) else int(direction)
    return (row * width + col) * CELL_ACTIONS + 1 + d + (4 if half else 0)


def normalise(action: Any, state: BotState) -> list[int]:
    """Accept the several shapes an agent might return; produce flat indices."""
    if action is None:
        return []
    if isinstance(action, tuple) and len(action) == 4 and isinstance(action[0], (int, np.integer)):
        r, c, d, half = action
        return [encode(int(r), int(c), d, bool(half), width=state.w)]
    if isinstance(action, (list, tuple)) and action and isinstance(action[0], (list, tuple, dict, np.integer, int)):
        out: list[int] = []
        for a in action:
            out.extend(normalise(a, state))
        return out
    if isinstance(action, dict):
        return [encode(int(action["r"]), int(action["c"]), action.get("dir", 0),
                       bool(action.get("half", False)), width=state.w)]
    return [int(action)]


Agent = Callable[[BotState, int], Any]


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #
def serve(agent: Agent, port: int = 8801, name: str = "Python agent", host: str = "0.0.0.0") -> None:
    """Expose `agent` at http://host:port/act until interrupted."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            # The hosted app calls this from a different origin.
            self.send_header("access-control-allow-origin", "*")
            self.send_header("access-control-allow-headers", "content-type")
            self.send_header("access-control-allow-methods", "POST, GET, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802
            self._send(204, {})

        def do_GET(self):  # noqa: N802
            self._send(200, {"protocol": PROTOCOL, "name": name})

        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                state = BotState.from_json(body["state"])
                player = int(body.get("player", body["state"].get("player", 0)))
                actions = normalise(agent(state, player), state)
                self._send(200, {"actions": actions})
            except Exception as exc:  # a crashing agent must not wedge the match
                self._send(200, {"actions": [PASS], "error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args):  # keep the console readable
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"{name} listening on http://{host}:{port}/  (protocol {PROTOCOL})")
    print("Point the web app at this URL, or pass it to local_server.py with --p0/--p1.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


class HttpAgent:
    """Call a remote `genmod-bot/1` endpoint as if it were a local agent."""

    def __init__(self, url: str, timeout: float = 2.0):
        self.url = url
        self.timeout = timeout

    def __call__(self, state: BotState, player: int) -> list[int]:
        payload = {
            "protocol": PROTOCOL,
            "player": player,
            "deadlineMs": int(self.timeout * 1000),
            "state": {
                "h": state.h, "w": state.w, "n": state.n, "tick": state.tick,
                "player": player,
                "owner": state.owner.reshape(-1).astype(int).tolist(),
                "army": state.army.reshape(-1).astype(int).tolist(),
                "mountain": state.mountain.reshape(-1).astype(int).tolist(),
                "castle": state.castle.reshape(-1).astype(int).tolist(),
                "general": state.general.reshape(-1).astype(int).tolist(),
                "generalIdx": state.general_idx.astype(int).tolist(),
                "alive": state.alive.astype(int).tolist(),
                "land": state.land.astype(int).tolist(),
                "armyTotal": state.army_total.astype(int).tolist(),
                "config": state.config,
            },
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        if "actions" in data:
            return [int(a) for a in data["actions"]]
        return normalise(data.get("action"), state)


# --------------------------------------------------------------------------- #
# loading agents by name
# --------------------------------------------------------------------------- #
def load_agent(spec: str) -> Agent:
    """Resolve an agent from a command-line spec.

    ``expander``            a bot from `genmod.bots`
    ``mymodule:act``        a callable in an importable module (or a .py path)
    ``path/to/file.py:act``  the same, by file path
    ``http://host:port/``   a remote `genmod-bot/1` endpoint
    """
    if re.match(r"^https?://", spec):
        return HttpAgent(spec)

    from genmod.bots import BOTS

    if spec in BOTS:
        return _jax_bot(BOTS[spec])

    if ":" not in spec:
        # A bare path is the common case: find the entry point the same way the
        # local server's file picker does, so `bot.py` and `bot.py:act` both work.
        import os

        if spec.endswith(".py") or os.path.exists(spec):
            from server.agents_registry import load_from_source

            path = pathlib.Path(spec)  # module-level pathlib; do not shadow it here
            if not path.exists():
                raise ValueError(f"no such file: {spec}")
            return load_from_source(path.read_text(), path.name).fn
        raise ValueError(
            f"unknown agent {spec!r}. Use a builtin ({', '.join(sorted(BOTS))}), "
            "a path like 'bot.py' or 'bot.py:act', 'module:function', or an http(s) URL."
        )

    target, _, func = spec.rpartition(":")
    if target.endswith(".py"):
        import importlib.util

        path = pathlib.Path(target).resolve()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot import {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        import importlib

        module = importlib.import_module(target)
    fn = getattr(module, func)
    return fn


def _jax_bot(bot_fn) -> Agent:
    """Adapt a `genmod.bots` policy (which wants a JAX State) to the agent contract."""
    import jax

    counter = {"k": 0}

    def agent(state: BotState, player: int):
        cfg = state.env_config()
        st = state.to_genmod_state(cfg)
        counter["k"] += 1
        key = jax.random.PRNGKey(counter["k"])
        return int(bot_fn(key, st, cfg, player))

    return agent
