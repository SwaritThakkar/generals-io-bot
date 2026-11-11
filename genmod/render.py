"""Recording and visualisation.

`record` replays a game inside one compiled `lax.scan` and returns plain numpy
arrays; `export_html` turns those into a self-contained, scrubbable replay page
(no network access, no dependencies) and `render_ansi` prints a board to a
terminal for quick debugging.
"""

from __future__ import annotations

import base64
import json
import pathlib
from functools import partial
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .config import DIR_NAMES, NUM_CELL_ACTIONS, EnvConfig
from .env import reset, step
from .state import State, army_counts, land_counts

_TEMPLATE = pathlib.Path(__file__).with_name("_replay_template.html")

PLAYER_NAMES = ("Red", "Blue", "Green", "Violet", "Amber", "Teal", "Rose", "Lime")


class Replay(NamedTuple):
    owner: np.ndarray    # (T, H, W) int8
    army: np.ndarray     # (T, H, W) int32
    castle: np.ndarray   # (T, H, W) bool
    general: np.ndarray  # (T, H, W) bool
    mountain: np.ndarray  # (H, W) bool
    moves: np.ndarray    # (T, N) int32 flat action indices taken *from* frame t
    land: np.ndarray     # (T, N) int32
    armies: np.ndarray   # (T, N) int32
    winner: int
    turns: int           # number of frames
    cfg: EnvConfig


def record(
    cfg: EnvConfig,
    key: jax.Array,
    policies: Sequence[Callable],
    num_steps: int | None = None,
) -> Replay:
    """Play one full game and return every frame as numpy arrays."""
    num_steps = num_steps or cfg.max_steps
    n = cfg.num_players

    def body(carry, _):
        state, k = carry
        k, *ks = jax.random.split(k, n + 1)
        actions = jnp.stack(
            [p(ks[i], state, cfg, i) for i, p in enumerate(policies)]
        ).astype(jnp.int32)
        nxt, _r, _d, _i = step(state, actions, cfg)
        frame = (
            state.owner, state.army, state.castle, state.general,
            actions, land_counts(state, n), army_counts(state, n), state.done,
        )
        return (nxt, k), frame

    @partial(jax.jit, static_argnames=())
    def run(key):
        k0, k1 = jax.random.split(key)
        s0 = reset(k0, cfg)
        (final, _), frames = jax.lax.scan(body, (s0, k1), None, length=num_steps)
        return final, frames

    final, frames = run(key)
    owner, army, castle, general, moves, land, armies, done = jax.device_get(frames)

    # Trim to the first finished frame (keep it so the final board is visible).
    end = int(np.argmax(done)) if done.any() else len(done) - 1
    t = end + 1
    tail = jax.device_get(final)
    owner = np.concatenate([owner[:t], tail.owner[None]])
    army = np.concatenate([army[:t], tail.army[None]])
    castle = np.concatenate([castle[:t], tail.castle[None]])
    general = np.concatenate([general[:t], tail.general[None]])
    moves = np.concatenate([moves[:t], np.zeros((1, n), np.int32)])
    land = np.concatenate([land[:t], np.asarray(land_counts(tail, n))[None]])
    armies = np.concatenate([armies[:t], np.asarray(army_counts(tail, n))[None]])

    return Replay(
        owner=np.asarray(owner, np.int8),
        army=np.asarray(army, np.int32),
        castle=np.asarray(castle, bool),
        general=np.asarray(general, bool),
        mountain=np.asarray(jax.device_get(final.mountain), bool),
        moves=np.asarray(moves, np.int32),
        land=np.asarray(land, np.int32),
        armies=np.asarray(armies, np.int32),
        winner=int(tail.winner),
        turns=len(owner),
        cfg=cfg,
    )


# --------------------------------------------------------------------------- #
# terminal
# --------------------------------------------------------------------------- #
_ANSI = ("\x1b[41m", "\x1b[44m", "\x1b[42m", "\x1b[45m", "\x1b[43m", "\x1b[46m")


def render_ansi(state: State, cfg: EnvConfig) -> str:
    """A compact coloured board for the terminal."""
    owner = np.asarray(state.owner)
    army = np.asarray(state.army)
    mtn = np.asarray(state.mountain)
    cas = np.asarray(state.castle)
    gen = np.asarray(state.general)
    lines = []
    for r in range(cfg.height):
        row = []
        for c in range(cfg.width):
            if mtn[r, c]:
                row.append("\x1b[100m ## \x1b[0m")
                continue
            mark = "^" if gen[r, c] else ("o" if cas[r, c] else " ")
            txt = f"{mark}{army[r, c]:>3}" if army[r, c] else f"{mark}  ."
            col = _ANSI[owner[r, c] % len(_ANSI)] if owner[r, c] >= 0 else "\x1b[47m\x1b[30m"
            row.append(f"{col}{txt}\x1b[0m")
        lines.append("".join(row))
    n = cfg.num_players
    head = f"turn {int(state.step)}  " + "  ".join(
        f"{PLAYER_NAMES[i]}: {int(land_counts(state, n)[i])}L/{int(army_counts(state, n)[i])}A"
        for i in range(n)
    )
    return head + "\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# html replay
# --------------------------------------------------------------------------- #
def _b64(arr: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype).tobytes()).decode()


def export_html(
    replay: Replay,
    path: str | pathlib.Path,
    title: str = "Fogless Generals Replay",
    subtitle: str = "",
    player_names: Sequence[str] | None = None,
    standalone: bool = True,
) -> pathlib.Path:
    """Write a self-contained replay viewer.

    `standalone=False` emits a body fragment (no `<html>`/`<head>`/`<body>`),
    which is what the Artifact publisher expects.
    """
    cfg = replay.cfg
    h, w, n = cfg.height, cfg.width, cfg.num_players
    names = list(player_names or PLAYER_NAMES[:n])

    flags = (replay.castle.astype(np.uint8) | (replay.general.astype(np.uint8) << 1))
    # Army counts are the bulk of the payload; 16 bits is plenty for most games.
    wide = int(replay.army.max()) > 32000
    data = {
        "armyBits": 32 if wide else 16,
        "h": h, "w": w, "n": n, "turns": replay.turns,
        "names": names,
        "winner": replay.winner,
        "subtitle": subtitle,
        "mountains": int(replay.mountain.sum()),
        "castles": int(replay.castle[0].sum()),
        "cellActions": NUM_CELL_ACTIONS,
        "dirNames": list(DIR_NAMES),
        "terrain": _b64(replay.mountain, np.uint8),
        "owner": _b64(replay.owner, np.int8),
        "army": _b64(replay.army, np.int32 if wide else np.int16),
        "flags": _b64(flags, np.uint8),
        "moves": _b64(replay.moves, np.int32),
        "land": _b64(replay.land, np.int32),
        "armies": _b64(replay.armies, np.int32),
    }

    html = _TEMPLATE.read_text()
    html = html.replace("__TITLE__", title).replace("__DATA__", json.dumps(data))
    if standalone:
        html = (
            "<!doctype html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "</head>\n<body>\n" + html + "\n</body>\n</html>\n"
        )
    path = pathlib.Path(path)
    path.write_text(html)
    return path
