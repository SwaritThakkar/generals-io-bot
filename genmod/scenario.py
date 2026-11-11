"""Build a `State` from an explicit board, and dump one back out.

Useful for loading a specific position, for cross-implementation tests, and for
handing a live game to a Python agent over the bot protocol.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .config import EnvConfig
from .state import State, army_counts, land_counts


def make_state(
    cfg: EnvConfig,
    owner: np.ndarray,
    army: np.ndarray,
    mountain: np.ndarray | None = None,
    castle: np.ndarray | None = None,
    general: np.ndarray | None = None,
    tick: int = 0,
    key: jax.Array | None = None,
) -> State:
    """Assemble a `State` from plain (H, W) numpy arrays."""
    h, w, n = cfg.height, cfg.width, cfg.num_players
    owner = np.asarray(owner, np.int8).reshape(h, w)
    army = np.asarray(army, np.int32).reshape(h, w)
    mountain = np.zeros((h, w), bool) if mountain is None else np.asarray(mountain, bool).reshape(h, w)
    castle = np.zeros((h, w), bool) if castle is None else np.asarray(castle, bool).reshape(h, w)
    general = np.zeros((h, w), bool) if general is None else np.asarray(general, bool).reshape(h, w)

    gi = np.full((n,), -1, np.int32)
    flat_owner, flat_gen = owner.reshape(-1), general.reshape(-1)
    for p in range(n):
        hits = np.flatnonzero(flat_gen & (flat_owner == p))
        if len(hits):
            gi[p] = hits[0]
    alive = gi >= 0

    pam = np.stack([np.where(owner == p, army, 0) for p in range(n)]).astype(np.int32)
    return State(
        army=jnp.asarray(army),
        owner=jnp.asarray(owner),
        mountain=jnp.asarray(mountain),
        castle=jnp.asarray(castle),
        general=jnp.asarray(general),
        playable=jnp.ones((h, w), bool),
        general_idx=jnp.asarray(gi),
        alive=jnp.asarray(alive),
        step=jnp.int32(tick),
        done=jnp.bool_(False),
        winner=jnp.int8(-1),
        seen=jnp.zeros((n, h, w), bool),
        prev_army_p=jnp.asarray(pam),
        delta_hist=jnp.zeros((cfg.history_length, n, h, w), jnp.int16),
        score_hist=jnp.zeros((cfg.series_length, n, 2), jnp.int32),
        key=jax.random.PRNGKey(0) if key is None else key,
    )


def state_to_dict(state: State, cfg: EnvConfig) -> dict:
    """A JSON-ready snapshot, using the same flat row-major layout as the web engine."""
    n = cfg.num_players
    return {
        "h": cfg.height,
        "w": cfg.width,
        "n": n,
        "tick": int(state.step),
        "done": bool(state.done),
        "winner": int(state.winner),
        "owner": np.asarray(state.owner).reshape(-1).astype(int).tolist(),
        "army": np.asarray(state.army).reshape(-1).astype(int).tolist(),
        "mountain": np.asarray(state.mountain).reshape(-1).astype(int).tolist(),
        "castle": np.asarray(state.castle).reshape(-1).astype(int).tolist(),
        "general": np.asarray(state.general).reshape(-1).astype(int).tolist(),
        "generalIdx": np.asarray(state.general_idx).astype(int).tolist(),
        "alive": np.asarray(state.alive).astype(int).tolist(),
        "land": np.asarray(land_counts(state, n)).astype(int).tolist(),
        "armyTotal": np.asarray(army_counts(state, n)).astype(int).tolist(),
    }
