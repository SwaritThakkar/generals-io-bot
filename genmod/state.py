"""Game state as a flat JAX PyTree (all fields have static shapes)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

NEUTRAL = -1


class State(NamedTuple):
    """Complete game state. All arrays have shapes fixed by `EnvConfig`.

    Batched states are the same PyTree with a leading batch axis on every leaf.
    """

    # --- board (H, W) -------------------------------------------------------
    army: jax.Array      # int32  units standing on each cell
    owner: jax.Array     # int8   -1 neutral, else player index
    mountain: jax.Array  # bool   impassable (includes the unplayable padding)
    castle: jax.Array    # bool   army-generating cell
    general: jax.Array   # bool   living generals
    playable: jax.Array  # bool   inside the active sub-rectangle

    # --- players (N,) -------------------------------------------------------
    general_idx: jax.Array  # int32 flat index of each general (-1 once dead)
    alive: jax.Array        # bool

    # --- bookkeeping --------------------------------------------------------
    step: jax.Array    # int32 scalar, number of resolved turns
    done: jax.Array    # bool  scalar
    winner: jax.Array  # int8  scalar, -1 = none/draw

    # --- observation memory -------------------------------------------------
    seen: jax.Array        # bool  (N, H, W) persistent visibility (fog modes)
    prev_army_p: jax.Array   # int32 (N, H, W) per-player army map, last step
    delta_hist: jax.Array    # int16 (Hh, N, H, W) newest-first army deltas
    score_hist: jax.Array    # int32 (S, N, 2) newest-first (land, army) totals

    key: jax.Array  # PRNG key carried for auto-reset convenience


def player_army_map(state: State, num_players: int) -> jax.Array:
    """(N, H, W) army counts split by owner."""
    pid = jnp.arange(num_players, dtype=jnp.int8)[:, None, None]
    return jnp.where(state.owner[None] == pid, state.army[None], 0)


def land_counts(state: State, num_players: int) -> jax.Array:
    """(N,) number of cells owned by each player."""
    pid = jnp.arange(num_players, dtype=jnp.int8)[:, None, None]
    return (state.owner[None] == pid).sum(axis=(1, 2)).astype(jnp.int32)


def army_counts(state: State, num_players: int) -> jax.Array:
    """(N,) total army of each player."""
    return player_army_map(state, num_players).sum(axis=(1, 2)).astype(jnp.int32)
