"""Baseline policies. Every bot is a pure `(key, state, cfg) -> action` function
and is safe to `jit`/`vmap`, so they double as opponents inside a training loop.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .config import DIRS, NUM_CELL_ACTIONS, EnvConfig
from .env import legal_action_mask
from .mapgen import BIG, bfs_multi
from .state import State

_DR = jnp.array([d[0] for d in DIRS], jnp.int32)
_DC = jnp.array([d[1] for d in DIRS], jnp.int32)


def _shift(x: jax.Array, d: int, fill) -> jax.Array:
    """Value of the neighbour in direction `d`, i.e. shifted towards this cell."""
    dr, dc = DIRS[d]
    p = jnp.pad(x, 1, constant_values=fill)
    h, w = x.shape
    return p[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]


def _neighbour_stack(x: jax.Array, fill) -> jax.Array:
    """(4, H, W) values of each cell's four neighbours."""
    return jnp.stack([_shift(x, d, fill) for d in range(4)])


def pass_bot(key, state: State, cfg: EnvConfig, player: int = 0) -> jax.Array:
    return jnp.int32(0)


@partial(jax.jit, static_argnames=("cfg", "player"))
def random_bot(key, state: State, cfg: EnvConfig, player: int = 0) -> jax.Array:
    """Uniform over legal *moves*; passes only when no move exists."""
    mask = legal_action_mask(state, cfg)[player].reshape(-1)
    moves = mask.at[jnp.arange(cfg.num_cells) * NUM_CELL_ACTIONS].set(False)
    g = jax.random.gumbel(key, moves.shape)
    pick = jnp.argmax(jnp.where(moves, g, -jnp.inf)).astype(jnp.int32)
    return jnp.where(moves.any(), pick, 0)


@partial(jax.jit, static_argnames=("cfg", "player"))
def expander_bot(key, state: State, cfg: EnvConfig, player: int = 0) -> jax.Array:
    """A decent scripted opponent.

    Priority 1: any move that immediately takes an adjacent non-owned cell,
                preferring the one that leaves the most army behind.
    Priority 2: march the largest stack down a BFS field towards the frontier.
    """
    h, w = cfg.height, cfg.width
    me = jnp.int8(player)
    passable = ~state.mountain
    own = state.owner == me
    army = state.army

    can_act = own & (army >= 2)
    nb_pass = _neighbour_stack(passable, False)          # (4, H, W)
    nb_owner = _neighbour_stack(state.owner, jnp.int8(-1))
    nb_army = _neighbour_stack(army, jnp.int32(BIG))
    nb_target = nb_pass & (nb_owner != me)

    # --- priority 1: immediate captures ------------------------------------
    surplus = army[None] - 1 - nb_army                    # units left after taking
    capture = can_act[None] & nb_target & (surplus > 0)
    cap_score = jnp.where(capture, surplus, -BIG)

    # --- priority 2: head for the nearest non-owned cell --------------------
    dist = bfs_multi(passable, passable & ~own, cfg.bfs_iters)
    nb_dist = _neighbour_stack(dist, jnp.int32(BIG))
    downhill = can_act[None] & nb_pass & (nb_dist < dist[None])
    # Bigger stacks move first; ties broken towards the closer frontier.
    march_score = jnp.where(downhill, army[None] * 8 - nb_dist, -BIG)

    best_cap = cap_score.max()
    score = jnp.where(best_cap > -BIG, cap_score, march_score)
    flat = jnp.argmax(score.reshape(-1))
    d, cell = flat // (h * w), flat % (h * w)

    action = cell * NUM_CELL_ACTIONS + 1 + d
    return jnp.where(score.reshape(-1)[flat] > -BIG, action.astype(jnp.int32), 0)


BOTS = {"pass": pass_bot, "random": random_bot, "expander": expander_bot}
