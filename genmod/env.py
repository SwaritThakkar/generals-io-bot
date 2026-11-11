"""Core dynamics of the modified (perfect-information) Generals.io game.

Rules follow Straka, Lisy & Schmid (arXiv:2606.23348), Section IV, with a single
modification: **there is no fog of war** -- both players observe the full board
(`EnvConfig.visibility="full"`).  The original fogged observation is still
available (`"fog"`) so the two regimes can be compared under identical dynamics.

Everything here is a pure function of a flat PyTree, so `jit`, `vmap` and `scan`
all compose freely:

    step_fn = jax.jit(jax.vmap(partial(step, cfg=cfg)))
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import mapgen
from .config import DIRS, NUM_CELL_ACTIONS, EnvConfig
from .state import State, army_counts, land_counts, player_army_map

_DR = jnp.array([d[0] for d in DIRS], jnp.int32)
_DC = jnp.array([d[1] for d in DIRS], jnp.int32)


class StepInfo(NamedTuple):
    land: jax.Array   # (N,) cells owned
    armies: jax.Array  # (N,) total army
    captured: jax.Array  # (N,) bool, whose general fell this step


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _dilate3(mask: jax.Array) -> jax.Array:
    """3x3 (Moore) binary dilation over the trailing two axes."""
    p = jnp.pad(mask, [(0, 0)] * (mask.ndim - 2) + [(1, 1), (1, 1)])
    out = mask
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            out = out | p[..., dr : dr + mask.shape[-2], dc : dc + mask.shape[-1]]
    return out


def _neighbour_passable(passable: jax.Array) -> jax.Array:
    """(4, H, W): is the neighbour in direction d in-bounds and passable?"""
    h, w = passable.shape
    p = jnp.pad(passable, 1, constant_values=False)
    return jnp.stack(
        [
            p[0:h, 1 : w + 1],        # up
            p[2 : h + 2, 1 : w + 1],  # down
            p[1 : h + 1, 0:w],        # left
            p[1 : h + 1, 2 : w + 2],  # right
        ]
    )


def _tree_where(pred, a, b):
    return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)


# --------------------------------------------------------------------------- #
# reset
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("cfg",))
def reset(key: jax.Array, cfg: EnvConfig) -> State:
    """Sample a fresh game."""
    key, sub = jax.random.split(key)
    mountain, castle, general, general_idx, playable, army, owner = mapgen.generate(sub, cfg)

    n, h, w = cfg.num_players, cfg.height, cfg.width
    pid = jnp.arange(n, dtype=jnp.int8)[:, None, None]
    pam = jnp.where(owner[None] == pid, army[None], 0).astype(jnp.int32)

    state = State(
        army=army,
        owner=owner,
        mountain=mountain,
        castle=castle,
        general=general,
        playable=playable,
        general_idx=general_idx,
        alive=jnp.ones((n,), bool),
        step=jnp.int32(0),
        done=jnp.bool_(False),
        winner=jnp.int8(-1),
        seen=jnp.zeros((n, h, w), bool),
        prev_army_p=pam,
        delta_hist=jnp.zeros((cfg.history_length, n, h, w), jnp.int16),
        score_hist=jnp.zeros((cfg.series_length, n, 2), jnp.int32),
        key=key,
    )
    if cfg.series_length > 0:
        totals = jnp.stack([land_counts(state, n), army_counts(state, n)], -1)
        state = state._replace(score_hist=state.score_hist.at[0].set(totals))
    return _refresh_seen(state, cfg)


# --------------------------------------------------------------------------- #
# visibility
# --------------------------------------------------------------------------- #
def visibility(state: State, cfg: EnvConfig) -> jax.Array:
    """(N, H, W) bool: what each player can see *this* step."""
    n = cfg.num_players
    if cfg.visibility == "full":
        return jnp.ones((n, cfg.height, cfg.width), bool)

    pid = jnp.arange(n, dtype=jnp.int8)[:, None, None]
    own = state.owner[None] == pid
    vis = _dilate3(own)
    if cfg.visibility == "opponent":
        # The mod, restricted: enemy territory (and its halo) is always exposed,
        # but neutral terrain still has to be explored.
        enemy = (state.owner[None] >= 0) & ~own
        vis = vis | _dilate3(enemy)
    return vis


def _refresh_seen(state: State, cfg: EnvConfig) -> State:
    if cfg.visibility == "full":
        return state
    return state._replace(seen=state.seen | visibility(state, cfg))


# --------------------------------------------------------------------------- #
# action space
# --------------------------------------------------------------------------- #
def decode_action(action: jax.Array, cfg: EnvConfig):
    """Flat action index -> (cell, is_pass, direction, send_half)."""
    a = jnp.clip(action, 0, cfg.num_actions - 1).astype(jnp.int32)
    cell, k = a // NUM_CELL_ACTIONS, a % NUM_CELL_ACTIONS
    m = k - 1
    return cell, k == 0, m % 4, m >= 4


def encode_action(row, col, direction, send_half, cfg: EnvConfig):
    """(row, col, dir, half) -> flat action index."""
    cell = row * cfg.width + col
    return cell * NUM_CELL_ACTIONS + 1 + direction + 4 * jnp.asarray(send_half, jnp.int32)


def pass_action(cfg: EnvConfig) -> jax.Array:
    return jnp.int32(0)


def legal_action_mask(state: State, cfg: EnvConfig) -> jax.Array:
    """(N, H, W, 9) bool. Channel 0 (pass) is always legal."""
    n, h, w = cfg.num_players, cfg.height, cfg.width
    pid = jnp.arange(n, dtype=jnp.int8)[:, None, None]
    src_ok = (state.owner[None] == pid) & (state.army[None] >= 2)
    src_ok = src_ok & state.alive[:, None, None] & ~state.done

    nb = _neighbour_passable(~state.mountain)          # (4, H, W)
    moves = src_ok[:, None] & nb[None]                 # (N, 4, H, W)
    moves = jnp.moveaxis(moves, 1, -1)                 # (N, H, W, 4)

    mask = jnp.concatenate(
        [jnp.ones((n, h, w, 1), bool), moves, moves], axis=-1
    )
    # A cell that cannot act at all keeps only its own pass entry; that is fine
    # because the policy samples a single (cell, action) pair per turn.
    return mask


# --------------------------------------------------------------------------- #
# step
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("cfg",))
def step(state: State, actions: jax.Array, cfg: EnvConfig):
    """Advance one turn. `actions` is (N,) int32 of flat action indices.

    Returns ``(next_state, rewards, done, info)``.  All players move
    simultaneously; see `docs` in the README for the resolution order.
    """
    n, h, w = cfg.num_players, cfg.height, cfg.width
    hw = h * w
    idx = jnp.arange(n)

    army_f = state.army.reshape(-1)
    owner_f = state.owner.reshape(-1)
    mountain_f = state.mountain.reshape(-1)

    # ---- decode & validate -------------------------------------------------
    cell, is_pass, d, half = decode_action(actions, cfg)
    src_r, src_c = cell // w, cell % w
    dst_r, dst_c = src_r + _DR[d], src_c + _DC[d]
    in_bounds = (dst_r >= 0) & (dst_r < h) & (dst_c >= 0) & (dst_c < w)
    dst = jnp.clip(dst_r, 0, h - 1) * w + jnp.clip(dst_c, 0, w - 1)

    src_army = army_f[cell]
    legal = (
        ~is_pass
        & in_bounds
        & ~mountain_f[dst]
        & (owner_f[cell] == idx.astype(jnp.int8))
        & (src_army >= 2)
        & state.alive
        & ~state.done
    )
    sent = jnp.where(legal, jnp.where(half, src_army // 2, src_army - 1), 0).astype(jnp.int32)

    # ---- departures --------------------------------------------------------
    army_f = army_f.at[cell].add(-sent)

    # ---- arrivals ----------------------------------------------------------
    inc = jnp.zeros((n, hw), jnp.int32).at[idx, dst].add(sent)

    # ---- simultaneous combat ----------------------------------------------
    # Every player's force at a cell = what they sent in + what they already had.
    # The strongest force survives with (largest - second largest); a neutral
    # garrison then absorbs that remainder. Ties keep the previous owner.
    occupancy = owner_f[None] == idx.astype(jnp.int8)[:, None]      # (N, HW)
    forces = inc + jnp.where(occupancy, army_f[None], 0)
    neutral_garrison = jnp.where(owner_f < 0, army_f, 0)

    if n == 2:
        f0, f1 = forces[0], forces[1]
        strength = jnp.abs(f0 - f1)
        challenger = (f1 > f0).astype(jnp.int8)
    else:
        vals, ids = jax.lax.top_k(forces.T, 2)
        strength = vals[:, 0] - vals[:, 1]
        challenger = ids[:, 0].astype(jnp.int8)
    contested = strength > 0
    is_neutral = owner_f < 0

    after_garrison = strength - neutral_garrison
    new_owner = jnp.where(
        is_neutral,
        jnp.where(after_garrison > 0, challenger, jnp.int8(-1)),
        jnp.where(contested, challenger, owner_f),
    )
    new_army = jnp.where(is_neutral, jnp.abs(after_garrison), strength).astype(jnp.int32)

    # Mountains never change hands.
    new_owner = jnp.where(mountain_f, jnp.int8(-1), new_owner)
    new_army = jnp.where(mountain_f, 0, new_army)

    # ---- general capture ---------------------------------------------------
    gi = jnp.clip(state.general_idx, 0, hw - 1)
    gen_owner = new_owner[gi]
    captured = state.alive & (gen_owner != idx.astype(jnp.int8))
    captor = jnp.where(captured, gen_owner, idx.astype(jnp.int8))

    # The captor inherits every cell of the fallen player, at half strength.
    remap = jnp.where(captured, captor, idx.astype(jnp.int8))
    owned = new_owner >= 0
    owner_slot = jnp.clip(new_owner, 0, n - 1)
    loses_cells = owned & captured[owner_slot]
    new_army = jnp.where(loses_cells, (new_army + 1) // 2, new_army)
    new_owner = jnp.where(owned, remap[owner_slot], new_owner)

    fallen = jnp.zeros((hw,), bool).at[gi].set(captured)
    new_general = state.general.reshape(-1) & ~fallen
    new_castle = state.castle.reshape(-1) | fallen  # a fallen general becomes a castle
    alive = state.alive & ~captured
    general_idx = jnp.where(captured, -1, state.general_idx)

    # ---- production --------------------------------------------------------
    t = state.step + 1
    grows_fast = (t % cfg.general_growth_interval) == 0
    grows_land = (t % cfg.land_growth_interval) == 0
    producers = new_general | new_castle
    owned = new_owner >= 0
    new_army = new_army + (grows_fast & owned & producers) + (grows_land & owned)

    # ---- scoreboard --------------------------------------------------------
    holds = new_owner[None] == jnp.arange(n, dtype=jnp.int8)[:, None]   # (N, HW)
    land = holds.sum(-1).astype(jnp.int32)
    total_army = jnp.where(holds, new_army[None], 0).sum(-1).astype(jnp.int32)

    # ---- termination -------------------------------------------------------
    n_alive = alive.sum()
    timeout = t >= cfg.max_steps
    done = (n_alive <= 1) | timeout

    last_standing = jnp.argmax(alive).astype(jnp.int8)
    winner = jnp.where(n_alive == 1, last_standing, jnp.int8(-1))
    if cfg.timeout_result == "land":
        # Lexicographic (land, army) tiebreak; a perfect tie stays a draw.
        best_land = jnp.where(alive, land, -1).max()
        top = alive & (land == best_land)
        best_army = jnp.where(top, total_army, -1).max()
        top = top & (total_army == best_army)
        winner = jnp.where(
            (n_alive > 1) & timeout & (top.sum() == 1),
            jnp.argmax(top).astype(jnp.int8),
            winner,
        )

    nxt = State(
        army=new_army.reshape(h, w),
        owner=new_owner.reshape(h, w),
        mountain=state.mountain,
        castle=new_castle.reshape(h, w),
        general=new_general.reshape(h, w),
        playable=state.playable,
        general_idx=general_idx,
        alive=alive,
        step=t,
        done=done,
        winner=winner,
        seen=state.seen,
        prev_army_p=state.prev_army_p,
        delta_hist=state.delta_hist,
        score_hist=state.score_hist,
        key=state.key,
    )

    # ---- observation memory ------------------------------------------------
    pam = player_army_map(nxt, n)
    if cfg.history_length > 0:
        delta = jnp.clip(pam - state.prev_army_p, -32768, 32767).astype(jnp.int16)
        nxt = nxt._replace(delta_hist=jnp.concatenate([delta[None], nxt.delta_hist[:-1]], 0))
    nxt = nxt._replace(prev_army_p=pam)
    if cfg.series_length > 0:
        totals = jnp.stack([land, total_army], -1).astype(jnp.int32)
        nxt = nxt._replace(
            score_hist=jnp.concatenate([totals[None], nxt.score_hist[:-1]], 0)
        )
    nxt = _refresh_seen(nxt, cfg)

    # A finished game is frozen: further steps are no-ops with zero reward.
    # (`key` is excluded so PRNG keys of either flavour survive untouched.)
    nxt = _tree_where(state.done, state._replace(key=nxt.key), nxt)

    just_finished = nxt.done & ~state.done
    won = jnp.arange(n, dtype=jnp.int8) == nxt.winner
    rewards = jnp.where(
        just_finished & (nxt.winner >= 0), jnp.where(won, 1.0, -1.0), 0.0
    ).astype(jnp.float32)

    info = StepInfo(
        land=land_counts(nxt, n), armies=army_counts(nxt, n), captured=captured & ~state.done
    )

    return nxt, rewards, nxt.done, info


@partial(jax.jit, static_argnames=("cfg",))
def step_auto_reset(state: State, actions: jax.Array, cfg: EnvConfig):
    """Like `step`, but immediately restarts a finished episode.

    The returned reward/info describe the *terminal* transition; the returned
    state is the first state of the new episode.
    """
    nxt, rewards, done, info = step(state, actions, cfg)
    key, sub = jax.random.split(nxt.key)
    fresh = reset(sub, cfg)._replace(key=key)
    # Swap in the fresh episode wherever the old one ended; the PRNG key always
    # advances, so it is carried across the select rather than selected on.
    out = _tree_where(done, fresh._replace(key=key), nxt._replace(key=key))
    return out, rewards, done, info
