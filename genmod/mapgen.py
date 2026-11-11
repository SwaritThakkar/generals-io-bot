"""JIT-friendly random map generation.

Every routine has static shapes so the whole generator can be `vmap`ped over a
batch of PRNG keys and fused into one XLA graph together with `reset`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import EnvConfig

BIG = jnp.int32(1 << 20)


def _shift_min(dist: jax.Array) -> jax.Array:
    """Min over the four 4-neighbours, with a BIG border."""
    p = jnp.pad(dist, 1, constant_values=BIG)
    up, down = p[:-2, 1:-1], p[2:, 1:-1]
    left, right = p[1:-1, :-2], p[1:-1, 2:]
    return jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right))


def bfs(passable: jax.Array, src_flat: jax.Array, iters: int) -> jax.Array:
    """4-connected BFS distances from `src_flat` over `passable` cells.

    Unreachable (or farther than `iters`) cells hold BIG.
    """
    h, w = passable.shape
    dist = jnp.full((h * w,), BIG, jnp.int32).at[src_flat].set(0).reshape(h, w)
    dist = jnp.where(passable, dist, BIG)

    def body(_, d):
        d = jnp.minimum(d, _shift_min(d) + 1)
        return jnp.where(passable, d, BIG)

    return jax.lax.fori_loop(0, iters, body, dist)


def bfs_multi(passable: jax.Array, sources: jax.Array, iters: int) -> jax.Array:
    """BFS distance to the nearest True cell of `sources`, over `passable`."""
    dist = jnp.where(sources & passable, jnp.int32(0), BIG)

    def body(_, d):
        d = jnp.minimum(d, _shift_min(d) + 1)
        return jnp.where(passable, d, BIG)

    return jax.lax.fori_loop(0, iters, body, dist)


def _sample_masked(key: jax.Array, mask: jax.Array, fallback: jax.Array) -> jax.Array:
    """Uniformly sample one True index of `mask`; use `fallback` if `mask` is empty.

    Both masks are flat. Gumbel-argmax gives a uniform draw in one fused pass.
    """
    g = jax.random.gumbel(key, mask.shape)
    eff = jnp.where(mask.any(), mask, fallback)
    return jnp.argmax(jnp.where(eff, g, -jnp.inf)).astype(jnp.int32)


def _topk_masked(key: jax.Array, mask: jax.Array, k: int) -> jax.Array:
    """Sample `k` distinct indices (gumbel top-k) restricted to `mask`."""
    g = jax.random.gumbel(key, mask.shape)
    return jax.lax.top_k(jnp.where(mask, g, -jnp.inf), k)[1].astype(jnp.int32)


def generate(key: jax.Array, cfg: EnvConfig):
    """Return the static map layout plus initial ownership.

    Returns ``(mountain, castle, general_map, general_idx, playable, army, owner)``.
    """
    h, w, n = cfg.height, cfg.width, cfg.num_players
    hw = h * w
    k_dim, k_mtn, k_g0, k_g1, k_cnt, k_gc, k_ex, k_army = jax.random.split(key, 8)

    # --- active sub-rectangle ----------------------------------------------
    if cfg.random_dims:
        ah = jax.random.randint(k_dim, (), cfg.min_height, h + 1)
        aw = jax.random.randint(jax.random.fold_in(k_dim, 1), (), cfg.min_width, w + 1)
    else:
        ah, aw = jnp.int32(h), jnp.int32(w)
    rows = jnp.arange(h)[:, None]
    cols = jnp.arange(w)[None, :]
    playable = (rows < ah) & (cols < aw)

    # --- mountains ----------------------------------------------------------
    mountain = (jax.random.uniform(k_mtn, (h, w)) < cfg.mountain_density) | (~playable)
    passable = ~mountain
    passable_f = passable.reshape(-1)
    any_cell = jnp.zeros((hw,), bool).at[0].set(True)

    # --- generals -----------------------------------------------------------
    g0 = _sample_masked(k_g0, passable_f, any_cell)
    dist0 = bfs(passable, g0, cfg.bfs_iters)

    general_idx = [g0]
    dists = [dist0]
    for i in range(1, n):
        d_prev = dists[-1].reshape(-1)
        band = passable_f & (d_prev >= cfg.min_spawn_dist) & (d_prev <= cfg.max_spawn_dist)
        # Fallback: the farthest cell still reachable from the previous general.
        reachable = passable_f & (d_prev < BIG) & (d_prev > 0)
        far = reachable & (d_prev == jnp.where(reachable, d_prev, -1).max())
        gi = _sample_masked(jax.random.fold_in(k_g1, i), band, far)
        general_idx.append(gi)
        dists.append(bfs(passable, gi, cfg.bfs_iters))

    general_idx = jnp.stack(general_idx).astype(jnp.int32)
    general_map = jnp.zeros((hw,), bool).at[general_idx].set(True)

    # --- castles ------------------------------------------------------------
    n_castles = jax.random.randint(k_cnt, (), cfg.min_castles, cfg.max_castles + 1)

    # One guaranteed neutral castle within `guaranteed_castle_radius` of each general.
    free = passable_f & ~general_map
    castle_f = jnp.zeros((hw,), bool)
    for i in range(n):
        d = dists[i].reshape(-1)
        near = free & (d >= 1) & (d <= cfg.guaranteed_castle_radius)
        ci = _sample_masked(jax.random.fold_in(k_gc, i), near, free)
        castle_f = castle_f.at[ci].set(True)

    # Remaining castles anywhere else.
    n_extra = jnp.maximum(n_castles - n, 0)
    k_max = cfg.max_castles
    cand = _topk_masked(k_ex, free & ~castle_f, k_max)
    keep = jnp.arange(k_max) < n_extra
    castle_f = castle_f | jnp.zeros((hw,), bool).at[cand].set(keep)
    castle_f = castle_f & free  # never overwrite a general

    # --- initial armies / ownership ----------------------------------------
    castle_army = jax.random.randint(k_army, (hw,), cfg.castle_army_min, cfg.castle_army_max + 1)
    army = jnp.where(castle_f, castle_army, 0)
    army = jnp.where(general_map, 1, army).astype(jnp.int32)

    owner = jnp.full((hw,), -1, jnp.int8)
    owner = owner.at[general_idx].set(jnp.arange(n, dtype=jnp.int8))

    return (
        mountain,
        castle_f.reshape(h, w),
        general_map.reshape(h, w),
        general_idx,
        playable,
        army.reshape(h, w),
        owner.reshape(h, w),
    )
