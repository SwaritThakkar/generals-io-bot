"""Observation tensors for the modified game.

Under the modification (`EnvConfig.visibility="full"`) nothing is hidden, so the
fog-related bookkeeping of the original paper collapses away: the persistent
`seen`/`enemy_seen`/`structures_in_fog`/`last_enemy_army_*` channels are no
longer needed and are replaced by their exact, current-frame counterparts.
Channel meanings are *perspective-normalised*: "own" always refers to the player
the observation is built for, so one shared policy serves every seat.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .config import EnvConfig
from .env import visibility
from .state import State, army_counts, land_counts

BASE_CHANNELS = (
    "own_army",
    "enemy_army",
    "neutral_army",
    "own_cells",
    "enemy_cells",
    "neutral_cells",
    "mountains",
    "own_general",
    "enemy_general",
    "castles",
    "own_castles",
    "enemy_castles",
    "neutral_castles",
    "visible",
    "seen",
    "playable",
    "coord_x",
    "coord_y",
    "timestep",
    "phase_land_growth",
    "phase_production",
    "own_land_count",
    "own_army_count",
    "enemy_land_count",
    "enemy_army_count",
    "land_advantage",
    "army_advantage",
)


def channel_names(cfg: EnvConfig) -> tuple[str, ...]:
    hh = cfg.history_length
    return (
        BASE_CHANNELS
        + tuple(f"own_army_delta_t-{k}" for k in range(hh))
        + tuple(f"enemy_army_delta_t-{k}" for k in range(hh))
    )


def num_channels(cfg: EnvConfig) -> int:
    return len(BASE_CHANNELS) + 2 * cfg.history_length


def _log(x, scale):
    """Signed log compression -- keeps unbounded army counts in a sane range."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x.astype(jnp.float32))) / scale


@partial(jax.jit, static_argnames=("cfg",))
def observe(state: State, cfg: EnvConfig) -> jax.Array:
    """(N, C, H, W) float32 observation, one plane stack per player."""
    n, h, w = cfg.num_players, cfg.height, cfg.width
    sc = cfg.army_log_scale
    pid = jnp.arange(n, dtype=jnp.int8)[:, None, None]

    own = state.owner[None] == pid                     # (N, H, W)
    enemy = (state.owner[None] >= 0) & ~own
    neutral = (state.owner[None] < 0) & ~state.mountain[None]

    vis = visibility(state, cfg)
    if cfg.visibility == "full":
        known = jnp.ones_like(vis)
        seen = known
    else:
        known = state.seen                             # persistent map knowledge
        seen = state.seen
    army = state.army[None]

    gen = state.general[None]
    cas = state.castle[None]

    ch = [
        _log(jnp.where(own & vis, army, 0), sc),
        _log(jnp.where(enemy & vis, army, 0), sc),
        _log(jnp.where(neutral & vis, army, 0), sc),
        (own & vis).astype(jnp.float32),
        (enemy & vis).astype(jnp.float32),
        (neutral & vis).astype(jnp.float32),
        (state.mountain[None] & known).astype(jnp.float32),
        (gen & own & known).astype(jnp.float32),
        (gen & enemy & known).astype(jnp.float32),
        (cas & known).astype(jnp.float32),
        (cas & own & known).astype(jnp.float32),
        (cas & enemy & known).astype(jnp.float32),
        (cas & neutral & known).astype(jnp.float32),
        vis.astype(jnp.float32),
        seen.astype(jnp.float32),
        jnp.broadcast_to(state.playable[None], (n, h, w)).astype(jnp.float32),
    ]

    xs = jnp.broadcast_to(jnp.arange(w, dtype=jnp.float32)[None, None] / max(w - 1, 1), (n, h, w))
    ys = jnp.broadcast_to(jnp.arange(h, dtype=jnp.float32)[None, :, None] / max(h - 1, 1), (n, h, w))
    ch += [xs, ys]

    t = state.step.astype(jnp.float32)
    scalars = [
        t / cfg.max_steps,
        (state.step % cfg.land_growth_interval) / cfg.land_growth_interval,
        (state.step % cfg.general_growth_interval) / cfg.general_growth_interval,
    ]

    land = land_counts(state, n).astype(jnp.float32)          # (N,)
    tot = army_counts(state, n).astype(jnp.float32)
    own_land, own_army = land, tot
    enemy_land = land.sum() - land
    enemy_army = tot.sum() - tot

    ch += [jnp.broadcast_to(jnp.asarray(v, jnp.float32), (n, h, w)) for v in scalars]
    for v in (own_land, own_army, enemy_land, enemy_army):
        ch.append(jnp.broadcast_to(_log(v, sc)[:, None, None], (n, h, w)))
    for a, b in ((own_land, enemy_land), (own_army, enemy_army)):
        adv = ((a - b) / (a + b + 1.0))[:, None, None]
        ch.append(jnp.broadcast_to(adv, (n, h, w)))

    if cfg.history_length > 0:
        d = state.delta_hist.astype(jnp.float32)              # (Hh, N, H, W)
        own_d = jnp.moveaxis(d, 0, 1)                         # (N, Hh, H, W)
        enemy_d = jnp.moveaxis(d.sum(1)[:, None] - d, 0, 1)
        vm = vis[:, None]
        ch.append(_log(jnp.where(vm, own_d, 0.0), sc))
        ch.append(_log(jnp.where(vm, enemy_d, 0.0), sc))
        planes = [c if c.ndim == 4 else c[:, None] for c in ch]
        return jnp.concatenate(planes, axis=1)

    return jnp.stack(ch, axis=1)


@partial(jax.jit, static_argnames=("cfg",))
def scoreboard_series(state: State, cfg: EnvConfig) -> jax.Array:
    """(N, S, 2, 2) float32: per player, the (own, enemy) x (land, army) window.

    Empty when `cfg.series_length == 0`.  Mirrors the paper's two non-spatial
    tokens; the public scoreboard is exact in both the original and modified game.
    """
    n = cfg.num_players
    hist = state.score_hist.astype(jnp.float32)          # (S, N, 2)
    own = jnp.moveaxis(hist, 0, 1)                        # (N, S, 2)
    enemy = jnp.moveaxis(hist.sum(1)[:, None] - hist, 0, 1)
    return _log(jnp.stack([own, enemy], axis=2), cfg.army_log_scale)
