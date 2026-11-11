"""Rule-level tests. Run with `python -m pytest tests -q` (or plain `python tests/test_rules.py`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from genmod import EnvConfig, legal_action_mask, observe, reset, step
from genmod.config import NUM_CELL_ACTIONS
from genmod.state import State

CFG = EnvConfig(height=5, width=5, num_players=2, min_spawn_dist=2, max_spawn_dist=8,
                min_castles=2, max_castles=2, bfs_iters=20, max_steps=100)


def blank(cfg=CFG) -> State:
    """An empty, mountain-free board with generals at the two corners."""
    h, w, n = cfg.height, cfg.width, cfg.num_players
    owner = np.full((h, w), -1, np.int8)
    army = np.zeros((h, w), np.int32)
    general = np.zeros((h, w), bool)
    gi = np.array([0, h * w - 1], np.int32)
    for p, idx in enumerate(gi):
        owner.reshape(-1)[idx] = p
        army.reshape(-1)[idx] = 1
        general.reshape(-1)[idx] = True
    return State(
        army=jnp.asarray(army), owner=jnp.asarray(owner),
        mountain=jnp.zeros((h, w), bool), castle=jnp.zeros((h, w), bool),
        general=jnp.asarray(general), playable=jnp.ones((h, w), bool),
        general_idx=jnp.asarray(gi), alive=jnp.ones((n,), bool),
        step=jnp.int32(0), done=jnp.bool_(False), winner=jnp.int8(-1),
        seen=jnp.zeros((n, h, w), bool), prev_army_p=jnp.zeros((n, h, w), jnp.int32),
        delta_hist=jnp.zeros((cfg.history_length, n, h, w), jnp.int16),
        score_hist=jnp.zeros((cfg.series_length, n, 2), jnp.int32),
        key=jax.random.PRNGKey(0),
    )


def put(s: State, r, c, owner=-1, army=0, castle=False) -> State:
    return s._replace(
        owner=s.owner.at[r, c].set(np.int8(owner)),
        army=s.army.at[r, c].set(np.int32(army)),
        castle=s.castle.at[r, c].set(castle),
    )


def act(r, c, direction, half=False, cfg=CFG):
    return (r * cfg.width + c) * NUM_CELL_ACTIONS + 1 + direction + 4 * int(half)


PASS = jnp.int32(0)
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


# --------------------------------------------------------------------------- #
def test_move_all_but_one_into_empty():
    s = put(blank(), 2, 2, owner=0, army=10)
    s, *_ = step(s, jnp.array([act(2, 2, RIGHT), PASS]), CFG)
    assert int(s.army[2, 2]) == 1 and int(s.owner[2, 2]) == 0
    assert int(s.army[2, 3]) == 9 and int(s.owner[2, 3]) == 0


def test_move_half_rounds_down():
    s = put(blank(), 2, 2, owner=0, army=9)
    s, *_ = step(s, jnp.array([act(2, 2, RIGHT, half=True), PASS]), CFG)
    assert int(s.army[2, 3]) == 4      # floor(9/2) leaves
    assert int(s.army[2, 2]) == 5


def test_single_unit_cannot_move():
    s = put(blank(), 2, 2, owner=0, army=1)
    s, *_ = step(s, jnp.array([act(2, 2, RIGHT), PASS]), CFG)
    assert int(s.army[2, 2]) == 1 and int(s.owner[2, 3]) == -1


def test_combat_by_subtraction_and_capture():
    s = put(put(blank(), 2, 2, owner=0, army=10), 2, 3, owner=1, army=4)
    s, *_ = step(s, jnp.array([act(2, 2, RIGHT), PASS]), CFG)
    assert int(s.owner[2, 3]) == 0 and int(s.army[2, 3]) == 5   # 9 attackers - 4


def test_defender_holds_on_tie():
    s = put(put(blank(), 2, 2, owner=0, army=6), 2, 3, owner=1, army=5)
    s, *_ = step(s, jnp.array([act(2, 2, RIGHT), PASS]), CFG)
    assert int(s.owner[2, 3]) == 1 and int(s.army[2, 3]) == 0   # ties keep the owner


def test_defender_survives_weak_attack():
    s = put(put(blank(), 2, 2, owner=0, army=4), 2, 3, owner=1, army=9)
    s, *_ = step(s, jnp.array([act(2, 2, RIGHT), PASS]), CFG)
    assert int(s.owner[2, 3]) == 1 and int(s.army[2, 3]) == 6   # 9 - 3


def test_neutral_castle_needs_full_garrison():
    s = put(put(blank(), 2, 2, owner=0, army=41), 2, 3, army=40, castle=True)
    s2, *_ = step(s, jnp.array([act(2, 2, RIGHT), PASS]), CFG)
    assert int(s2.owner[2, 3]) == -1 and int(s2.army[2, 3]) == 0  # exactly drained

    s = put(put(blank(), 2, 2, owner=0, army=45), 2, 3, army=40, castle=True)
    s2, *_ = step(s, jnp.array([act(2, 2, RIGHT), PASS]), CFG)
    assert int(s2.owner[2, 3]) == 0 and int(s2.army[2, 3]) == 4   # 44 - 40


def test_head_on_collision_resolves_simultaneously():
    """Both stacks leave, both arrive; the survivor is the difference."""
    s = put(put(blank(), 2, 1, owner=0, army=10), 2, 3, owner=1, army=6)
    s, *_ = step(s, jnp.array([act(2, 1, RIGHT), act(2, 3, LEFT)]), CFG)
    assert int(s.owner[2, 2]) == 0 and int(s.army[2, 2]) == 4   # 9 vs 5
    assert int(s.army[2, 1]) == 1 and int(s.army[2, 3]) == 1    # garrisons stay


def test_mutual_swap_passes_through():
    s = put(put(blank(), 2, 2, owner=0, army=10), 2, 3, owner=1, army=8)
    s, *_ = step(s, jnp.array([act(2, 2, RIGHT), act(2, 3, LEFT)]), CFG)
    # 9 attackers hit a garrison of 1; 7 attackers hit a garrison of 1.
    assert int(s.owner[2, 3]) == 0 and int(s.army[2, 3]) == 8
    assert int(s.owner[2, 2]) == 1 and int(s.army[2, 2]) == 6


def test_general_capture_ends_the_game():
    s = blank()
    s = put(s, 0, 1, owner=1, army=9)                     # next to player 0's general
    s, r, done, info = step(s, jnp.array([PASS, act(0, 1, LEFT)]), CFG)
    assert bool(done) and int(s.winner) == 1
    assert not bool(s.alive[0]) and bool(s.castle[0, 0])  # fallen general becomes a castle
    assert r.tolist() == [-1.0, 1.0]


def test_finished_games_are_frozen():
    s = blank()
    s = put(s, 0, 1, owner=1, army=9)
    s, *_ = step(s, jnp.array([PASS, act(0, 1, LEFT)]), CFG)
    before = int(s.step)
    s2, r, done, _ = step(s, jnp.array([PASS, PASS]), CFG)
    assert int(s2.step) == before and float(np.abs(np.asarray(r)).sum()) == 0.0


def test_production_schedule():
    cfg = CFG
    s = put(blank(), 2, 2, owner=0, army=3)               # a plain owned cell
    for _ in range(2):
        s, *_ = step(s, jnp.array([PASS, PASS]), cfg)
    assert int(s.army[0, 0]) == 2                          # general ticked once at t=2
    assert int(s.army[2, 2]) == 3                          # plains do not tick yet
    for _ in range(48):
        s, *_ = step(s, jnp.array([PASS, PASS]), cfg)
    assert int(s.step) == 50
    assert int(s.army[2, 2]) == 4                          # land tick at t=50
    assert int(s.army[0, 0]) == 1 + 25 + 1                 # 25 general ticks + land tick


def test_legal_mask_matches_the_dynamics():
    cfg = EnvConfig(height=7, width=7, min_spawn_dist=3, max_spawn_dist=12, bfs_iters=28)
    s = reset(jax.random.PRNGKey(5), cfg)
    for _ in range(30):
        s, *_ = step(s, jnp.array([jax.random.randint(jax.random.PRNGKey(1), (), 0, cfg.num_actions),
                                   jnp.int32(0)]), cfg)
    mask = np.asarray(legal_action_mask(s, cfg))
    owner, army, mtn = np.asarray(s.owner), np.asarray(s.army), np.asarray(s.mountain)
    for p in (0, 1):
        for r in range(cfg.height):
            for c in range(cfg.width):
                src_ok = owner[r, c] == p and army[r, c] >= 2 and not bool(s.done)
                for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                    rr, cc = r + dr, c + dc
                    inb = 0 <= rr < cfg.height and 0 <= cc < cfg.width
                    want = src_ok and inb and not mtn[rr, cc]
                    assert bool(mask[p, r, c, 1 + d]) == want
                    assert bool(mask[p, r, c, 5 + d]) == want
                assert bool(mask[p, r, c, 0])


def test_army_is_conserved_when_nothing_is_produced():
    """Between production ticks, total army only ever shrinks (by cancellation)."""
    cfg = EnvConfig(height=9, width=9, min_spawn_dist=4, max_spawn_dist=14, bfs_iters=36,
                    general_growth_interval=10 ** 6, land_growth_interval=10 ** 6)
    key = jax.random.PRNGKey(11)
    s = reset(key, cfg)
    prev = int(s.army.sum())
    for i in range(80):
        key, k = jax.random.split(key)
        a = jax.random.randint(k, (2,), 0, cfg.num_actions, dtype=jnp.int32)
        s, *_ = step(s, a, cfg)
        now = int(s.army.sum())
        assert now <= prev
        prev = now


def test_map_generation_respects_its_constraints():
    cfg = EnvConfig(height=23, width=23, min_spawn_dist=17, bfs_iters=92)
    from genmod.mapgen import bfs
    for seed in range(12):
        s = reset(jax.random.PRNGKey(seed), cfg)
        gi = np.asarray(s.general_idx)
        assert len(set(gi.tolist())) == cfg.num_players
        assert not np.asarray(s.mountain).reshape(-1)[gi].any()
        d = np.asarray(bfs(~s.mountain, s.general_idx[0], cfg.bfs_iters)).reshape(-1)
        assert d[gi[1]] >= cfg.min_spawn_dist
        n_cas = int(s.castle.sum())
        assert cfg.min_castles <= n_cas <= cfg.max_castles
        cas_army = np.asarray(s.army)[np.asarray(s.castle)]
        assert cas_army.min() >= cfg.castle_army_min and cas_army.max() <= cfg.castle_army_max
        for p in range(cfg.num_players):
            dp = np.asarray(bfs(~s.mountain, s.general_idx[p], cfg.bfs_iters)).reshape(-1)
            near = np.asarray(s.castle).reshape(-1) & (dp <= cfg.guaranteed_castle_radius)
            assert near.any(), f"player {p} has no castle within radius"


def test_no_fog_means_identical_information():
    cfg = EnvConfig(height=11, width=11, min_spawn_dist=5, max_spawn_dist=16, bfs_iters=44)
    s = reset(jax.random.PRNGKey(2), cfg)
    vis = np.asarray(jax.jit(lambda st: __import__("genmod").visibility(st, cfg))(s))
    assert vis.all(), "full visibility must expose every cell"
    o = np.asarray(observe(s, cfg))
    assert np.isfinite(o).all()
    # Player 1's view is player 0's view with the own/enemy roles exchanged.
    swap = list(range(o.shape[1]))
    for a, b in [(0, 1), (3, 4), (7, 8), (10, 11), (21, 23), (22, 24)]:
        swap[a], swap[b] = swap[b], swap[a]
    hh = cfg.history_length
    base = 27
    for k in range(hh):
        swap[base + k], swap[base + hh + k] = swap[base + hh + k], swap[base + k]
    assert np.allclose(o[0][swap], o[1], atol=1e-6)


def test_batched_rollout_runs():
    import genmod
    from genmod import bots
    cfg = EnvConfig(height=13, width=13, min_spawn_dist=6, max_spawn_dist=18,
                    bfs_iters=52, max_steps=300)
    st = genmod.reset_batch(jax.random.PRNGKey(0), cfg, 32)
    pol = genmod.bot_policy([bots.expander_bot, bots.random_bot])
    roll = genmod.make_rollout(cfg, pol, 300)
    st, traj = roll(jax.random.PRNGKey(1), st)
    assert traj.rewards.shape == (300, 32, 2)
    assert int(traj.done.sum()) > 0
    # Zero-sum: every terminal reward vector sums to zero.
    assert float(np.abs(np.asarray(traj.rewards).sum(-1)).max()) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
