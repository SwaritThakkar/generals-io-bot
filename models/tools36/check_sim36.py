#!/usr/bin/env python3
"""Check `models/sim36.py` against the JAX reference `genmod.env.step`.

Random games use a mix of legal moves, illegal moves, passes and deliberately
colliding moves (both players into one cell, swaps across an edge) so the
simultaneous combat rules are exercised. Any board mismatch aborts.
"""
from __future__ import annotations

import pathlib
import random
import sys

import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

import jax  # noqa: E402
from genmod import EnvConfig, reset, step  # noqa: E402
from sim36 import Board, Sim  # noqa: E402


def random_action(sim, p, rng, other=None):
    owned = [c for c in range(sim.b.n) if sim.owner[c] == p and sim.army[c] >= 2]
    roll = rng.random()
    if roll < 0.05 or not owned:
        return 0
    if roll < 0.08:
        return rng.randrange(1, sim.b.n * 9)       # often illegal
    if other is not None and roll < 0.25:
        # collide: aim at a cell next to the other player's move destination
        cell, k = divmod(other, 9)
        if k:
            dst = sim.b.step_to[cell][(k - 1) % 4]
            if dst >= 0:
                for c in sim.b.nbrs[dst]:
                    if sim.owner[c] == p and sim.army[c] >= 2:
                        return c * 9 + 1 + sim.b.direction(c, dst) + rng.choice((0, 4))
                if sim.owner[dst] == p and sim.army[dst] >= 2:
                    return dst * 9 + 1 + sim.b.direction(dst, cell) + rng.choice((0, 4))
    c = rng.choice(owned)
    return c * 9 + 1 + rng.randrange(4) + rng.choice((0, 4))


def main(games=40, ticks=260, seed=0):
    rng = random.Random(seed)
    cfg = EnvConfig(height=23, width=23, max_steps=ticks, bfs_iters=92)
    checked = 0
    for g in range(games):
        st = reset(jax.random.PRNGKey(1000 + g), cfg)
        board = Board(np.asarray(st.mountain), np.asarray(st.castle), 23, 23)
        sim = Sim(board, np.asarray(st.owner), np.asarray(st.army),
                  list(np.asarray(st.general_idx)), 0, ticks)
        while not bool(st.done):
            a0 = random_action(sim, 0, rng)
            a1 = random_action(sim, 1, rng, other=a0)
            st, *_ = step(st, jnp.asarray([a0, a1], jnp.int32), cfg)
            sim.step2(a0, a1)
            checked += 1
            if sim.done:
                assert bool(st.done), "sim ended early"
                assert int(st.winner) == sim.winner, (int(st.winner), sim.winner)
                break
            o = np.asarray(st.owner).reshape(-1).tolist()
            a = np.asarray(st.army).reshape(-1).tolist()
            if o != sim.owner or a != sim.army or int(st.step) != sim.step:
                bad = [i for i in range(len(o)) if o[i] != sim.owner[i] or a[i] != sim.army[i]]
                raise SystemExit(f"mismatch game {g} tick {sim.step} cells {bad[:5]} "
                                 f"actions {a0} {a1}")
        if not sim.done:
            assert bool(st.done)
    print(f"ok: {games} games, {checked} joint transitions identical")


if __name__ == "__main__":
    main()
