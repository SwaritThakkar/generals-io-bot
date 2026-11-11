#!/usr/bin/env python3
"""Script-vs-script games inside sim36 on the harness's maps (fast iteration).

Maps come from `genmod.reset(PRNGKey(seed))`, the same as `Match`.
"""
from __future__ import annotations

import pathlib
import sys
import time

import jax
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from genmod import EnvConfig, reset  # noqa: E402
import core36  # noqa: E402
from sim36 import Board, Sim  # noqa: E402

_MAPS = {}


def start(seed, max_steps):
    if seed not in _MAPS:
        cfg = EnvConfig(height=23, width=23, max_steps=max_steps, bfs_iters=92)
        st = reset(jax.random.PRNGKey(seed), cfg)
        _MAPS[seed] = (np.asarray(st.mountain), np.asarray(st.castle),
                       np.asarray(st.owner), np.asarray(st.army),
                       list(np.asarray(st.general_idx)))
    m, c, o, a, gi = _MAPS[seed]
    board = Board(m, c, 23, 23)
    return core36.Ctx(board), Sim(board, o, a, gi, 0, max_steps)


def play(seed, f0, f1, max_steps=500, verbose=0):
    ctx, sim = start(seed, max_steps)
    s0, s1 = f0(ctx, 0), f1(ctx, 1)
    while not sim.done:
        a0, a1 = s0.act(sim), s1.act(sim)
        sim.step2(a0, a1)
        if verbose and sim.step % verbose == 0:
            land, tot = sim.totals()
            print(sim.step, land, tot, [sim.army[g] for g in sim.gen])
    return sim.winner, sim.step


if __name__ == "__main__":
    E = lambda c, p: core36.Econ(c, p)
    R = lambda c, p: core36.Rusher(c, p)
    CR = lambda c, p: core36.CastleRusher(c, p)
    for name, opp in (("rusher", R), ("castle", CR), ("econ", E)):
        res = []
        t0 = time.perf_counter()
        for seed in (92020, 92021, 92022, 92023, 92024, 92025):
            for seat in (0, 1):
                f = (E, opp) if seat == 0 else (opp, E)
                w, t = play(seed, *f)
                res.append("W" if w == seat else "D" if w < 0 else "L")
                res[-1] += str(t)
        print(name, " ".join(res), f"{time.perf_counter()-t0:.1f}s")
