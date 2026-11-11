#!/usr/bin/env python3
"""Teacher labels for the v40 plan ranker (simulation on CPU cores).

Positions come from sim36 games of v37 against a pool (v37, GatherRusher
t0=60/120/200, EconLite, Expand, Rusher, CastleRusher), sampled every 15
ticks. For one random player per position the teacher builds the matrix
our-plans x opponent-replies with PGS+-style exact rollouts and solves it
by regret matching. Saved per position: board planes, plan vectors, the
equilibrium mix over plans and each plan's value against the opponent mix.

This is game simulation (Python, CPU). Network training is done separately
on the Mac GPU by train_ranker40.py.

    python models/gen_labels40.py --workers 12 --seeds 96000 96240
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import pathlib
import pickle
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "ranker40_data"


def setup():
    sys.path[:0] = [str(ROOT), str(ROOT / "models"), str(ROOT / "models/tools36")]


def worker(seeds):
    setup()
    import numpy as np
    import adversaries38
    import core36
    import core37
    import core40
    from arena36 import start
    spec = importlib.util.spec_from_file_location("v37", ROOT / "models/v37_phase_planner.py")
    v37 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v37)
    value = core37.ValueModel()
    opps = {
        "v37": lambda c, p: v37.Planner(c, p),
        "gather60": lambda c, p: adversaries38.GatherRusher(c, p, t0=60),
        "gather120": lambda c, p: adversaries38.GatherRusher(c, p, t0=120),
        "gather200": lambda c, p: adversaries38.GatherRusher(c, p, t0=200),
        "econ": lambda c, p: core40.EconLite(c, p),
        "expand": lambda c, p: core37.Expand(c, p),
        "rusher": lambda c, p: core36.Rusher(c, p),
        "castle_rusher": lambda c, p: core36.CastleRusher(c, p),
    }
    for seed in seeds:
        rng = random.Random(seed)
        out_f = OUT / f"seed{seed}.pkl"
        if out_f.exists():
            continue
        t0 = time.perf_counter()
        opp = rng.choice(list(opps))
        cap = rng.choice((500, 800))
        ctx, sim = start(seed, cap)
        seat = rng.randrange(2)
        a_s = v37.Planner(ctx, seat)
        o_s = opps[opp](ctx, 1 - seat)
        rows = []
        while not sim.done and sim.step < cap - 20:
            if sim.step >= 10 and sim.step % 15 == 0:
                p = rng.randrange(2)
                if sim.gen[p] >= 0 and sim.gen[1 - p] >= 0:
                    m0 = core40.margin(ctx, sim, p)
                    phase = core37.detect_phase(ctx, sim, p, m0)
                    cands = core37.plan_set(ctx, sim, p, phase)
                    models = core40.replies(ctx, 1 - p)
                    horizon = min(core37.HORIZON[phase], sim.max_steps - sim.step)
                    M, names = core40.plan_matrix(ctx, sim, p, phase, value, cands, models,
                                                  horizon, core37.COMMIT[phase])
                    sr, sc, v = core40.regret_matching(M)
                    rows.append({
                        "seed": seed, "tick": sim.step, "phase": phase, "opp": opp,
                        "planes": core40.planes(ctx, sim, p).astype(np.float16),
                        "plans": [n for n, _f in cands],
                        "plan_vecs": np.array([core40.plan_vec(ctx, sim, p, n, m0) for n, _f in cands],
                                              np.float32),
                        "sigma": sr.astype(np.float32),
                        "row_value": (M @ sc).astype(np.float32),
                        "matrix": M.astype(np.float32), "replies": names, "game_value": v,
                    })
            a, b = a_s.act(sim), o_s.act(sim)
            a = a if a and a > 0 else 0
            b = b if b and b > 0 else 0
            sim.step2(*((a, b) if seat == 0 else (b, a)))
        with open(out_f, "wb") as f:
            pickle.dump(rows, f)
        print(f"seed {seed} {opp:13s} positions={len(rows)} {time.perf_counter()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs=2, default=[96000, 96240])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(*a.seeds))
    chunks = [seeds[i::a.workers] for i in range(a.workers)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        pool.map(worker, chunks)


if __name__ == "__main__":
    main()
