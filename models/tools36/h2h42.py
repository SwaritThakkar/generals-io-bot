#!/usr/bin/env python3
"""Parallel sim36 matches on the harness maps (fast iteration, CPU cores).

    python models/tools36/h2h42.py CAND OPP [--seeds 93001 93010] [--cap 500]

CAND / OPP: a bot file exposing Planner (e.g. models/v42_collect_race.py),
or one of: gather60 gather120 gather200 spread spread_late expand rusher
castle_rusher econ. Reports W/D/L per seat, castles held at the end, and
mean / max decision time of CAND.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import multiprocessing as mp
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def setup():
    sys.path[:0] = [str(ROOT), str(ROOT / "models"), str(ROOT / "models/tools36")]


def factory(spec):
    import adversaries38
    import adversaries42
    import adversaries53
    import core36
    import core37
    named = {
        "gather60": lambda c, p: adversaries38.GatherRusher(c, p, t0=60),
        "gather120": lambda c, p: adversaries38.GatherRusher(c, p, t0=120),
        "gather200": lambda c, p: adversaries38.GatherRusher(c, p, t0=200),
        "spread": lambda c, p: adversaries42.SpreadCollector(c, p),
        "spread_late": lambda c, p: adversaries42.SpreadCollector(c, p, t_castle=150, t_collect=450),
        "expand": lambda c, p: core37.Expand(c, p),
        "rusher": lambda c, p: core36.Rusher(c, p),
        "castle_rusher": lambda c, p: core36.CastleRusher(c, p),
        "econ": lambda c, p: core36.Econ(c, p),
        "siege": lambda c, p: adversaries53.Besieger(c, p),
        "siege2": lambda c, p: adversaries53.Besieger(c, p, t0=150, post=2, posts=2),
    }
    if spec in named:
        return named[spec]
    path = ROOT / spec
    mod_spec = importlib.util.spec_from_file_location(path.stem + "_h2h", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return lambda c, p: mod.Planner(c, p)


def job(args):
    cand, opp, seed, seat, cap = args
    setup()
    from arena36 import start
    ctx, sim = start(seed, cap)
    fc, fo = factory(cand), factory(opp)
    bots = [None, None]
    bots[seat], bots[1 - seat] = fc(ctx, seat), fo(ctx, 1 - seat)
    times = []
    while not sim.done:
        t = time.perf_counter()
        a = bots[seat].act(sim)
        times.append(time.perf_counter() - t)
        b = bots[1 - seat].act(sim)
        a = a if a and a > 0 else 0
        b = b if b and b > 0 else 0
        sim.step2(*((a, b) if seat == 0 else (b, a)))
    land, tot = sim.totals()
    cas = [sum(1 for c in ctx.castles if sim.owner[c] == q) for q in (seat, 1 - seat)]
    return (seed, seat, sim.winner, sim.step, land[seat], land[1 - seat], tot[seat], tot[1 - seat],
            cas, 1000 * sum(times) / len(times), 1000 * max(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("opp", nargs="+")
    ap.add_argument("--seeds", type=int, nargs=2, default=[93001, 93010])
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    seeds = range(a.seeds[0], a.seeds[1] + 1)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for opp in a.opp:
            t0 = time.perf_counter()
            jobs = [(a.cand, opp, s, seat, a.cap) for s in seeds for seat in (0, 1)]
            res = pool.map(job, jobs)
            seatres = collections.Counter()
            cas, ms, mx = [0, 0], [], 0
            for seed, seat, w, step, *_rest, c, m, x in res:
                r = "W" if w == seat else "D" if w < 0 else "L"
                seatres[(seat, r)] += 1
                cas[0] += c[0]; cas[1] += c[1]; ms.append(m); mx = max(mx, x)
            W = sum(v for (s, r), v in seatres.items() if r == "W")
            D = sum(v for (s, r), v in seatres.items() if r == "D")
            L = sum(v for (s, r), v in seatres.items() if r == "L")
            n = len(res)
            print(f"{opp:28s} W{W:3d} D{D:3d} L{L:3d}  seat0 {seatres[(0,'W')]}-{seatres[(0,'L')]} "
                  f"seat1 {seatres[(1,'W')]}-{seatres[(1,'L')]}  castles/game {cas[0]/n:.1f} vs {cas[1]/n:.1f}  "
                  f"ms mean {sum(ms)/n:.1f} max {mx:.0f}  {time.perf_counter()-t0:.0f}s", flush=True)
            if a.v:
                for r in res:
                    print("   ", r[:9], [round(r[9], 1)])


if __name__ == "__main__":
    main()
