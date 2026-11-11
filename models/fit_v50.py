#!/usr/bin/env python3
"""Fit v50's paper models -> models/v50_model.json.

    python models/fit_v50.py [--seeds 93101 93110] [--workers 12]

Data: v46 with each parameter arm plays each opponent type (core37's four
opponent scripts) on the given seeds, both seats, cap 500, in sim36 (game
simulation runs on CPU worker processes; it is not training).
Fitting runs on the Mac GPU (torch, MPS):
  * Synnaeve & Bessiere timing model: per type and event, P(event happens),
    and the mean / std of its tick (Gaussian);
  * Weber & Mateas kNN set: one event-time vector per game;
  * Tavares et al. meta-game: arm x type win-rate matrix; its Nash
    equilibrium (regret matching) and the best-response arm per type.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).with_name("v50_model.json")
TYPES = ("rusher", "castle_rusher", "expander", "hold")
ARMS = ("base", "greedy", "safe", "castles")
NEVER = 600


def job(args):
    arm, ty, seed, seat = args
    sys.path[:0] = [str(ROOT), str(ROOT / "models"), str(ROOT / "models/tools36")]
    import importlib.util
    import core37
    from arena36 import start
    spec = importlib.util.spec_from_file_location("v50_fit", ROOT / "models/v50_26modified.py")
    v50 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v50)
    ctx, sim = start(seed, 500)
    me = v50._v46.Planner46(ctx, seat, arm)
    op = core37.opponent_models(ctx, 1 - seat)[ty]()
    ev, prev = {}, None
    while not sim.done:
        prev = v50.track_events(ctx, sim, seat, ev, prev)
        a = me.act(sim)
        b = op.act(sim)
        a = a if a and a > 0 else 0
        b = b if b and b > 0 else 0
        sim.step2(*((a, b) if seat == 0 else (b, a)))
    win = 1.0 if sim.winner == seat else 0.0 if sim.winner == 1 - seat else 0.5
    return arm, ty, [ev.get(e, NEVER) for e in v50.EVENTS], win


def regret_matching(M, iters=5000):
    import torch
    dev = torch.device("mps")
    A = torch.tensor(M, dtype=torch.float32, device=dev)
    m, n = A.shape
    rr, rc = torch.zeros(m, device=dev), torch.zeros(n, device=dev)
    sr, sc = torch.zeros(m, device=dev), torch.zeros(n, device=dev)
    for _ in range(iters):
        x = rr.clamp(min=0)
        x = x / x.sum() if x.sum() > 0 else torch.full((m,), 1.0 / m, device=dev)
        y = rc.clamp(min=0)
        y = y / y.sum() if y.sum() > 0 else torch.full((n,), 1.0 / n, device=dev)
        ur, uc = A @ y, -(x @ A)
        rr += ur - x @ ur
        rc += uc - y @ uc
        sr += x
        sc += y
    return (sr / sr.sum()).cpu().tolist(), (sc / sc.sum()).cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[93101, 93110])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = [(arm, ty, s, seat) for arm in ARMS for ty in TYPES
            for s in range(a.seeds[0], a.seeds[1] + 1) for seat in (0, 1)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(job, jobs, chunksize=2)

    import torch
    assert torch.backends.mps.is_available(), "fitting must run on the Mac GPU (MPS)"
    dev = torch.device("mps")
    sys.path[:0] = [str(ROOT / "models")]
    events = ("castle", "attack", "drained", "land30", "stack40")
    timing = {}
    for ty in TYPES:
        X = torch.tensor([r[2] for r in rows if r[1] == ty], dtype=torch.float32, device=dev)
        timing[ty] = {}
        for j, e in enumerate(events):
            col = X[:, j]
            seen = col[col < NEVER]
            p = float(seen.numel()) / max(1, col.numel())
            mu = float(seen.mean()) if seen.numel() else 300.0
            sd = float(seen.std()) if seen.numel() > 1 else 60.0
            timing[ty][e] = {"p": p, "mu": mu, "sd": sd if sd == sd else 60.0}
    W = torch.zeros(len(ARMS), len(TYPES), device=dev)
    N = torch.zeros(len(ARMS), len(TYPES), device=dev)
    for arm, ty, _ev, win in rows:
        W[ARMS.index(arm), TYPES.index(ty)] += win
        N[ARMS.index(arm), TYPES.index(ty)] += 1
    M = (W / N.clamp(min=1)).cpu().tolist()
    x, y = regret_matching(M)
    br = {ty: ARMS[max(range(len(ARMS)), key=lambda i: M[i][j])] for j, ty in enumerate(TYPES)}
    model = {"never": NEVER, "timing": timing,
             "knn": [[ty, ev] for arm, ty, ev, _w in rows if arm == "base"],
             "payoff": {arm: dict(zip(TYPES, M[i])) for i, arm in enumerate(ARMS)},
             "nash": dict(zip(ARMS, x)), "nash_types": dict(zip(TYPES, y)),
             "best_response": br, "games": len(rows), "device": "mps"}
    OUT.write_text(json.dumps(model, indent=1))
    print("payoff", json.dumps(model["payoff"]))
    print("nash", {k: round(v, 3) for k, v in model["nash"].items()}, "best response", br)
    for ty in TYPES:
        print(ty, {e: (round(v["p"], 2), round(v["mu"]), round(v["sd"])) for e, v in timing[ty].items()})


if __name__ == "__main__":
    main()
