#!/usr/bin/env python3
"""Train the v37 per-phase logistic value model from self-play in sim36.

Games pair v36 postures, rusher scripts and v37 macros on harness maps
(genmod.reset), with caps 300/500/800 so "ticks left" varies. Every 5 ticks
both players' features are recorded with the final result (draws dropped).
Train seeds 95000-95159, held-out 95160-95199. Writes models/value37.json
with weights and held-out log-loss / accuracy against a land-only baseline.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "models"), str(ROOT / "models/tools36")]

import core36  # noqa: E402
import core37  # noqa: E402
from arena36 import start  # noqa: E402

POLICIES = {
    "econ": lambda c, p: core36.Econ(c, p),
    "econ_aggr": lambda c, p: core36.Econ(c, p, aggression=-5, pad=0),
    "econ_nopad": lambda c, p: core36.Econ(c, p, pad=-20),
    "rusher": lambda c, p: core36.Rusher(c, p),
    "castle_rusher": lambda c, p: core36.CastleRusher(c, p),
    "expand": lambda c, p: core37.Expand(c, p),
    "strike": lambda c, p: core37.Strike(c, p),
}


def game(seed, f0, f1, cap):
    ctx, sim = start(seed, cap)
    sim.max_steps = cap
    s = [f0(ctx, 0), f1(ctx, 1)]
    rows = []
    while not sim.done:
        if sim.step % 5 == 0:
            summ = core37.summary(sim)
            for p in (0, 1):
                ph = core37.detect_phase(ctx, sim, p, None, summ)
                rows.append((p, ph, core37.features(sim, p, summ)))
        a = [x if x and x > 0 else 0 for x in (s[0].act(sim), s[1].act(sim))]
        sim.step2(a[0], a[1])
    return [(ph, x, 1.0 if sim.winner == p else 0.0) for p, ph, x in rows] if sim.winner >= 0 else []


def collect(seeds, rng):
    data = []
    names = list(POLICIES)
    for seed in seeds:
        a, b = rng.choice(names), rng.choice(names)
        cap = rng.choice((300, 500, 800))
        data += game(seed, POLICIES[a], POLICIES[b], cap)
    return data


def fit(X, y, l2=1e-2, iters=300):
    w = np.zeros(X.shape[1])
    for _ in range(iters):            # Newton steps for L2 logistic regression
        z = np.clip(X @ w, -30, 30)
        pr = 1 / (1 + np.exp(-z))
        g = X.T @ (pr - y) / len(y) + l2 * w
        H = (X * (pr * (1 - pr))[:, None]).T @ X / len(y) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def logloss(X, y, w):
    pr = np.clip(1 / (1 + np.exp(-np.clip(X @ w, -30, 30))), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean()), float(((pr > .5) == y).mean())


def main():
    rng = random.Random(7)
    t0 = time.perf_counter()
    train = collect(range(95000, 95160), rng)
    test = collect(range(95160, 95200), rng)
    print(f"rows train={len(train)} test={len(test)} {time.perf_counter()-t0:.0f}s")
    out = {"features": list(core37.FEATURES), "weights": {}, "report": {}}
    for ph in core37.PHASES:
        tr = [(x, y) for p_, x, y in train if p_ == ph]
        te = [(x, y) for p_, x, y in test if p_ == ph]
        if len(tr) < 50:
            continue
        Xtr = np.array([x for x, _ in tr]); ytr = np.array([y for _, y in tr])
        Xte = np.array([x for x, _ in te]); yte = np.array([y for _, y in te])
        w = fit(Xtr, ytr)
        wb = fit(Xtr[:, :2], ytr)          # baseline: land difference only
        rep = {"n_train": len(tr), "n_test": len(te)}
        if len(te):
            rep["test_logloss"], rep["test_acc"] = logloss(Xte, yte, w)
            rep["baseline_logloss"], rep["baseline_acc"] = logloss(Xte[:, :2], yte, wb)
        out["weights"][ph] = [round(float(v), 5) for v in w]
        out["report"][ph] = rep
        print(ph, rep, dict(zip(core37.FEATURES, np.round(w, 3))))
    (ROOT / "models/value37.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
