#!/usr/bin/env python3
"""Train v51's opponent classifier -> models/v51_classifier.json.

    python models/train_v51_classifier.py [--train 93101 93125] [--test 93126 93133]

Data: v46 plays each of 10 opponent types on every seed, both seats, cap 500,
in sim36 (simulation on CPU worker processes). Every 2nd tick the opponent's
14 board features (v51.OppFeatures) are recorded with its type.
Training: a 14-64-64-10 MLP, Adam, cross-entropy, on the Mac GPU (MPS).
Held-out maps report accuracy per game phase, per tick and of the online
posterior v51 actually uses (0.95 forgetting, 0.4 weight).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).with_name("v51_classifier.json")


def opponent(ty, ctx, o, seed=0):
    import adversaries38
    import adversaries42
    import core36
    import core37
    import importlib.util
    if ty == "rusher":
        return core36.Rusher(ctx, o)
    if ty == "castle_rusher":
        return core36.CastleRusher(ctx, o)
    if ty == "expander":
        return core37.Expand(ctx, o)
    if ty == "turtle":
        return core36.Econ(ctx, o, turtle=True)
    if ty == "gather_early":
        return adversaries38.GatherRusher(ctx, o, t0=60)
    if ty == "gather_mid":
        return adversaries38.GatherRusher(ctx, o, t0=120)
    if ty == "gather_late":
        return adversaries38.GatherRusher(ctx, o, t0=200)
    if ty == "spread":
        return adversaries42.SpreadCollector(ctx, o, t_castle=80 + (seed % 3) * 35, t_collect=300 + (seed % 2) * 150)
    if ty == "econ":
        return core36.Econ(ctx, o)
    if ty == "planner":
        spec = importlib.util.spec_from_file_location("v37_cls", ROOT / "models/v37_phase_planner.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.Planner(ctx, o)
    raise ValueError(ty)


def job(args):
    ty, seed, seat = args
    sys.path[:0] = [str(ROOT), str(ROOT / "models"), str(ROOT / "models/tools36")]
    import importlib.util
    from arena36 import start
    spec = importlib.util.spec_from_file_location("v51_tr", ROOT / "models/v51_fast_defence.py")
    v51 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v51)
    ctx, sim = start(seed, 500)
    me = v51._v46.Planner46(ctx, seat)
    op = opponent(ty, ctx, 1 - seat, seed)
    feat = v51.OppFeatures(ctx, seat)
    rows = []
    while not sim.done:
        x = feat.update(sim)
        if sim.step % 2 == 0:
            rows.append(x)
        a = me.act(sim)
        b = op.act(sim)
        a = a if a and a > 0 else 0
        b = b if b and b > 0 else 0
        sim.step2(*((a, b) if seat == 0 else (b, a)))
    return ty, seed, rows


def main():
    sys.path[:0] = [str(ROOT / "models")]
    import importlib.util
    spec = importlib.util.spec_from_file_location("v51_tr", ROOT / "models/v51_fast_defence.py")
    v51 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v51)
    TYPES = v51.CLS_TYPES
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, nargs=2, default=[93101, 93125])
    ap.add_argument("--test", type=int, nargs=2, default=[93126, 93133])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    seeds = list(range(a.train[0], a.train[1] + 1)) + list(range(a.test[0], a.test[1] + 1))
    test = set(range(a.test[0], a.test[1] + 1))
    jobs = [(ty, s, seat) for ty in TYPES for s in seeds for seat in (0, 1)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        games = pool.map(job, jobs, chunksize=2)

    import torch
    assert torch.backends.mps.is_available(), "training must run on the Mac GPU (MPS)"
    dev = torch.device("mps")
    torch.manual_seed(0)
    Xtr, ytr, Xte, yte, test_games = [], [], [], [], []
    for ty, seed, rows in games:
        k = TYPES.index(ty)
        if seed in test:
            Xte += rows
            yte += [k] * len(rows)
            test_games.append((k, rows))
        else:
            Xtr += rows
            ytr += [k] * len(rows)
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    yte_t = torch.tensor(yte, device=dev)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-3)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, len(TYPES))).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    n = Xtr.shape[0]
    for ep in range(60):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            loss = torch.nn.functional.cross_entropy(net((Xtr[idx] - mu) / sd), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 9:
            with torch.no_grad():
                acc = (net((Xte_t - mu) / sd).argmax(1) == yte_t).float().mean().item()
            print(f"epoch {ep + 1}: loss {loss.item():.3f}, held-out per-tick accuracy {acc:.3f}", flush=True)
    net.eval()
    with torch.no_grad():
        pred = net((Xte_t - mu) / sd).argmax(1).cpu().tolist()
        tick = (Xte_t[:, 0] * 500).cpu().tolist()
    for lo, hi in ((0, 100), (100, 200), (200, 300), (300, 500)):
        sel = [i for i, t in enumerate(tick) if lo <= t < hi]
        acc = sum(pred[i] == yte[i] for i in sel) / max(1, len(sel))
        print(f"per-tick accuracy, ticks {lo}-{hi}: {acc:.3f} ({len(sel)} samples)")
    # online posterior exactly as v51 uses it
    with torch.no_grad():
        right = {100: [0, 0], 200: [0, 0], 300: [0, 0]}
        for k, rows in test_games:
            lp = torch.log_softmax(net((torch.tensor(rows, dtype=torch.float32, device=dev) - mu) / sd), 1).cpu()
            post = torch.zeros(len(TYPES))
            for j in range(lp.shape[0]):
                for _ in range(2):           # rows are every 2nd tick
                    post = 0.95 * post + 0.4 * lp[j]
                t = int(rows[j][0] * 500)
                if t in right or t + 1 in right:
                    key = t if t in right else t + 1
                    right[key][0] += int(post.argmax().item() == k)
                    right[key][1] += 1
        for t, (r, c) in right.items():
            print(f"online posterior accuracy at tick {t}: {r / max(1, c):.3f} ({c} games)")
    conf = [[0] * len(TYPES) for _ in TYPES]
    for p_, y_ in zip(pred, yte):
        conf[y_][p_] += 1
    print("confusion (rows = true):")
    for i, ty in enumerate(TYPES):
        print(f"  {ty:13s}", " ".join(f"{v:5d}" for v in conf[i]))
    W = [m for m in net if isinstance(m, torch.nn.Linear)]
    out = {"types": list(TYPES), "mu": mu.cpu().tolist(), "sd": sd.cpu().tolist(), "device": "mps",
           "train_rows": n, "test_rows": len(yte)}
    for i, lin in enumerate(W, 1):
        out[f"W{i}"] = lin.weight.detach().cpu().tolist()
        out[f"b{i}"] = lin.bias.detach().cpu().tolist()
    OUT.write_text(json.dumps(out))
    print("saved", OUT)


if __name__ == "__main__":
    main()
