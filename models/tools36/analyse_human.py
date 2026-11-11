#!/usr/bin/env python3
"""Replay a recorded human game in sim36 and ask the bot to re-decide every
tick, printing its reasoning (why, margin, threats) next to the human stack.

    python models/tools36/analyse_human.py FILE.npz [--bot models/v37_phase_planner.py]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "models")]
import core36  # noqa: E402
from sim36 import Board, Sim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--bot", default="models/v37_phase_planner.py")
ap.add_argument("--every", type=int, default=10)
ap.add_argument("--tail", type=int, default=40)
a = ap.parse_args()

d = np.load(a.file, allow_pickle=True)
owner, army, moves = d["owner"], d["army"], d["moves"]
human = int(d["human"]); bot = 1 - human
board = Board(d["mountain"], d["castle"][0], 23, 23)
gi = [int(np.flatnonzero(owner[0] == p)[0]) for p in (0, 1)]
sim = Sim(board, owner[0], army[0], gi, 0, int(d["max_ticks"]))
spec = importlib.util.spec_from_file_location("bot", ROOT / a.bot)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ctx = core36.Ctx(board)
planner = (getattr(mod, "KillPlanner", None) or mod.Planner)(ctx, bot)
T = len(moves)
mismatch = 0
for t in range(T - 1):
    pred = planner.act(sim)
    pred = pred if pred and sim.legal(bot, pred) else 0
    rec = [int(x) if x >= 0 else 0 for x in moves[t + 1]]
    if pred != rec[bot]:
        mismatch += 1
    show = t % a.every == 0 or t >= T - 1 - a.tail
    if show:
        hs = core36.top_stacks(sim, human, 1, 1)
        hc = hs[0] if hs else -1
        dg = ctx.dist(sim.gen[bot])
        thr = core36.threats(ctx, sim, bot)
        top = max(thr, default=None)
        land, tot = sim.totals()
        print(f"t={t:4d} bot:{planner.why:26s} m0={planner.margin if planner.margin < 10**8 else 'inf':>5} "
              f"G={sim.army[sim.gen[bot]]:4d} | human stack {sim.army[hc] if hc>=0 else 0:4d} "
              f"d={dg[hc] if hc>=0 else -1:3d} hG={sim.army[sim.gen[human]]:4d} "
              f"thr={(top[0], top[1]) if top else None} land b/h={land[bot]}/{land[human]} "
              f"army b/h={tot[bot]}/{tot[human]}{'' if pred == rec[bot] else '  [recorded bot move differs]'}")
    sim.step2(rec[0], rec[1])
    o2, a2 = owner[t + 1].tolist(), army[t + 1].tolist()
    if sim.owner != o2 or sim.army != a2:
        # keep in sync with the real game if anything diverged
        sim.owner, sim.army = o2, a2
print("bot-move mismatches vs recording:", mismatch, "of", T - 1)
