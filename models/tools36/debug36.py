#!/usr/bin/env python3
"""Per-tick v36 decision trace in a real Match: why, margin, top threat."""
from __future__ import annotations

import argparse
import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from genmod import EnvConfig  # noqa: E402
from server.local_server import Match, spec_to_agent  # noqa: E402
import core36  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cand"); ap.add_argument("opp")
ap.add_argument("--seed", type=int, default=92020)
ap.add_argument("--seat", type=int, default=0)
ap.add_argument("--max-steps", type=int, default=500)
ap.add_argument("--from-tick", type=int, default=0)
ap.add_argument("--to-tick", type=int, default=10**9)
a = ap.parse_args()

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("cand", ROOT / a.cand)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
log = []


def wrapped(state, player):
    act = mod.act(state, player)
    ctx = mod.live_ctx(state)
    scr = mod._MEM[(id(ctx), player)][1]
    sim = core36.Sim.from_state(state, ctx.b)
    thr = core36.threats(ctx, sim, player)
    t = max(thr, default=None)
    g = sim.gen[player]; og = sim.gen[1 - player]
    if a.from_tick <= state.tick <= a.to_tick:
        print(f"t={state.tick:4d} why={getattr(scr,'why','?'):18s} act={act} "
              f"src={act//9 if act else -1} margin={getattr(scr,'margin','?')} "
              f"G={sim.army[g]} oppG={sim.army[og]} thr={(t[0],t[1],t[2]) if t else None}",
              flush=True)
    return act


cand = spec_to_agent(a.cand); cand.fn = wrapped
opp = spec_to_agent(a.opp)
cfg = EnvConfig(height=23, width=23, max_steps=a.max_steps, bfs_iters=92)
m = Match(cfg, [cand, opp] if a.seat == 0 else [opp, cand], a.seed, tick_ms=0)
m.run(threading.Event())
print("winner", int(m.state.winner), "ticks", int(m.state.step))
