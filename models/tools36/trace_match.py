#!/usr/bin/env python3
"""Play one headless match and print a compact per-tick trace.

    python models/tools36/trace_match.py P0 P1 --seed 92020 --max-steps 400 --every 10
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from genmod import EnvConfig  # noqa: E402
from server.local_server import Match, spec_to_agent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p0"); ap.add_argument("p1")
    ap.add_argument("--seed", type=int, default=92020)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    cfg = EnvConfig(height=23, width=23, max_steps=a.max_steps, bfs_iters=92)
    m = Match(cfg, [spec_to_agent(a.p0), spec_to_agent(a.p1)], a.seed, tick_ms=0)
    t0 = time.perf_counter()
    while not bool(m.state.done):
        m.advance()
        t = int(m.state.step)
        if t % a.every == 0 or bool(m.state.done):
            own = np.asarray(m.state.owner); arm = np.asarray(m.state.army)
            gi = np.asarray(m.state.general_idx); cas = np.asarray(m.state.castle)
            parts = []
            for p in (0, 1):
                land = int((own == p).sum()); tot = int(arm[own == p].sum())
                g = int(arm.reshape(-1)[gi[p]]) if gi[p] >= 0 else -1
                top = int(arm[own == p].max()) if land else 0
                ncas = int((cas & (own == p)).sum())
                parts.append(f"p{p} land={land:3d} army={tot:4d} gen={g:4d} max={top:4d} cas={ncas}")
            print(f"t={t:4d} " + " | ".join(parts), flush=True)
    print("winner", int(m.state.winner), "ticks", int(m.state.step),
          "alive", np.asarray(m.state.alive).tolist(), "notes", m.notes,
          f"{time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
