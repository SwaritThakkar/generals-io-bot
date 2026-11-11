#!/usr/bin/env python3
"""Seat-swapped targeted evaluation for v36+ candidates (not a round robin).

    python models/tools36/eval36.py CANDIDATE OPPONENT --seeds 92020 92021 \
        --max-steps 500 --suite direct500

Writes one JSON row per game to models/tournaments/<cand>_<suite>.jsonl with
seed, seats, cap, winner, ticks, how the game ended (capture / timeout
tiebreak / double capture draw), errors, runtime and decision latency.
Rows already present (same seed+seat+opponent+cap) are skipped, so a run can
resume.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from genmod import EnvConfig  # noqa: E402
from server.local_server import Match, spec_to_agent  # noqa: E402


class Timed:
    def __init__(self, fn):
        self.fn, self.times = fn, []

    def __call__(self, state, player):
        t = time.perf_counter()
        try:
            return self.fn(state, player)
        finally:
            self.times.append(time.perf_counter() - t)


def run_game(cand, opp, seed, seat, max_steps, agents):
    cfg = EnvConfig(height=23, width=23, max_steps=max_steps, bfs_iters=92)
    ca = spec_to_agent(cand)
    timed = Timed(ca.fn)
    ca.fn = timed
    oa = agents[opp]
    seats = [ca, oa] if seat == 0 else [oa, ca]
    t0 = time.perf_counter()
    m = Match(cfg, seats, seed, tick_ms=0)
    m.run(threading.Event())
    st = m.state
    winner = int(st.winner)
    alive = np.asarray(st.alive).tolist()
    if not all(alive):
        end = "capture" if any(alive) else "double_capture_draw"
    else:
        end = "timeout_tiebreak" if winner >= 0 else "timeout_draw"
    owner = np.asarray(st.owner); army = np.asarray(st.army)
    tt = sorted(timed.times) or [0.0]
    return {
        "candidate": cand, "opponent": opp, "seed": seed, "candidate_seat": seat,
        "max_steps": max_steps, "winner": winner, "won": winner == seat,
        "draw": winner < 0, "end": end, "ticks": int(st.step),
        "land": [int((owner == p).sum()) for p in (0, 1)],
        "army": [int(army[owner == p].sum()) for p in (0, 1)],
        "errors": [e for e in m.notes if e],
        "seconds": round(time.perf_counter() - t0, 2),
        "decisions": len(timed.times),
        "act_mean_ms": round(1000 * sum(tt) / len(tt), 2),
        "act_p95_ms": round(1000 * tt[int(0.95 * (len(tt) - 1))], 2),
        "act_max_ms": round(1000 * tt[-1], 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("opponents", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[92020, 92021, 92022])
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--suite", required=True)
    a = ap.parse_args()
    out = ROOT / "models/tournaments" / f"{pathlib.Path(a.candidate).stem}_{a.suite}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["opponent"], r["seed"], r["candidate_seat"], r["max_steps"]))
    agents = {o: spec_to_agent(o) for o in a.opponents}
    for opp in a.opponents:
        for seed in a.seeds:
            for seat in (0, 1):
                if (opp, seed, seat, a.max_steps) in done:
                    continue
                r = run_game(a.candidate, opp, seed, seat, a.max_steps, agents)
                with out.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                res = "W" if r["won"] else "D" if r["draw"] else "L"
                print(f"{pathlib.Path(opp).stem:22s} seed={seed} seat={seat} {res} "
                      f"{r['end']:17s} t={r['ticks']:4d} land={r['land']} "
                      f"{r['seconds']:6.1f}s act={r['act_mean_ms']}ms "
                      f"p95={r['act_p95_ms']} max={r['act_max_ms']} err={r['errors'][:1]}",
                      flush=True)


if __name__ == "__main__":
    main()
