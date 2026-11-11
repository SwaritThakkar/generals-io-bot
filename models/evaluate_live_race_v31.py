#!/usr/bin/env python3
"""Seat-swapped stress evaluation of the live-race candidate.

This is a focused matchup suite, not a round robin. It tests the exact
producer-wait -> all-in general rush, the castle-first variant, and v11.
Results append per game so the run can resume after interruption.
"""
from __future__ import annotations

import json
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genmod import EnvConfig  # noqa: E402
from server.local_server import Match, spec_to_agent  # noqa: E402

CHALLENGER = "models/v31_wait_window_reserve.py"
OPPONENTS = (
    "models/general_rusher.py",
    "models/castle_general_rusher.py",
    "models/v11_launch_intercept.py",
)
SEEDS = (92020, 92021, 92022)
OUT = ROOT / "models/tournaments/live_race_v31.jsonl"


def schedule():
    out = []
    k = 0
    for opponent in OPPONENTS:
        for seed in SEEDS:
            out.append((k, opponent, seed, 0)); k += 1
            out.append((k, opponent, seed, 1)); k += 1
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    complete = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                complete.add(int(json.loads(line)["game_id"]))
            except (ValueError, KeyError, TypeError):
                pass

    specs = [CHALLENGER, *OPPONENTS]
    agents = {spec: spec_to_agent(spec) for spec in specs}
    cfg = EnvConfig(height=23, width=23, max_steps=400, bfs_iters=92)
    games = schedule()
    start = time.perf_counter()
    for idx, opponent, seed, challenger_seat in games:
        if idx in complete:
            continue
        ordered = ([CHALLENGER, opponent] if challenger_seat == 0
                   else [opponent, CHALLENGER])
        t0 = time.perf_counter()
        match = Match(cfg, [agents[x] for x in ordered], seed, tick_ms=0)
        match.run(threading.Event())
        winner = int(match.state.winner)
        row = {
            "game_id": idx,
            "opponent": opponent,
            "seed": seed,
            "challenger_seat": challenger_seat,
            "winner_seat": winner,
            "challenger_won": winner == challenger_seat,
            "draw": winner < 0,
            "ticks": int(match.state.step),
            "errors": [e for e in match.notes if e],
            "seconds": round(time.perf_counter() - t0, 3),
        }
        with OUT.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(f"{idx + 1:2d}/{len(games)} v31 vs {pathlib.Path(opponent).stem} "
              f"seed={seed} v31-seat={challenger_seat} "
              f"result={'W' if row['challenger_won'] else 'D' if row['draw'] else 'L'} "
              f"ticks={row['ticks']} {row['seconds']:.1f}s", flush=True)

    rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
    print(f"Completed {len(rows)}/{len(games)} matches in {time.perf_counter()-start:.1f}s")
    for opponent in OPPONENTS:
        rs = [r for r in rows if r["opponent"] == opponent]
        wins = sum(r["challenger_won"] for r in rs)
        draws = sum(r["draw"] for r in rs)
        print(f"{pathlib.Path(opponent).stem}: {wins}W/{draws}D/"
              f"{len(rs)-wins-draws}L across {len(rs)} seat-balanced games")


if __name__ == "__main__":
    main()
