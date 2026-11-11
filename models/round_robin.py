#!/usr/bin/env python3
"""Balanced, resumable partial round robin for every numbered model.

For 25 entrants, each bot faces 10 distinct opponents and every pairing is
played twice with the seats reversed: exactly 250 games. Results are appended
to JSONL immediately so an interrupted tournament can resume safely.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genmod import EnvConfig  # noqa: E402
from server.local_server import Match, spec_to_agent  # noqa: E402


def roster():
    paths = sorted((ROOT / "models").glob("v[0-9]*_*.py"),
                   key=lambda p: int(p.name.split("_", 1)[0][1:]))
    return [f"models/{p.name}" for p in paths]


def schedule(specs):
    n = len(specs)
    if n < 11 or n % 2 == 0:
        raise ValueError("schedule expects an odd roster of at least 11 bots")
    pairs = [(i, (i + d) % n) for i in range(n) for d in range(1, 6)]
    assert len(pairs) == n * 5
    assert len({tuple(sorted(p)) for p in pairs}) == len(pairs)
    assert all(sum(j in p for p in pairs) == 10 for j in range(n))
    games = []
    for k, (i, j) in enumerate(pairs):
        seed = 91000 + k
        games.extend(((k * 2, i, j, seed), (k * 2 + 1, j, i, seed)))
    return games


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=23)
    ap.add_argument("--max-ticks", type=int, default=300)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", default="models/tournaments/rr_v1_v25.jsonl")
    args = ap.parse_args()
    specs = roster()
    if len(specs) != 25 or [int(s.split("v", 1)[1].split("_", 1)[0])
                            for s in specs] != list(range(1, 26)):
        raise SystemExit(f"expected v1-v25; found {len(specs)} files")
    games = schedule(specs)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(int(json.loads(line)["game_id"]))
            except (ValueError, KeyError, TypeError):
                continue
    agents = [spec_to_agent(s) for s in specs]
    cfg = EnvConfig(height=args.size, width=args.size, max_steps=args.max_ticks,
                    bfs_iters=4 * args.size)
    t0 = time.perf_counter()
    for k, (game_id, p0, p1, seed) in enumerate(games):
        if game_id in done:
            continue
        started = time.perf_counter()
        match = Match(cfg, [agents[p0], agents[p1]], seed + args.seed_offset, tick_ms=0)
        match.run(threading.Event())
        ticks = int(match.state.step)
        winner = int(match.state.winner)
        errors = [e for e in match.notes if e]
        row = {
            "game_id": game_id, "p0_model": p0 + 1, "p1_model": p1 + 1,
            "p0_spec": specs[p0], "p1_spec": specs[p1],
            "seed": seed + args.seed_offset, "winner_seat": winner,
            "winner_model": (p0 if winner == 0 else p1) + 1 if winner >= 0 else None,
            "ticks": ticks, "draw": winner < 0, "errors": errors,
            "seconds": round(time.perf_counter() - started, 3),
        }
        with out.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(f"{k + 1:3d}/{len(games)} game={game_id:03d} "
              f"v{p0 + 1:02d} vs v{p1 + 1:02d} seed={row['seed']} "
              f"winner={row['winner_model'] or 'draw'} ticks={ticks} "
              f"{row['seconds']:.1f}s", flush=True)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    summary = []
    for model in range(1, len(specs) + 1):
        games_for = [r for r in rows if r["p0_model"] == model or r["p1_model"] == model]
        wins = sum(r["winner_model"] == model for r in games_for)
        draws = sum(r["draw"] for r in games_for)
        summary.append({"model": model, "spec": specs[model - 1], "games": len(games_for),
                        "wins": wins, "draws": draws, "losses": len(games_for) - wins - draws,
                        "score": round((wins + draws / 2) / max(1, len(games_for)), 4),
                        "mean_ticks": round(sum(r["ticks"] for r in games_for) / max(1, len(games_for)), 1),
                        "errors": sum(bool(r["errors"]) for r in games_for)})
    summary.sort(key=lambda r: (-r["score"], -r["wins"], r["model"]))
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader(); writer.writerows(summary)
    print(f"Finished {len(rows)}/{len(games)} games in {time.perf_counter()-t0:.1f}s")
    print(f"Results: {out.relative_to(ROOT)} and {csv_path.relative_to(ROOT)}")
    for rank, row in enumerate(summary, 1):
        print(f"{rank:2d}. v{row['model']:02d}: {row['score']:.3f} "
              f"({row['wins']}W/{row['draws']}D/{row['losses']}L), "
              f"errors={row['errors']}")


if __name__ == "__main__":
    main()
