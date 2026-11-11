"""Headless end-to-end check of arena.session against every playable bot.

    python -m arena.selftest                    # every fast/ok bot, full games
    python -m arena.selftest --bots final v36_scripted_guard --max-ticks 200
    python -m arena.selftest --tiers fast ok slow

Each game: one GameSession at tick_ms=0 against a scripted "human" that
queues random legal-looking moves (some go stale, the queue is sometimes
cleared). Results go to a temporary directory (arena.store paths are
monkeypatched), never to the real arena_results/. Checks per game:

    * the game ends by itself (capture or timeout) within max_ticks
    * exactly one record in games.jsonl with every store.KEYS field, winner
      consistent with the Sim, and a replay .npz with T+1 boards and T moves
    * finish() is idempotent; view() and stats() keep working after the end

Plus quick unit checks of the API (queue validation, stale queue skipping,
tick timing, pause, resign and quit).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import random
import tempfile
import time
import traceback
import warnings

# quiet pygame (imported via arena.board_draw) in every worker process
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
warnings.filterwarnings("ignore", category=UserWarning)


def patch_store(tmp: pathlib.Path) -> None:
    from arena import store
    store.RESULTS = tmp
    store.GAMES = tmp / "games.jsonl"
    store.REPLAYS = tmp / "replays"


class ScriptedHuman:
    """Random queued moves from owned stacks; mimics a human's clicking."""

    def __init__(self, session, rng):
        self.s, self.rng = session, rng

    def think(self):
        s, rng = self.s, self.rng
        sim, p = s.sim, s.human_seat
        if rng.random() < 0.03:
            s.clear_queue()
        if len(s.queued()) > 3 or rng.random() < 0.3:
            return
        cells = [c for c in range(s.board.n) if sim.owner[c] == p and sim.army[c] >= 2]
        if not cells:
            return
        c = rng.choice(cells)
        for _ in range(rng.randint(1, 4)):        # a short path, possibly going stale
            dirs = [d for d in range(4) if s.board.step_to[c][d] >= 0]
            if not dirs:
                break
            d = rng.choice(dirs)
            assert s.queue_move(c, d, rng.random() < 0.15)
            c = s.board.step_to[c][d]


def wait_tick(s, now=0.0, timeout=60.0) -> bool:
    """Call update until one tick resolves (the bot thread may still think)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if s.update(now):
            return True
        if s.stats()["done"]:
            return False
        time.sleep(0.0005)
    raise TimeoutError("no tick within timeout")


def play_one(args) -> dict:
    """One full game in a temp results dir. Runs in a worker process."""
    bot_id, seed, seat, max_ticks, tmpdir = args
    tmp = pathlib.Path(tmpdir) / bot_id
    tmp.mkdir(parents=True, exist_ok=True)
    patch_store(tmp)
    from arena import store
    from arena.board_draw import BoardView
    from arena.session import GameSession
    out = {"bot": bot_id, "ok": False, "problems": []}
    bad = out["problems"]
    try:
        t0 = time.perf_counter()
        s = GameSession(bot_id, player="  selftest  ", human_seat=seat, seed=seed,
                        max_ticks=max_ticks, tick_ms=0)
        human = ScriptedHuman(s, random.Random(seed))
        ticks = 0
        while not s.stats()["done"]:
            human.think()
            if wait_tick(s, timeout=120.0):
                ticks += 1
            if ticks > max_ticks + 1:
                bad.append("game did not end at max_ticks")
                break
        st = s.stats()
        out.update(ticks=st["tick"], end=st["end"], winner=st["winner"],
                   secs=round(time.perf_counter() - t0, 1), bot_passes=s.bot_passes,
                   bot_errors=s.agent.errors, bot_illegal=s.agent.illegal,
                   load_error=s.agent.load_error)
        if st["end"] not in ("capture", "timeout"):
            bad.append(f"end={st['end']}")
        if st["end"] == "timeout" and st["tick"] != max_ticks:
            bad.append(f"timeout at tick {st['tick']}")
        if st["winner"] != s.sim.winner:
            bad.append("stats winner != sim winner")
        if s.agent.load_error:
            bad.append(f"load: {s.agent.load_error}")
        # finish() ran automatically; calling it again must not write twice
        rec = s.finish("quit")
        rec2 = s.finish()
        if rec is not rec2:
            bad.append("finish not idempotent")
        lines = [json.loads(x) for x in store.GAMES.read_text().splitlines() if x.strip()]
        if len(lines) != 1:
            bad.append(f"{len(lines)} records in games.jsonl")
        else:
            r = lines[0]
            missing = [k for k in store.KEYS if k not in r]
            if missing:
                bad.append(f"record missing {missing}")
            want = ("draw" if s.sim.winner < 0 else
                    "human" if s.sim.winner == seat else "bot")
            if r["winner"] != want:
                bad.append(f"record winner {r['winner']} != {want}")
            if r["player"] != "selftest" or r["bot_id"] != bot_id or r["ticks"] != st["tick"]:
                bad.append("record fields wrong")
            out["result"] = r["winner"]
            import numpy as np
            z = np.load(store.REPLAYS / f"{r['id']}.npz")
            T = r["ticks"]
            if z["owner"].shape != (T + 1, 529) or z["army"].shape != (T + 1, 529) \
                    or z["moves"].shape != (T, 2):
                bad.append(f"replay shapes {z['owner'].shape} {z['moves'].shape}")
        v = s.view(5)
        if not isinstance(v, BoardView) or v.selection != 5 or len(v.owner) != 529:
            bad.append("view() after end")
        if s.update(1e9):
            bad.append("update ticked after end")
        out["ok"] = not bad
    except Exception as exc:
        bad.append(f"{type(exc).__name__}: {exc}")
        out["trace"] = traceback.format_exc(limit=6)
    return out


def unit_checks(tmpdir) -> list[tuple[str, bool, str]]:
    """API behaviour against the cheap expander bot."""
    tmp = pathlib.Path(tmpdir) / "_unit"
    tmp.mkdir(parents=True, exist_ok=True)
    patch_store(tmp)
    from arena import store
    from arena.session import GameSession
    res = []

    def check(name, cond, note=""):
        res.append((name, bool(cond), note))

    s = GameSession("expander", "unit", human_seat=1, seed=3, max_ticks=100, tick_ms=100)
    b = s.board
    g = s.sim.gen[1]
    mtn = next((c for c in range(b.n) if b.mountain[c]), None)
    check("queue rejects mountain source", mtn is None or not s.queue_move(mtn, 0, False))
    check("queue rejects off-board", not s.queue_move(0, 0, False) and not s.queue_move(-1, 1, False)
          and not s.queue_move(b.n, 0, False))
    nb_dirs = [d for d in range(4) if b.step_to[g][d] >= 0]
    check("queue accepts own general", s.queue_move(g, nb_dirs[0], False) and len(s.queued()) == 1)
    s.clear_queue()
    check("clear_queue", s.queued() == [])
    # timing: first update starts the clock, the tick waits tick_ms
    check("first update starts clock", not s.update(0.0))
    check("no tick before tick_ms", not s.update(50.0))
    check("tick at tick_ms", wait_tick(s, 100.0) and s.stats()["tick"] == 1)
    check("set_speed", (s.set_speed(0), s.tick_ms == 0)[1])
    # stale queue: a cell we do not own, then a legal move; the legal one plays
    foreign = next(c for c in range(b.n) if s.sim.owner[c] == -1 and not b.mountain[c]
                   and b.step_to[c][0] >= 0)
    while s.sim.army[g] < 2:
        wait_tick(s, 1e6)
    d = nb_dirs[0]
    s.queue_move(foreign, 0, False)
    s.queue_move(g, d, True)
    before = s.sim.army[g]
    wait_tick(s, 2e6)
    played = [m for m in s.view().last_moves if m[3] == 1]
    check("stale move skipped, next plays", played == [(g, d, True, 1)] and s.queued() == [],
          f"{played} army {before}->{s.sim.army[g]}")
    # pause
    s.paused = True
    check("paused: no tick", not s.update(3e6) and not s.update(4e6))
    s.paused = False
    check("unpause: tick", wait_tick(s, 5e6))
    st = s.stats()
    check("stats keys", set(st) == {"tick", "land", "army", "castles", "done", "winner", "end",
                                    "bot_thinking"}, str(sorted(st)))
    s.resign()
    rec = s.finish()
    check("resign", rec["end"] == "resign" and rec["winner"] == "bot" and s.stats()["done"]
          and s.stats()["winner"] == s.bot_seat and rec["human_seat"] == 1)
    q = GameSession("random", "unit", seed=None, max_ticks=50, tick_ms=0)
    r2 = q.finish("quit")
    check("quit at tick 0", r2["end"] == "quit" and r2["ticks"] == 0 and not store.counted(r2))
    check("seed=None picks cached map", isinstance(q.seed, int))
    n = len(store.load_games())
    check("records written", n == 2, f"{n} records")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bots", nargs="*", help="bot ids (default: every bot in --tiers)")
    ap.add_argument("--tiers", nargs="*", default=["fast", "ok"])
    ap.add_argument("--max-ticks", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    from arena.bots import list_bots
    bots = [b for b in list_bots(include_broken=True)
            if (b.id in a.bots if a.bots else b.tier in a.tiers)]
    with tempfile.TemporaryDirectory(prefix="arena_selftest_") as tmpdir:
        print("unit checks")
        unit = unit_checks(tmpdir)
        for name, ok, note in unit:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}  {note if not ok else ''}")
        jobs = [(b.id, 1 + (i * 7) % 200, i % 2, a.max_ticks, tmpdir) for i, b in enumerate(bots)]
        print(f"\n{len(jobs)} full games ({a.max_ticks} ticks cap, {a.workers} workers)")
        t0 = time.perf_counter()
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
            results = pool.map(play_one, jobs, chunksize=1)
        print(f"{'bot':<36} {'tier':<6} {'res':<5} {'ticks':>5} {'end':<8} {'human':<6} "
              f"{'secs':>6} {'pass':>4} {'err':>4}  problems")
        tiers = {b.id: b.tier for b in bots}
        for r in results:
            print(f"{r['bot']:<36} {tiers[r['bot']]:<6} {'PASS' if r['ok'] else 'FAIL':<5} "
                  f"{r.get('ticks', '-'):>5} {str(r.get('end')):<8} {r.get('result', '-'):<6} "
                  f"{r.get('secs', '-'):>6} {r.get('bot_passes', '-'):>4} {r.get('bot_errors', '-'):>4}  "
                  f"{'; '.join(r['problems'])}")
            if r.get("trace"):
                print(r["trace"])
        n_ok = sum(r["ok"] for r in results)
        u_ok = sum(ok for _n, ok, _x in unit)
        print(f"\nunit {u_ok}/{len(unit)}  games {n_ok}/{len(results)}  "
              f"({time.perf_counter() - t0:.0f}s)")
        raise SystemExit(0 if n_ok == len(results) and u_ok == len(unit) else 1)


if __name__ == "__main__":
    main()
