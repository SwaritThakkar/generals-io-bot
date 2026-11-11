"""Check whether the bot actually follows producer-wait -> castle -> general.

Runs seat-swapped games against the repo's wait-then-charge stress bot and
records strategy events as well as winners. Example: ``--games 12``.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import genmod
from models import castle_general_rusher as rusher
from models import v25_multicastle_wait_probcut as candidate
from server.local_server import to_bot_state
from server.bot_protocol import normalise


def _legal_action(state, player, action):
    if action <= 0 or action % 9 == 0:
        return action == 0
    owner, army = np.asarray(state.owner), np.asarray(state.army)
    mountain = np.asarray(state.mountain, bool)
    h, w = owner.shape
    src = action // 9
    if owner.reshape(-1)[src] != player or army.reshape(-1)[src] < 2:
        return False
    r, c = divmod(src, w)
    dr, dc = candidate.DIRS[(action - 1) % 4]
    rr, cc = r + dr, c + dc
    return 0 <= rr < h and 0 <= cc < w and not mountain[rr, cc]


def run(seed, candidate_seat, max_ticks):
    state = genmod.reset(jax.random.PRNGKey(seed), CFG)
    events = {"wait_opportunities": 0, "waits_taken": 0,
              "ready_launches": 0, "path_launches": 0, "castle_captures": 0}
    queues = [[], []]
    prev_owner = np.asarray(state.owner).copy()
    while not bool(state.done) and int(state.step) < max_ticks:
        actions = [0, 0]
        for p in range(2):
            bs = to_bot_state(state, CFG, p)
            if p == candidate_seat:
                plans = candidate._charge_general_plans(bs, p, limit=1)
                if not queues[p]:
                    queues[p] = normalise(candidate.act(bs, p), bs)
                while queues[p] and not _legal_action(state, p, int(queues[p][0])):
                    queues[p].clear()
                    break
                actions[p] = int(queues[p].pop(0)) if queues[p] else 0
                if plans:
                    plan = plans[0]
                    if plan["wait"] > 0:
                        events["wait_opportunities"] += 1
                        src = int(actions[p]) // 9 if actions[p] else -1
                        anchor = plan["source"][0] * bs.w + plan["source"][1]
                        if actions[p] == 0 or src == anchor:
                            events["waits_taken"] += 1
                    else:
                        expected = candidate._charge_action(bs, p, plan)
                        if expected:
                            events["ready_launches"] += 1
                            if actions[p] == expected:
                                events["path_launches"] += 1
            else:
                if not queues[p]:
                    queues[p] = normalise(rusher.act(bs, p), bs)
                while queues[p] and not _legal_action(state, p, int(queues[p][0])):
                    queues[p].clear()
                    break
                actions[p] = int(queues[p].pop(0)) if queues[p] else 0
        state, _, _, _ = genmod.step(state, jnp.asarray(actions, jnp.int32), CFG)
        owner = np.asarray(state.owner)
        c = np.asarray(state.castle, bool)
        events["castle_captures"] += int(np.sum(c & (owner == candidate_seat) & (prev_owner != candidate_seat)))
        prev_owner = owner.copy()
    return int(state.winner), int(state.step), events


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=72100)
    ap.add_argument("--max-ticks", type=int, default=240)
    args = ap.parse_args()
    CFG = genmod.EnvConfig(max_steps=args.max_ticks)
    totals = {"wait_opportunities": 0, "waits_taken": 0,
              "ready_launches": 0, "path_launches": 0, "castle_captures": 0}
    wins = {0: 0, 1: 0, -1: 0}
    for i in range(args.games):
        seat = i % 2
        winner, ticks, events = run(args.seed + i // 2, seat, args.max_ticks)
        wins[winner] += 1
        for key in totals:
            totals[key] += events[key]
        print(f"game={i + 1}/{args.games} seed={args.seed + i // 2} seat={seat} "
              f"winner={winner} ticks={ticks} events={events}", flush=True)
    print(f"wins={wins} aggregate={totals}")
    print("wait-follow-through=" +
          (f"{totals['waits_taken']}/{totals['wait_opportunities']}" if totals["wait_opportunities"] else "no opportunities"))
    print("shortest-path launches=" +
          (f"{totals['path_launches']}/{totals['ready_launches']}" if totals["ready_launches"] else "no ready plans"))


if __name__ == "__main__":
    main()
