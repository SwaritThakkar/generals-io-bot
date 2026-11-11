"""Find a real position where weighted A* at epsilon = 3, 2 and 1 disagrees.

Plays legal games with the real engine, and at every tick runs the weighted A*
in `strategies/wastar.py` three times from the *same* state. Records a position
only when all three searches return different next moves, all three moves are
ones we would be happy to play, and the epsilon = 1 move scores best.

Nothing about the board is ever edited. Writes position.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import genmod
from genmod.bots import expander_bot
from genmod.scenario import state_to_dict
from strategies import wastar
from strategies.greedy_eval import evaluate

EPSILONS = (3.0, 2.0, 1.0)
HOLD = 12          # ticks simulated forward with each plan held
MIN_TICK = 40      # before this the board is too empty to be worth showing
MAX_TICK = 700
LOSS = -1e5        # evaluate() charges 1e6 for hanging the general


def kind_of(target, owner, castle, general, me, opp, h, w):
    """Coarser target classes, the ones the figure wants to contrast."""
    tr, tc = target
    if general[tr, tc] and owner[tr, tc] == opp:
        return "enemy general"
    gi = np.nonzero(general & (owner == opp))
    if len(gi[0]):
        gr, gc = int(gi[0][0]), int(gi[1][0])
        if abs(tr - gr) + abs(tc - gc) == 1:
            return "next to enemy general"
    if castle[tr, tc] and owner[tr, tc] < 0:
        return "neutral city"
    if castle[tr, tc]:
        return "enemy city"
    if owner[tr, tc] == opp:
        return "enemy tile"
    return "neutral tile"


def losing_attack(plan, owner, army, me):
    """Does this move throw army at something it cannot take?"""
    dr, dc = plan.dest
    sr, sc = plan.mover
    if owner[dr, dc] == me:
        return False
    sent = int(army[sr, sc]) - 1
    return sent <= int(army[dr, dc])


def score_plan(st, plan, cfg, player, key):
    """Hold this plan for HOLD ticks against the scripted opponent, then evaluate."""
    s = st
    path = plan.path
    k = key
    for i in range(HOLD):
        if i == 0:
            a0 = plan.action
        elif i < len(path) - 1:
            (r0, c0), (r1, c1) = path[i], path[i + 1]
            if int(np.asarray(s.owner)[r0, c0]) != player or int(np.asarray(s.army)[r0, c0]) < 2:
                a0 = 0
            else:
                d = genmod.DIRS.index((r1 - r0, c1 - c0))
                a0 = (r0 * cfg.width + c0) * genmod.NUM_CELL_ACTIONS + 1 + d
        else:
            a0 = 0
        k, k1 = jax.random.split(k)
        acts = jnp.stack([jnp.int32(a0), expander_bot(k1, s, cfg, 1 - player)])
        if player == 1:
            acts = acts[::-1]
        s, _, done, _ = genmod.step(s, acts, cfg)
        if bool(done):
            break
    return float(evaluate(s, player, cfg.num_players))


def directions_spread(plans):
    """Are the three targets in at least two distinct compass quadrants?"""
    quads = set()
    for p in plans:
        dr = np.sign(p.target[0] - p.mover[0])
        dc = np.sign(p.target[1] - p.mover[1])
        quads.add((int(dr), int(dc)))
    return len(quads) >= 2


def main(seeds=range(500), budget_s=2400.0):
    cfg = genmod.EnvConfig(height=23, width=23)
    t0 = time.time()
    best = None      # (rank, payload) fallback if the ideal trio never appears
    WANT = {"neutral city", "enemy tile", "next to enemy general"}

    for seed in seeds:
        if time.time() - t0 > budget_s:
            print(f"budget reached after seed {seed}")
            break
        st = genmod.reset(jax.random.PRNGKey(seed), cfg)
        key = jax.random.PRNGKey(10_000 + seed)

        plans = None
        for tick in range(1, MAX_TICK + 1):
            a0 = plans[2].action if plans and plans[2] else 0
            key, k1 = jax.random.split(key)
            acts = jnp.stack([jnp.int32(a0), expander_bot(k1, st, cfg, 1)])
            st, _, done, _ = genmod.step(st, acts, cfg)
            if bool(done):
                break
            plans = [wastar.search(st, 0, e, single_goal=False) for e in EPSILONS]
            if tick < MIN_TICK:
                continue
            if any(p is None for p in plans):
                continue
            moves = {(p.mover, p.dest) for p in plans}
            if len(moves) < 3:
                continue

            owner = np.asarray(st.owner)
            army = np.asarray(st.army)
            castle = np.asarray(st.castle)
            general = np.asarray(st.general)
            if any(losing_attack(p, owner, army, 0) for p in plans):
                continue
            if not directions_spread(plans):
                continue

            key, sk = jax.random.split(key)
            scores = [score_plan(st, p, cfg, 0, sk) for p in plans]
            if any(s <= LOSS for s in scores):
                continue
            if scores[2] < max(scores):          # epsilon = 1 must score best
                continue

            kinds = [kind_of(p.target, owner, castle, general, 0, 1, cfg.height, cfg.width)
                     for p in plans]
            e0, e1, e2 = (p.expanded for p in plans)
            mono = e0 <= e1 <= e2 and e0 < e2
            rank = ((200 if mono else 0) + len(WANT & set(kinds)) * 20
                    + len(set(kinds)) * 5 + min(e2 - e0, 40))
            payload = dict(seed=seed, tick=int(st.step), plans=plans, scores=scores,
                           kinds=kinds, state=st)
            if best is None or rank > best[0]:
                best = (rank, payload)
                print(f"  seed {seed} tick {int(st.step)}: kinds={kinds} "
                      f"expanded={[p.expanded for p in plans]} rank={rank}")
            if set(kinds) == WANT and mono:
                print(f"accepting rank {rank}: {kinds}")
                return dump(payload, cfg)

    if best is None:
        print("nothing found")
        return None
    print(f"settling for rank {best[0]}")
    return dump(best[1], cfg)


def dump(p, cfg):
    st = p["state"]
    d = state_to_dict(st, cfg)
    d["seed"] = int(p["seed"])
    d["tick"] = int(p["tick"])
    d["panels"] = [
        dict(epsilon=pl.epsilon, mover=list(pl.mover), dest=list(pl.dest),
             direction=int(pl.direction), dir_name=genmod.DIR_NAMES[pl.direction],
             target=list(pl.target), target_kind=k, expanded=int(pl.expanded),
             generated=int(pl.generated), ticks=int(pl.ticks),
             cost=float(pl.cost), score=float(s), action=int(pl.action),
             path=[list(x) for x in pl.path])
        for pl, k, s in zip(p["plans"], p["kinds"], p["scores"])
    ]
    Path(__file__).with_name("position.json").write_text(json.dumps(d, indent=1))
    print(f"wrote position.json  seed={d['seed']} tick={d['tick']}")
    for pn in d["panels"]:
        print(f"  eps={pn['epsilon']}: {tuple(pn['mover'])} -> {tuple(pn['dest'])} "
              f"({pn['dir_name']}) target {tuple(pn['target'])} [{pn['target_kind']}] "
              f"expanded={pn['expanded']} score={pn['score']:.1f}")
    return d


if __name__ == "__main__":
    main()
