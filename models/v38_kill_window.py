"""v38: v37 plus a counterfactual "kill window" attack layer.

v37 attacks only after a static margin (arrival > garrison + growth + one
reinforcement), a pessimistic race check and rollouts against four replies
including a full turtle defence all agree. That is stricter than the
defence real opponents (v36/v37 included) play, so it waits too long.

v38 decides attacks by exact counterfactual rollouts in sim36:
  launch the stack along its walk; the opponent answers with `Defender`
  (pull stacks home, hit an adjacent attacker: what v36/v37 do when hit).
  Launch if that captures their general within d+8 ticks AND an enemy
  all-in counter-rush on our home (`Rusher`) does not win first.
A committed strike is re-verified every 4 ticks and dropped if it fails.
When the best stack reaches 60% of the needed force, v38 gathers into it
instead of waiting for it to grow.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

for _root in [pathlib.Path(__file__).resolve().parent, *map(pathlib.Path, sys.path)]:
    for _cand in (_root, _root / "models"):
        if (_cand / "core37.py").exists():
            sys.path.insert(0, str(_cand))
            break
    else:
        continue
    break

import numpy as np  # noqa: E402

import core36  # noqa: E402
import core37  # noqa: E402
from core36 import (Defender, PathRush, Rusher, best_walk, ctx_for, growth,  # noqa: E402
                    mv, reinforcement, route_tree, top_stacks)
from sim36 import Board, Sim  # noqa: E402

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("_v37", pathlib.Path(core36.__file__).with_name("v37_phase_planner.py"))
_v37 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v37)

_MEM = {}
LOG = os.environ.get("V38_LOG")
STATS = {"checks": 0, "launches": 0, "aborts": 0, "assemble": 0}


def counterfactual(ctx, sim, p, path, reply, horizon):
    s2 = sim.copy()
    ours, theirs = PathRush(ctx, p, path), reply()
    for _ in range(horizon):
        if s2.done:
            break
        a, b = ours.act(s2), theirs.act(s2)
        a = a if a and a > 0 else 0
        b = b if b and b > 0 else 0
        s2.step2(a, b) if p == 0 else s2.step2(b, a)
    return s2.winner if s2.done else -2


class KillPlanner(_v37.Planner):
    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.strike = None          # committed path
        self.checked = -10
        self.cands = None

    def candidates(self, sim):
        """[(ratio, path)] walks on the enemy general, best first."""
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        tree = route_tree(self.ctx, sim, p, tg)
        out = []
        G = sim.army[tg]
        for s in top_stacks(sim, p, 6, 8):
            if s == sim.gen[p]:
                continue
            x, path = best_walk(self.ctx, sim, p, s, tg, tree)
            if path is None or x <= 0:
                continue
            d = len(path) - 1
            need = G + growth(sim, sim.step, d) + reinforcement(self.ctx, sim, o, d - 1) + 1
            out.append((x / need, path))
        out.sort(key=lambda t: -t[0])
        return out

    def verified(self, sim, path):
        p, o = self.p, 1 - self.p
        d = len(path) - 1
        STATS["checks"] += 1
        # strongest realistic defence: v36's turtle (sweep home, defend),
        # with its own attack search disabled so the rollout stays cheap
        turtle = lambda: core36.Econ(self.ctx, o, turtle=True, aggression=10 ** 9)  # noqa: E731
        for reply in (lambda: Defender(self.ctx, o), turtle):
            if counterfactual(self.ctx, sim, p, path, reply, d + 8) != p:
                return False
        counter = counterfactual(self.ctx, sim, p, path, lambda: Rusher(self.ctx, o), d + 8)
        return counter != o

    def kill_window(self, sim):
        p = self.p
        o = 1 - p
        if sim.gen[o] < 0:
            return 0
        # continue a committed strike, re-verified every 4 ticks
        if self.strike:
            path = self.strike
            ok = sim.owner[path[0]] == p and sim.army[path[0]] >= 2 and len(path) >= 2
            if ok and sim.step - self.checked >= 4:
                self.checked = sim.step
                ok = self.verified(sim, path)
                if not ok:
                    STATS["aborts"] += 1
            if ok:
                self.strike = path[1:]
                self.why = "kill_walk"
                return mv(self.ctx.b, path[0], path[1])
            self.strike = None
        if sim.step % 2 == 0 or self.cands is None:
            self.cands = self.candidates(sim)
        cands = self.cands
        for ratio, path in cands[:2]:
            if ratio < 0.9 or sim.step % 2:
                break
            if self.verified(sim, path):
                STATS["launches"] += 1
                self.strike, self.checked = path[1:], sim.step
                self.why = "kill_launch"
                if LOG:
                    with open(LOG, "a") as f:
                        f.write(json.dumps({"tick": sim.step, "p": p, "ratio": round(ratio, 2),
                                            "d": len(path) - 1}) + "\n")
                return mv(self.ctx.b, path[0], path[1])
        # nearly ready: gather into the best stack instead of waiting
        phase = core37.detect_phase(self.ctx, sim, p, self.econ.margin)
        if cands and cands[0][0] >= 0.5 and phase in ("mid", "end"):
            target = cands[0][1][0]
            if sim.owner[target] != p:
                return 0
            a = self.econ.gather(sim, target)
            if a:
                thr, m0 = self.econ.margin_of(sim)
                self.econ.checks = 0
                if a // 9 != target and self.econ.safe_after(sim, a, m0, self.econ.pad):
                    STATS["assemble"] += 1
                    self.why = "kill_assemble"
                    return a
        if phase != "opening":
            a = self.raid(sim)
            if a:
                return a
        return 0

    def raid(self, sim):
        """Land tiebreak lever: once no neutral tile is reachable, walk spare
        stacks into enemy tiles they can take (each capture swings land by 2)."""
        p, o = self.p, 1 - self.p
        owner, army, b = sim.owner, sim.army, self.ctx.b
        if self.econ.expansion(sim, allow_producer=False) or self.econ.to_frontier(sim):
            return 0
        g = sim.gen[p]
        for s in top_stacks(sim, p, 6, 4):
            if s == g or s in sim.producers:
                continue
            seen = {s: None}
            q = [s]
            hit = None
            for u in q:
                for v in b.nbrs[u]:
                    if v in seen or b.castle[v]:
                        continue
                    seen[v] = u
                    if owner[v] == o and army[v] < army[s] - len(q) // 8 - 2:
                        hit = v
                        break
                    if owner[v] == p:
                        q.append(v)
                if hit is not None or len(q) > 200:
                    break
            if hit is None:
                continue
            v = hit
            while seen[v] != s:
                v = seen[v]
            a = mv(b, s, v)
            thr, m0 = self.econ.margin_of(sim)
            self.econ.checks = 0
            if self.econ.safe_after(sim, a, m0, self.econ.pad):
                STATS["raid"] = STATS.get("raid", 0) + 1
                self.why = "raid"
                return a
        return 0

    def act(self, sim):
        p = self.p
        if sim.gen[p] < 0:
            return 0
        # immediate capture first
        tg = sim.gen[1 - p]
        if tg >= 0:
            for v in self.ctx.b.nbrs[tg]:
                if sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg]:
                    return mv(self.ctx.b, v, tg)
        thr, m0 = self.econ.margin_of(sim)
        # home defence keeps priority; a strike stack never leaves the general
        if m0 > self.econ.pad or not self.econ.launched(sim, thr):
            a = self.kill_window(sim)
            if a and sim.legal(p, a):
                self.observe(sim)
                return a
        return super().act(sim)


def live_ctx(state):
    mountain = np.asarray(state.mountain, bool)
    castle = np.asarray(state.castle, bool)
    key = (state.h, state.w, mountain.tobytes(), castle.tobytes())
    return ctx_for(key, lambda: Board(mountain, castle, state.h, state.w))


def act(state, player):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        mem = [int(state.tick), KillPlanner(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
