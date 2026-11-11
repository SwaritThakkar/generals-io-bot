"""v43: v42 + collect-and-strike + in-game parameter adaptation.

v42 takes castles but, once the map is full, only grows: v36's strike
(attack_plan) launches a single stack, and against a hoarding general no
single stack is ever enough. v43 adds the second half of the human plan:

1. Collect-and-strike (paper section 8.4). Staging cell S = our owned castle
   nearest the enemy general (or the general itself). Every few ticks we
   estimate, for collection budgets T, the stack S would hold after the
   budgeted gather (same greedy Steiner-tree gather as the race margin) and
   what it would deliver on the enemy general along the current walk, against
   their general + growth + what they can collect home in T + d. When the
   estimate wins by a factor (1 + kappa), we collect onto S, one walk at a
   time, each step guarded by the race margin (the general may join only if
   home survives its departure). The strike itself is v36's attack_plan:
   exact walk, race check and verify() rollouts.
2. Live adaptation. Each game the planner tunes its own parameters from what
   it observes:
     kappa   (collect surplus)  down when the enemy hoards in its general and
                                never sends stacks at us, or when our land
                                has stalled; up after a collection that did
                                not end in a strike
     buffer  (castle send risk) up when we lose a castle or the enemy strikes
                                near our general; slowly back down when quiet
     race_slack (strike check)  0 against a hoarding enemy, 2 otherwise
"""
from __future__ import annotations

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

import importlib.util as _ilu  # noqa: E402

import numpy as np  # noqa: E402

import core36  # noqa: E402
from core36 import INF, ctx_for, descent, growth, mv, top_stacks, walk  # noqa: E402
from sim36 import Board, Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v42_for_v43", pathlib.Path(core36.__file__).with_name("v42_collect_race.py"))
_v42 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v42)

_MEM = {}
STATS = {"collect_start": 0, "collect_step": 0, "collect_abort": 0, "strike": 0}
OFF = set(filter(None, os.environ.get("V43_OFF", "").split(",")))
BUDGETS = (8, 16, 24, 32, 48)


class Adapt:
    """Per-game parameters learned from the live game."""

    def __init__(self):
        self.kappa0 = 0.15
        self.kappa_fail = 0.0
        self.buffer = _v42.BUFFER
        self.hoard = 0.5              # EMA of enemy general's share of enemy army
        self.pressure = 0             # ticks an enemy stack sat near our general
        self.land_hist = []
        self.castles_prev = None
        self.quiet = 0

    def observe(self, ctx, sim, p):
        o = 1 - p
        g, tg = sim.gen[p], sim.gen[o]
        land, tot = sim.totals()
        if tg >= 0 and tot[o] > 0:
            self.hoard = 0.97 * self.hoard + 0.03 * (sim.army[tg] / tot[o])
        self.land_hist.append(land[p])
        if len(self.land_hist) > 80:
            self.land_hist.pop(0)
        # pressure: a hostile stack worth a quarter of our general within 8 steps
        hit = False
        if g >= 0:
            dist = ctx.dist(g)
            for s in top_stacks(sim, o, 3, 5):
                if s != tg and dist[s] <= 8 and sim.army[s] >= 0.25 * max(10, sim.army[g]):
                    hit = True
        if hit:
            self.pressure += 1
            self.buffer = min(0.6, self.buffer + 0.01)
            self.quiet = 0
        else:
            self.quiet += 1
            if self.quiet % 50 == 0:
                self.buffer = max(0.1, self.buffer - 0.02)
        cas = sum(1 for c in ctx.castles if sim.owner[c] == p)
        if self.castles_prev is not None and cas < self.castles_prev:
            self.buffer = min(0.6, self.buffer + 0.1)
            self.quiet = 0
        self.castles_prev = cas

    def stalled(self, sim):
        return sim.step > 200 and len(self.land_hist) >= 80 and self.land_hist[-1] - self.land_hist[0] <= 3

    def hoarder(self):
        return self.hoard > 0.5 and self.pressure < 20

    def kappa(self, sim):
        k = self.kappa0 + self.kappa_fail
        if self.hoarder():
            k -= 0.05
        if self.stalled(sim):
            k -= 0.05
        return max(0.0, min(0.8, k))


class Planner43(_v42.Planner42):
    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.adapt = Adapt()
        self.collect = None           # (S, deadline)
        self.cooldown = -1
        self.next_check = 0

    # -- collect-and-strike --------------------------------------------------
    def staging(self, sim):
        p, o = self.p, 1 - self.p
        g, tg = sim.gen[p], sim.gen[o]
        dt = self.ctx.dist(tg)
        best, best_d = g, dt[g]
        for c in self.ctx.castles:
            if sim.owner[c] == p and dt[c] < best_d:
                best, best_d = c, dt[c]
        return best

    def items_to(self, sim, S, pad):
        """Our walks onto S through our own land; the general's walk only if
        home survives its departure."""
        p = self.p
        g = sim.gen[p]
        out = []
        for it in _v42._items_us(self.ctx, sim, p, S):
            d, x, s, ps = it
            if s == g:
                path = descent(self.ctx, sim, p, g, S)
                if path is None or not _v42.walk_safe(self.ctx, sim, p, path, pad, self.adapt.buffer):
                    continue
            out.append(it)
        return out

    def strike_estimate(self, sim, S, pad):
        """Smallest collection budget T whose collected strike from S beats
        the enemy general with (1 + kappa) surplus, else None."""
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        path = descent(self.ctx, sim, p, S, tg)
        if path is None or len(path) < 2:
            return None
        d = len(path) - 1
        us = self.items_to(sim, S, pad)
        home = _v42._items_us(self.ctx, sim, o, tg)
        k = self.adapt.kappa(sim)
        for T in BUDGETS:
            got = _v42.deliver(us, T)
            x = walk(sim, p, path, start=sim.army[S] + got)
            need = sim.army[tg] + growth(sim, sim.step, T + d) + _v42.deliver(home, T + d - 2)
            if x > (1 + k) * need + 5:
                return T
        return None

    def collect_step(self, sim, m0, pad):
        S, deadline = self.collect
        p = self.p
        if sim.owner[S] != p or sim.step > deadline:
            if sim.step > deadline:
                STATS["collect_abort"] += 1
                self.adapt.kappa_fail = min(0.5, self.adapt.kappa_fail + 0.1)   # estimate was optimistic
            self.collect = None
            self.cooldown = sim.step + 40
            return 0
        g = sim.gen[p]
        items = sorted(self.items_to(sim, S, pad), key=lambda t: -t[1])
        for d, x, s, ps in items[:4]:
            path = descent(self.ctx, sim, p, s, S)
            if path is None or len(path) < 2:
                continue
            a = mv(self.ctx.b, path[0], path[1])
            self.econ.checks = 0
            if s == g or self.econ.safe_after(sim, a, m0, pad):
                STATS["collect_step"] += 1
                return a
        # nothing left to bring: hand over to the strike check / planner
        self.collect = None
        self.cooldown = sim.step + 20
        return 0

    def maybe_collect(self, sim, pad):
        if "collect" in OFF or sim.step < self.cooldown or sim.step < self.next_check:
            return
        self.next_check = sim.step + 5
        if self.phase == "opening":
            return
        S = self.staging(sim)
        T = self.strike_estimate(sim, S, pad)
        if T is not None:
            STATS["collect_start"] += 1
            self.collect = (S, sim.step + T + 6)

    # -- tick ----------------------------------------------------------------
    def act(self, sim):
        p = self.p
        g = sim.gen[p]
        if g < 0:
            return 0
        tg = sim.gen[1 - p]
        if "adapt" not in OFF:
            self.adapt.observe(self.ctx, sim, p)
            self.buffer = self.adapt.buffer
            self.econ.race_slack = 0 if self.adapt.hoarder() else 2
        if tg >= 0 and not self.econ.kind == "rush":
            econ = self.econ
            thr, m0 = econ.margin_of(sim)
            pad = econ.pad + econ.pad_frac * core36.mobile_top(sim, 1 - p)
            if m0 > pad:
                path, _m = econ.attack_plan(sim, econ.aggression)
                if path is None:
                    if self.collect is None:
                        self.maybe_collect(sim, pad)
                    if self.collect is not None:
                        a = self.collect_step(sim, m0, pad)
                        if a and sim.legal(p, a):
                            self.observe(sim)
                            self.margin = m0
                            self.why = "collect"
                            return a
                else:
                    STATS["strike"] += 1
                    self.collect = None
        return super().act(sim)


Planner = Planner43


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
        mem = [int(state.tick), Planner43(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
