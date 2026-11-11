"""v41: v37 plus a surplus-castle rule (from recorded human games).

Replays of human wins showed v37 taking zero castles in an 886-tick game
while the human took five from tick 380. v37 had 400-660 army and a home
margin of 116-272 (its general alone beat every hostile walk by that much),
yet its castle plan only knew all-in walks: emptying the general was rightly
refused, and the "send half" move was never considered. The planner also
kept choosing gather_home with that margin.

Rule (checked each tick after v37's urgent play: capture, verified rush,
defence): if a castle can be captured by an exact walk from
  * the general sending half (or all-but-one if that is still safe), or
  * any other stack of ours,
and the strict v36 invariant still holds after the send, detach the stack
and walk it all-in to the cheapest such castle. A castle yields 0.5 army per
tick for the rest of the game; surplus left in the general yields nothing.
gather_home is only offered to the planner when the home margin is low.
"""
from __future__ import annotations

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
import core37  # noqa: E402
from core36 import INF, ctx_for, descent, growth, mv, top_stacks, walk  # noqa: E402
from sim36 import Board, Sim  # noqa: E402


def _load(name, file):
    spec = _ilu.spec_from_file_location(name, pathlib.Path(core36.__file__).with_name(file))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_v37 = _load("_v37_for_v41", "v37_phase_planner.py")
_v39 = _load("_v39_for_v41", "v39_counter_planner.py")

_MEM = {}
STATS = {"castle_launch": 0, "castle_walk": 0}


class Planner41(_v37.Planner):
    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.cwalk = None

    def castle_surplus(self, sim, m0, pad):
        p, o = self.p, 1 - self.p
        econ = self.econ
        g = sim.gen[p]
        G = sim.army[g]
        dist_g = self.ctx.dist(g)
        castles = [c for c in self.ctx.castles if sim.owner[c] != p and dist_g[c] < 30]
        castles.sort(key=lambda c: sim.army[c] + dist_g[c])
        sources = [g] + [s for s in top_stacks(sim, p, 4, 15) if s != g]
        best = None
        for c in castles[:4]:
            enemy = sim.owner[c] == o
            for s in sources:
                path = descent(self.ctx, sim, p, s, c)
                if path is None or len(path) < 2 or len(path) > 25:
                    continue
                d = len(path) - 1
                need = sim.army[c] + (growth(sim, sim.step, d) if enemy else 0)
                modes = (True, False) if s == g else (False,)
                for half in modes:
                    sent = sim.army[s] // 2 if half else sim.army[s] - 1
                    if sent < 2:
                        continue
                    x = walk(sim, p, path, start=sent + 1)
                    if x <= need:
                        continue
                    a = mv(self.ctx.b, s, path[1], half=half)
                    econ.checks = 0
                    if not econ.safe_after(sim, a, m0, pad):
                        continue
                    key = (d, sim.army[c])
                    if best is None or key < best[0]:
                        best = (key, a, path[1:])
                    break
        if best is None:
            return 0
        STATS["castle_launch"] += 1
        self.cwalk = best[2]
        return best[1]

    def follow_castle(self, sim, m0, pad):
        path = self.cwalk
        p = self.p
        if not path or len(path) < 2 or sim.owner[path[0]] != p or sim.army[path[0]] < 2:
            self.cwalk = None
            return 0
        c = path[-1]
        if sim.owner[c] == p:
            self.cwalk = None
            return 0
        x = walk(sim, p, path)
        need = sim.army[c] + (growth(sim, sim.step, len(path) - 1) if sim.owner[c] >= 0 else 0)
        if x <= need:
            self.cwalk = None
            return 0
        a = mv(self.ctx.b, path[0], path[1])
        self.econ.checks = 0
        if not self.econ.safe_after(sim, a, m0, pad):
            self.cwalk = None
            return 0
        self.cwalk = path[1:]
        STATS["castle_walk"] += 1
        return a

    def act(self, sim):
        p = self.p
        g = sim.gen[p]
        if g < 0:
            return 0
        econ = self.econ
        tg = sim.gen[1 - p]
        if tg >= 0:
            for v in self.ctx.b.nbrs[tg]:
                if sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg]:
                    self.why = "capture"
                    return mv(self.ctx.b, v, tg)
        thr, m0 = econ.margin_of(sim)
        pad = econ.pad + econ.pad_frac * core36.mobile_top(sim, 1 - p)
        urgent = m0 <= pad or econ.kind == "rush"
        if not urgent and tg >= 0:
            path, _m = econ.attack_plan(sim, econ.aggression)
            urgent = path is not None
        if not urgent:
            a = self.follow_castle(sim, m0, pad) if self.cwalk else 0
            if not a:
                a = self.castle_surplus(sim, m0, pad)
            if a and sim.legal(p, a):
                self.observe(sim)
                self.margin = m0
                self.why = "castle_surplus"
                return a
        return super().act(sim)

    def replan(self, sim, phase):
        # v39 plan set: gather_home only when the home margin is low
        name, fac = _v39.plan39(self.ctx, sim, self.p, phase, _v37._VALUE, self.weights,
                                self.margin, self.econ.pad)
        self.macro, self.macro_name = fac(), name
        self.until = sim.step + core37.COMMIT[phase]


Planner = Planner41


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
        mem = [int(state.tick), Planner41(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
