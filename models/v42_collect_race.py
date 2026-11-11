"""v42: v37 with a symmetric collection-race margin and castle play.

Paper: "v37: A Phase-Aware Macro Planner ..." section 8 (human games showed
v37 hoarding everything in its general, taking no castles, and not seeing a
spread-out enemy until it was collected).

1. Race margin replaces the standalone home margin. For a horizon T, D_p(X, T)
   is the army player p can bring onto X with T moves (one action per tick:
   chosen walks' lengths must sum to <= T; a walk picks up our own cells on
   its way). Safety is
       min_T  G + growth(T) + D_us(g, T-1) - D_them(g, T)  >  pad
   so stacks parked on castles near home count as defence, and several
   hostile stacks that could be collected count as one threat.
   (Rooted prize-collecting Steiner-tree gathering, as in Human.exe;
   Straka et al. 2026.)
2. Castle layer: after v36's urgent play, take the castle with the best
   payback (garrison + walk) from the general (half or all-but-one), an owned
   castle, or a big stack, preferring castles toward the enemy general, as
   long as the race margin after the send stays above the pad.
3. Faster attacks: the strike race check uses the race margin (our other
   stacks can still come home), so forward castles become strike bases.
v37's modules are not modified; everything is subclassed here.
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
import core37  # noqa: E402
from core36 import (INF, STATIC_CAP, Econ, best_walk, ctx_for, descent, growth,  # noqa: E402
                    mv, route_tree, top_stacks, walk)
from sim36 import Board, Sim  # noqa: E402


def _load(name, file):
    spec = _ilu.spec_from_file_location(name, pathlib.Path(core36.__file__).with_name(file))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_v37 = _load("_v37_for_v42", "v37_phase_planner.py")
_v39 = _load("_v39_for_v42", "v39_counter_planner.py")

_MEM = {}
STATS = {"castle_launch": 0, "castle_walk": 0}
#: ablation hook, e.g. V42_OFF=race,castle,strike
OFF = set(filter(None, os.environ.get("V42_OFF", "").split(",")))
T_MAX = 60
LEAD = int(os.environ.get("V42_LEAD", "2"))
BUFFER = float(os.environ.get("V42_BUFFER", "0.2"))


# --------------------------------------------------------------------------- #
# collection race
# --------------------------------------------------------------------------- #
def _items_us(ctx, sim, p, X):
    """Our walks home: only through our own land (a walk across hostile or
    neutral cells can be cut)."""
    out = []
    owner = sim.owner
    for s in top_stacks(sim, p, 10, 3):
        if s == X:
            continue
        path = descent(ctx, sim, p, s, X)
        if path is None or len(path) < 2 or any(owner[v] != p for v in path[1:]):
            continue
        x = walk(sim, p, path)
        if x > 0:
            out.append((len(path) - 1, x, s, frozenset(path)))
    return out


def _items_them(ctx, sim, att, X):
    tree = route_tree(ctx, sim, att, X, "garrison", cap=STATIC_CAP)
    out = []
    for s in top_stacks(sim, att, 10, 3):
        x, path = best_walk(ctx, sim, att, s, X, tree, cap=STATIC_CAP)
        if path is not None and x > 0:
            out.append((len(path) - 1, x, s, frozenset(path)))
    return out


def deliver(items, T):
    """Greedy budgeted gather: biggest walks first, total moves <= T, a walk
    that crosses an already chosen source (or vice versa) is skipped so no
    stack is counted twice."""
    used = tot = 0
    cells, srcs = set(), set()
    for d, x, s, ps in sorted(items, key=lambda t: -t[1]):
        if used + d > T or s in cells or (ps & srcs):
            continue
        used += d
        tot += x
        cells |= ps
        srcs.add(s)
    return tot


def race_margin(ctx, sim, p, t_max=T_MAX, advance=0):
    """(threats, margin). threats keep core36's (x, d, s, path) shape.
    `advance`: hostile walks are that many steps further along (used on a
    board projected past one of our own walks)."""
    g = sim.gen[p]
    if g < 0:
        return [], -INF
    them = _items_them(ctx, sim, 1 - p, g)
    if advance:
        them = [(max(1, d - advance), x, s, ps) for d, x, s, ps in them]
    them = [t for t in them if t[0] <= t_max]
    if not them:
        return [], INF
    us = _items_us(ctx, sim, p, g)
    G = sim.army[g]
    ds = sorted(set(t[0] for t in them))
    Ts = set(ds)
    for i, a in enumerate(ds):
        for b in ds[i:]:
            if a + b <= t_max:
                Ts.add(a + b)
    m = INF
    for T in sorted(Ts):
        # our walks must land with slack: defence starts while there is time
        m = min(m, G + growth(sim, sim.step, T) + deliver(us, T - LEAD - T // 10) - deliver(them, T))
    thr = [(x, d, s, list(ps)) for d, x, s, ps in sorted(them, key=lambda t: -t[1])[:6]]
    return thr, m


class Econ42(Econ):
    swap_safe = True
    standalone = False
    race_slack = 2          # hostile arrivals up to d + slack must not beat home

    def margin_of(self, sim):
        if "race" in OFF or self.standalone:
            return super().margin_of(sim)
        return race_margin(self.ctx, sim, self.p)

    def safe_after(self, sim, a, m0, pad):
        # a stack leaving the general would be counted by the race as coming
        # straight back: general moves keep v37's standalone invariant
        if a and a // 9 == sim.gen[self.p]:
            self.standalone = True
            try:
                return super().safe_after(sim, a, m0, pad)
            finally:
                self.standalone = False
        return super().safe_after(sim, a, m0, pad)

    def race_safe(self, sim, path, d):
        if "strike" in OFF:
            return super().race_safe(sim, path, d)
        p = self.p
        a = mv(self.ctx.b, path[0], path[1])
        s2 = self.after(sim, a)
        if s2.owner[path[1]] == p:
            s2.army[path[1]] = 1          # the marching stack is not home
        _thr, m = race_margin(self.ctx, s2, p, t_max=d + self.race_slack)
        return m > self.pad


def project(sim, p, path):
    """Board after our stack walks `path` to its end (the enemy passes)."""
    s2 = sim.copy()
    for u, v in zip(path, path[1:]):
        if s2.done or s2.owner[u] != p or s2.army[u] < 2:
            break
        a = mv(s2.b, u, v)
        s2.step2(*((a, 0) if p == 0 else (0, a)))
    return s2


def walk_safe(ctx, sim, p, path, pad, buffer=None):
    """The race margin still holds once the walk is spent at its target,
    with every hostile walk advanced by the walk's length, plus a buffer
    that scales with the biggest hostile stack."""
    s2 = project(sim, p, path)
    if s2.done:
        return s2.winner == p
    _thr, m = race_margin(ctx, s2, p, advance=len(path) - 1)
    top = max([sim.army[c] for c in top_stacks(sim, 1 - p, 1, 1)] + [0])
    return m > pad + (BUFFER if buffer is None else buffer) * top


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
class Planner42(_v37.Planner):
    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.econ = Econ42(ctx, p)
        self.cwalk = None
        self.buffer = BUFFER

    def castle_options(self, sim):
        """(key, action, path) for every castle walk that captures, best first."""
        p, o = self.p, 1 - self.p
        ctx = self.ctx
        g = sim.gen[p]
        tg = sim.gen[o]
        dist_g = ctx.dist(g)
        dist_t = ctx.dist(tg) if tg >= 0 else None
        left = sim.max_steps - sim.step
        castles = [c for c in ctx.castles if sim.owner[c] != p and dist_g[c] < 30]
        sources = [g] + [c for c in ctx.castles if sim.owner[c] == p and sim.army[c] >= 4]
        sources += [s for s in top_stacks(sim, p, 4, 10) if s not in sources]
        out = []
        for c in castles:
            enemy = sim.owner[c] == o
            for s in sources:
                path = descent(ctx, sim, p, s, c)
                if path is None or len(path) < 2 or len(path) > 25:
                    continue
                d = len(path) - 1
                need = sim.army[c] + (growth(sim, sim.step, d) if enemy else 0)
                if 2 * (need + d) > min(left - d, 400):
                    continue                      # does not pay back in time
                modes = (True, False) if (s == g or sim.army[s] >= 40) else (False,)
                for half in modes:
                    sent = sim.army[s] // 2 if half else sim.army[s] - 1
                    if sent < 2:
                        continue
                    if walk(sim, p, path, start=sent + 1) <= need + 1:
                        continue
                    fwd = (dist_t[g] - dist_t[c]) if dist_t is not None and dist_t[c] < INF else 0
                    key = need + d - 0.6 * fwd - (15 if enemy else 0) + (0 if half else 2)
                    out.append((key, mv(ctx.b, s, path[1], half=half), path[1:], path, half))
                    break
        out.sort(key=lambda t: t[0])
        return out

    def castle_launch(self, sim, m0, pad):
        econ = self.econ
        for i, (_k, a, path, full, half) in enumerate(self.castle_options(sim)):
            if i >= 4:
                break
            econ.checks = 0
            s1 = econ.after(sim, a)
            if s1.done or not walk_safe(self.ctx, s1, self.p, path, pad, self.buffer):
                continue
            if full[0] == sim.gen[self.p] or econ.safe_after(sim, a, m0, pad):
                STATS["castle_launch"] += 1
                self.cwalk = path
                return a
        return 0

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
        need = sim.army[c] + (growth(sim, sim.step, len(path) - 1) if sim.owner[c] >= 0 else 0)
        if walk(sim, p, path) <= need:
            self.cwalk = None
            return 0
        a = mv(self.ctx.b, path[0], path[1])
        self.econ.checks = 0
        if not walk_safe(self.ctx, sim, p, path, pad, self.buffer) or (
                path[0] != sim.gen[p] and not self.econ.safe_after(sim, a, m0, pad)):
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
        if "castle" in OFF:
            return super().act(sim)
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
                a = self.castle_launch(sim, m0, pad)
            if a and sim.legal(p, a):
                self.observe(sim)
                self.margin = m0
                self.why = "castle"
                return a
        return super().act(sim)

    def replan(self, sim, phase):
        name, fac = _v39.plan39(self.ctx, sim, self.p, phase, _v37._VALUE, self.weights,
                                self.margin, self.econ.pad)
        self.macro, self.macro_name = fac(), name
        self.until = sim.step + core37.COMMIT[phase]


Planner = Planner42


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
        mem = [int(state.tick), Planner42(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
