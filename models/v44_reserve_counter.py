"""v44: v42 + three fixes from the human win 20261103_154954 (272 ticks).

There v42 drained its general to 2-15 on castles and a strike, its army sat
diluted over 105 tiles, and the human walked their general 25 tiles,
collecting 84 -> 137. v42's race margin read +10..+40 until 2 ticks too late.

1. Each tile counted once. Home walks that share a corridor each counted the
   land army on it; a later walk now loses (army - 1) for every own cell an
   earlier chosen walk already picked up (both sides).
2. General reserve. A move out of the general must leave a race margin above
   the pad with the departing stack NOT counted as coming home, and the
   general must still hold RESERVE x every hostile walk's arrival (the enemy
   general included, no horizon cap).
3. Counter-strike. When home is losing the race, any stack whose exact walk
   beats the enemy general (+ growth + what they can collect home) and lands
   strictly before the first tick our home falls, and that survives
   verify() rollouts, is launched.
v42's module is loaded privately and its race / walk_safe functions are
replaced in that private copy only (final.py is untouched).
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
from core36 import INF, Econ, PathRush, ctx_for, descent, growth, mv, top_stacks, verify, walk  # noqa: E402
from sim36 import Board, Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v42_for_v44", pathlib.Path(core36.__file__).with_name("v42_collect_race.py"))
_v42 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v42)

_MEM = {}
STATS = {"counter": 0, "reserve_veto": 0}
OFF = set(filter(None, os.environ.get("V44_OFF", "").split(",")))
RESERVE = float(os.environ.get("V44_RESERVE", "0.5"))
HORIZONS = (0, 30, 60)


# --------------------------------------------------------------------------- #
# 1. race margin, each tile once
# --------------------------------------------------------------------------- #
def deliver_once(items, T, sim, q):
    used = tot = 0
    cells, srcs = set(), set()
    owner, army = sim.owner, sim.army
    for d, x, s, ps in sorted(items, key=lambda t: -t[1]):
        if used + d > T or s in cells or (ps & srcs):
            continue
        if "once" not in OFF:
            x -= sum(army[v] - 1 for v in ps & cells if owner[v] == q and army[v] > 1)
            if x <= 0:
                continue
        used += d
        tot += x
        cells |= ps
        srcs.add(s)
    return tot


def race_profile(ctx, sim, p, t_max=_v42.T_MAX, advance=0, exclude=()):
    """(threats, [(T, margin_T)]). `exclude`: our cells that are on a mission
    and must not be counted as coming home."""
    g = sim.gen[p]
    if g < 0:
        return [], [(0, -INF)]
    them = _v42._items_them(ctx, sim, 1 - p, g)
    if advance:
        them = [(max(1, d - advance), x, s, ps) for d, x, s, ps in them]
    them = [t for t in them if t[0] <= t_max]
    if not them:
        return [], []
    us = [t for t in _v42._items_us(ctx, sim, p, g) if t[2] not in exclude]
    G = sim.army[g]
    ds = sorted(set(t[0] for t in them))
    Ts = set(ds)
    for i, a in enumerate(ds):
        for b in ds[i:]:
            if a + b <= t_max:
                Ts.add(a + b)
    prof = []
    for T in sorted(Ts):
        prof.append((T, G + growth(sim, sim.step, T)
                     + deliver_once(us, T - _v42.LEAD - T // 10, sim, p)
                     - deliver_once(them, T, sim, 1 - p)))
    thr = [(x, d, s, list(ps)) for d, x, s, ps in sorted(them, key=lambda t: -t[1])[:6]]
    return thr, prof


def race_margin(ctx, sim, p, t_max=_v42.T_MAX, advance=0):
    thr, prof = race_profile(ctx, sim, p, t_max, advance)
    if not prof:
        return thr, INF
    return thr, min(m for _T, m in prof)


# v42's castle checks (walk_safe) and Econ42 call these module globals
_v42.race_margin = race_margin


# --------------------------------------------------------------------------- #
# 2. general reserve
# --------------------------------------------------------------------------- #
def reserve_ok(ctx, sim, p, keep=None):
    """The general (holding `keep`, default its army) beats RESERVE x every
    hostile walk's arrival, at any distance, now and if the enemy waits H
    ticks first (their producers and the land on the walk keep growing, and
    so does our general). A walk that is harmless now is not harmless in 60
    ticks: the human's general grew 15 -> 84 and collected to 137."""
    if "reserve" in OFF:
        return True
    g = sim.gen[p]
    G = sim.army[g] if keep is None else keep
    o = 1 - p
    t = sim.step
    prods = set(sim.producers)
    owner = sim.owner
    for d, x, s, ps in _v42._items_them(ctx, sim, o, g):
        land = sum(1 for v in ps if owner[v] == o)
        for H in HORIZONS:
            xf = x + (growth(sim, t, H) if s in prods else 0) + land * ((t + H) // sim.land_int - t // sim.land_int)
            if G + growth(sim, t, H + d) < RESERVE * xf:
                return False
    for H in HORIZONS[1:]:
        thr, d = mass_threat(ctx, sim, p, H)
        if thr > 0 and G + growth(sim, t, H + d) < RESERVE * thr:
            return False
    return True


def mass_threat(ctx, sim, p, H):
    """(army the enemy can gather in H moves and walk onto our general, walk
    length). Each move gathers at most one cell, so the mass is the H best
    (army - 1) of their cells plus their general's growth; the walk from
    their general pays 1 per step plus what stands on cells not theirs
    (our cells capped like threat walks: stacks can step aside)."""
    o = 1 - p
    g, tg = sim.gen[p], sim.gen[o]
    if g < 0 or tg < 0:
        return 0, INF
    owner, army = sim.owner, sim.army
    vals = sorted((army[c] - 1 for c in range(len(owner)) if owner[c] == o and army[c] > 1), reverse=True)
    mass = sum(vals[:H]) + 1 + growth(sim, sim.step, H)
    path = descent(ctx, sim, o, tg, g)
    if path is None:
        return 0, INF
    cost = 0
    for v in path[1:-1]:
        cost += 1
        if owner[v] == p:
            cost += min(army[v], core36.STATIC_CAP)
        elif owner[v] < 0:
            cost += army[v]
    return mass - cost, len(path) - 1


def general_move_ok(ctx, sim, p, a, pad):
    """A move out of the general: race margin with the departing stack not
    counted, and the reserve kept."""
    g = sim.gen[p]
    cell, k = divmod(a, 9)
    if cell != g:
        return True
    keep = sim.army[g] // 2 + sim.army[g] % 2 if k >= 5 else 1
    if not reserve_ok(ctx, sim, p, keep):
        STATS["reserve_veto"] += 1
        return False
    s2 = sim.copy()
    s2.step2(*((a, 0) if p == 0 else (0, a)))
    if s2.done:
        return s2.winner == p
    dest = ctx.b.step_to[cell][(k - 1) % 4]
    _thr, prof = race_profile(ctx, s2, p, exclude=(dest,))
    return not prof or min(m for _T, m in prof) > pad


class Econ44(_v42.Econ42):
    def safe_after(self, sim, a, m0, pad):
        if a and a // 9 == sim.gen[self.p] and not general_move_ok(self.ctx, sim, self.p, a, pad):
            return False
        return super().safe_after(sim, a, m0, pad)


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
class Planner44(_v42.Planner42):
    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.econ = Econ44(ctx, p)

    def castle_launch(self, sim, m0, pad):
        # castle sends from the general skip econ.safe_after in v42: add the
        # reserve on the general's keep
        econ = self.econ
        g = sim.gen[self.p]
        for i, (_k, a, path, full, half) in enumerate(self.castle_options(sim)):
            if i >= 4:
                break
            if full[0] == g and not general_move_ok(self.ctx, sim, self.p, a, pad):
                continue
            econ.checks = 0
            s1 = econ.after(sim, a)
            if s1.done or not _v42.walk_safe(self.ctx, s1, self.p, path, pad, self.buffer):
                continue
            if full[0] == g or econ.safe_after(sim, a, m0, pad):
                _v42.STATS["castle_launch"] += 1
                self.cwalk = path
                return a
        return 0

    def follow_castle(self, sim, m0, pad):
        path = self.cwalk
        g = sim.gen[self.p]
        if path and len(path) >= 2 and path[0] == g:
            # the walk passes through the general: the step out is a general move
            if not general_move_ok(self.ctx, sim, self.p, mv(self.ctx.b, path[0], path[1]), pad):
                self.cwalk = None
                return 0
        return super().follow_castle(sim, m0, pad)

    def counter(self, sim):
        """3. Home is losing the race: hit their general first if we can."""
        if "counter" in OFF:
            return 0
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        if tg < 0:
            return 0
        _thr, prof = race_profile(self.ctx, sim, p)
        falls = [T for T, m in prof if m <= 0]
        t_hit = min(falls) if falls else INF
        home = _v42._items_us(self.ctx, sim, o, tg)
        best = None
        for s in top_stacks(sim, p, 8, 5):
            path = descent(self.ctx, sim, p, s, tg)
            if path is None or len(path) < 2:
                continue
            d = len(path) - 1
            if d >= t_hit:
                continue
            x = walk(sim, p, path)
            need = sim.army[tg] + growth(sim, sim.step, d) + deliver_once(home, d - 1, sim, o) + 3
            if x <= need:
                continue
            if best is None or d < best[0]:
                best = (d, path)
        if best is None:
            return 0
        d, path = best
        if not verify(self.ctx, sim, p, lambda: PathRush(self.ctx, p, path), d + 4):
            return 0
        STATS["counter"] += 1
        self.econ.kind, self.econ.path = "rush", path
        return mv(self.ctx.b, path[0], path[1])

    def act(self, sim):
        p = self.p
        if sim.gen[p] >= 0 and sim.gen[1 - p] >= 0 and self.econ.kind != "rush":
            thr, m0 = self.econ.margin_of(sim)
            pad = self.econ.pad + self.econ.pad_frac * core36.mobile_top(sim, 1 - p)
            if m0 <= pad:
                a = self.counter(sim)
                if a and sim.legal(p, a):
                    self.observe(sim)
                    self.why = "counter"
                    return a
        return super().act(sim)


Planner = Planner44


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
        mem = [int(state.tick), Planner44(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
