"""v39: v37 with three fixes found by replaying recorded human wins.

Human games (models/tournaments/human_games/, analysed with
tools36/analyse_human.py) showed:
  1. Two-wave attacks: one stack trades with the general, a second (or a
     stack merged on the way) finishes it. v36/v37 margins test one walk at
     a time. v39 subtracts the cumulative arrivals of hostile walks in order
     of arrival time, and counts merges within 15 steps.
  2. The human's general sat at 1-12 units for 15-20 ticks during the attack
     and v37 never countered: its race check refuses when home is doomed,
     which is exactly when a faster counter-strike is the only win. v39
     launches any stack whose exact walk beats the drained general (plus
     growth and their reinforcement) if it arrives strictly before the
     earliest hostile walk that would beat our general, or if home holds.
  3. Over-turtling: any 4-unit hostile walk counted as contact, the planner
     then chose gather_home for 300 ticks and fell 27 vs 63 in land. v39
     offers gather_home only when the home margin is actually low and needs
     a threat of at least a quarter of our general for the mid phase.
v37's modules are not modified; everything is subclassed here.
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
from core36 import (INF, Econ, best_walk, ctx_for, growth, mv, reinforcement,  # noqa: E402
                    route_tree, threats, top_stacks)
from sim36 import Board, Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v37_for_v39", pathlib.Path(core36.__file__).with_name("v37_phase_planner.py"))
_v37 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v37)

_MEM = {}
STATS = {"counter": 0}
import os as _os  # noqa: E402
#: ablation hook, e.g. V39_OFF=counter,cumulative,minor,turtle
OFF = set(filter(None, _os.environ.get("V39_OFF", "cumulative,widescan,counter,breach").split(",")))


def cumulative_margin(ctx, sim, p, thr):
    """min over k of G + growth(d_k) - (sum of the k earliest hostile arrivals)."""
    g = sim.gen[p]
    G = sim.army[g]
    m = INF
    tot = 0
    # two waves: the earliest arrivals, as in the recorded human wins
    for x, d, _s, _path in sorted(thr, key=lambda t: t[1])[:2]:
        tot += x
        m = min(m, G + growth(sim, sim.step, d) - tot)
    return m


class Econ39(Econ):
    swap_safe = True

    def margin_of(self, sim):
        thr = threats(self.ctx, sim, self.p, k=4 if "widescan" in OFF else 6)
        if not thr:
            return thr, INF
        if "cumulative" in OFF:
            m = core36.home_margin(self.ctx, sim, self.p, thr=thr, reinf=False)
        else:
            m = cumulative_margin(self.ctx, sim, self.p, thr)
        m -= core36.merge_potential(self.ctx, sim, 1 - self.p, thr,
                                    radius=10 if "widescan" in OFF else 15)
        G = sim.army[sim.gen[self.p]]
        if "minor" not in OFF and m > 2 * self.pad + 6 and max(t[0] for t in thr) < 0.25 * G:
            # a minor walk is not contact: keep the opening economy going
            return [], INF
        return thr, m


def plan_set39(ctx, sim, p, phase, m0, pad):
    out = [x for x in core37.plan_set(ctx, sim, p, phase) if x[0] != "gather_home"]
    if m0 <= 3 * pad + 10 or "turtle" in OFF:
        out.append(("gather_home", lambda: core37.GatherHome(ctx, p)))
    return out


def reachable_free(ctx, sim, p, limit=50):
    """Neutral empty tiles reachable from our land without crossing a castle."""
    owner, army, b = sim.owner, sim.army, ctx.b
    seen = set()
    q = [c for c in range(len(owner)) if owner[c] == p]
    seen.update(q)
    free = 0
    for u in q:
        for v in b.nbrs[u]:
            if v in seen or b.castle[v]:
                continue
            seen.add(v)
            if owner[v] == p or (owner[v] < 0 and army[v] == 0):
                if owner[v] < 0:
                    free += 1
                    if free >= limit:
                        return free
                q.append(v)
    return free


BREACH_FREE = 6
BREACH_HORIZON = 90


class Breach(core37.TakeCastle):
    """Bank everything (no general sends to expansion) for the castle."""

    def expand(self, sim):
        e = self.econ
        return e.expansion(sim, allow_producer=False) or e.gather(sim, sim.gen[self.p]) or 0


def plan39(ctx, sim, p, phase, value, weights, m0, pad):
    """core37.plan with the v39 plan set and breach mode (same scoring)."""
    horizon = min(core37.HORIZON[phase], sim.max_steps - sim.step)
    breach = "breach" not in OFF and reachable_free(ctx, sim, p) < BREACH_FREE
    if breach:
        horizon = min(BREACH_HORIZON, sim.max_steps - sim.step)
    models = core37.opponent_models(ctx, 1 - p)
    tot = sum(weights.values()) or 1.0
    best, best_s, seen = None, -1.0, set()
    cands = plan_set39(ctx, sim, p, phase, m0, pad)
    if breach:
        g = sim.gen[p]
        border = sorted((sim.army[c] + ctx.dist(c)[g], c) for c in ctx.castles
                        if sim.owner[c] != p and any(sim.owner[v] == p for v in ctx.b.nbrs[c]))
        for _s, c in border[:2]:
            cands.insert(0, (f"breach{c}", (lambda c=c: Breach(ctx, p, c))))
    for name, fac in cands:
        vals, sig = {}, []
        for oname, ofac in models.items():
            leaf = core37.rollout(sim, p, fac, ofac, horizon)
            vals[oname] = value(leaf, p, phase)
            sig.append(hash((tuple(leaf.owner), tuple(leaf.army))))
        if tuple(sig) in seen:
            continue
        seen.add(tuple(sig))
        s = 0.5 * sum(weights[o] * v for o, v in vals.items()) / tot + 0.5 * min(vals.values())
        if s > best_s:
            best, best_s = (name, fac), s
    return best


class Planner39(_v37.Planner):
    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.econ = Econ39(ctx, p)
        self.counter_path = None

    def counter_strike(self, sim, thr):
        """Hit a drained enemy general if we arrive before it can hit us."""
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        if tg < 0:
            return 0
        if self.counter_path:
            path = self.counter_path
            if len(path) >= 2 and sim.owner[path[0]] == p and sim.army[path[0]] >= 2:
                self.counter_path = path[1:]
                return mv(self.ctx.b, path[0], path[1])
            self.counter_path = None
        g = sim.gen[p]
        G = sim.army[g]
        # earliest tick at which the hostile walks, summed in arrival order,
        # break our general
        t_them = INF
        tot = 0
        dist_g = self.ctx.dist(g)
        for x, d, s, _path in sorted(thr, key=lambda t: dist_g[t[2]]):
            tot += x
            dd = dist_g[s]
            if tot > G + growth(sim, sim.step, dd):
                t_them = dd
                break
        tree = route_tree(self.ctx, sim, p, tg)
        best, best_d = None, INF
        for s in top_stacks(sim, p, 10, 3):
            if s == g and t_them < INF:
                continue
            x, path = best_walk(self.ctx, sim, p, s, tg, tree)
            if path is None or x <= 0:
                continue
            d = len(path) - 1
            need = sim.army[tg] + growth(sim, sim.step, d) + reinforcement(self.ctx, sim, o, d - 1)
            if x <= need:
                continue
            if s == g and core36.home_margin(self.ctx, sim, p, general_army=1, reinf=False) <= 0:
                continue
            if d < t_them and d < best_d:
                best, best_d = path, d
        if best is None:
            return 0
        STATS["counter"] += 1
        self.counter_path = best[1:]
        return mv(self.ctx.b, best[0], best[1])

    def act(self, sim):
        p = self.p
        if sim.gen[p] < 0:
            return 0
        tg = sim.gen[1 - p]
        if tg >= 0:
            for v in self.ctx.b.nbrs[tg]:
                if sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg]:
                    self.why = "capture"
                    return mv(self.ctx.b, v, tg)
        thr = threats(self.ctx, sim, p, k=6)
        a = 0 if "counter" in OFF else self.counter_strike(sim, thr)
        if a and sim.legal(p, a):
            self.observe(sim)
            self.why = "counter_strike"
            return a
        return super().act(sim)

    def replan(self, sim, phase):
        thr, m0 = self.econ.margin_of(sim)
        name, fac = plan39(self.ctx, sim, self.p, phase, _v37._VALUE, self.weights, m0, self.econ.pad)
        self.macro, self.macro_name = fac(), name
        self.until = sim.step + core37.COMMIT[phase]


def live_ctx(state):
    mountain = np.asarray(state.mountain, bool)
    castle = np.asarray(state.castle, bool)
    key = (state.h, state.w, mountain.tobytes(), castle.tobytes())
    return ctx_for(key, lambda: Board(mountain, castle, state.h, state.w))


Planner = Planner39


def act(state, player):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        mem = [int(state.tick), Planner39(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
