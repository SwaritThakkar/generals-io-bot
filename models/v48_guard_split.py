"""v48: v46 + a home guard (defence stacks vs attack stacks) as a branch of
the decision tree.

Why: v46 is beaten by a human who collects. The enemy's attack is built from
many small tiles gathered one per move, while v46's safety only counts
existing stacks (>= 3) and only stops the *general* from leaving; its other
stacks wander off, so an early collected strike finds home with nothing near.

Guard margin, symmetric in gathering (one cell per move for both sides):
    M_us(T)   = best k = T - LEAD values of (army - 1) among our cells within
                T - LEAD steps of our general       (what we can pull home)
    M_them(T) = best k = T - d values of (army - 1) among their cells, plus
                their general's growth, minus the walk cost of d steps onto our
                general                               (gather, then walk d)
    m_guard   = min over T of  G + growth(T) + M_us(T) - M_them(T)
Defence stacks = what M_us needs near home; everything else is the attack
force. Every move (any stack) must keep m_guard above the guard pad, so the
attack force can never take what the guard needs. When m_guard falls to the
pad, the GUARD branch pulls the strongest stack that can reach home in time
toward the general, before the enemy launches. The rest of v46 (race margin,
general reserve, counter, castles, opponent learning) is unchanged.
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
from core36 import INF, STATIC_CAP, ctx_for, descent, growth, mv  # noqa: E402
from sim36 import Board, Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v46_for_v48", pathlib.Path(core36.__file__).with_name("v46_opponent_learner.py"))
_v46 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v46)
_v44 = _v46._v44

_MEM = {}
STATS = {"guard_pull": 0, "guard_veto": 0}
OFF = set(filter(None, os.environ.get("V48_OFF", "").split(",")))
HORIZONS = (8, 14, 20, 30, 45, 60)
LEAD = 2
REACH_K = float(os.environ.get("V48_REACH", "1.5"))   # their tiles within REACH_K * T of our general count
MOB_WINDOW, MOB_LATCH = 15, 30
KAPPA = 0.15
BUDGETS = (8, 16, 24, 32, 48)
TOLERANCE = 3          # guard loss a single move may cause while under the pad
GUARD_FRAC = float(os.environ.get("V48_GUARD_FRAC", "0.05"))   # pad share of the threat


def guard_profile(ctx, sim, p):
    """[(T, margin_T, threat_T)] for the symmetric gather race onto our general."""
    o = 1 - p
    g, tg = sim.gen[p], sim.gen[o]
    if g < 0 or tg < 0:
        return []
    owner, army = sim.owner, sim.army
    dist = ctx.dist(g)
    ours = sorted(((army[c] - 1, dist[c]) for c in range(len(owner))
                   if owner[c] == p and c != g and army[c] > 1), reverse=True)
    # their tiles with their distance to OUR general: a tile counts only if it
    # can be on our general within the horizon
    theirs = sorted(((army[c] - 1, dist[c]) for c in range(len(owner))
                     if owner[c] == o and army[c] > 1 and dist[c] < INF), reverse=True)
    path = descent(ctx, sim, o, tg, g)
    if path is None:
        return []
    d = len(path) - 1
    cost = 0
    for v in path[1:-1]:
        cost += 1
        if owner[v] == p:
            cost += min(army[v], STATIC_CAP)
        elif owner[v] < 0:
            cost += army[v]
    G = army[g]
    out = []
    near = min((dc for _v, dc in theirs), default=INF)
    for T in HORIZONS:
        if T <= d and T <= near:
            continue
        # gather budget: moves left after the first walk could start arriving
        k_them, threat = T - min(d, near), 0
        for val, dc in theirs:
            if k_them <= 0:
                break
            if dc <= REACH_K * T:
                threat += val
                k_them -= 1
        threat += 1 + growth(sim, sim.step, T) - cost
        reach = T - LEAD
        k_us, us = reach, 0
        for val, dc in ours:
            if k_us <= 0:
                break
            if dc <= reach:
                us += val
                k_us -= 1
        out.append((T, G + growth(sim, sim.step, T) + max(0, us) - threat, threat))
    return out


def guard_margin(ctx, sim, p):
    prof = guard_profile(ctx, sim, p)
    if not prof:
        return INF, 0
    m = min(x[1] for x in prof)
    top = max(x[2] for x in prof)
    return m, top


class Econ48(_v44.Econ44):
    #: set by the planner each tick: the guard binds only while the enemy masses
    guard_on = False

    def guard_pad(self, threat):
        return self.pad + GUARD_FRAC * max(0, threat)

    def safe_after(self, sim, a, m0, pad):
        if not super().safe_after(sim, a, m0, pad):
            return False
        if "guard" in OFF or not a or not self.guard_on:
            return True
        m_now, threat = guard_margin(self.ctx, sim, self.p)
        s2 = self.after(sim, a)
        if s2.done:
            return s2.winner == self.p
        m_after, threat2 = guard_margin(self.ctx, s2, self.p)
        gp = self.guard_pad(max(threat, threat2))
        # never take the guard below its pad; when already below, only moves
        # that do not make it worse
        # small moves (expansion with 2-3 army) never freeze the economy
        if m_after > gp or m_after >= m_now - TOLERANCE:
            return True
        STATS["guard_veto"] += 1
        return False


class Planner48(_v46.Planner46):
    def __init__(self, ctx, p, arm="base"):
        super().__init__(ctx, p, arm)
        self.econ = Econ48(ctx, p)
        self.mob_hist = []          # enemy's biggest free stack, last MOB_WINDOW ticks
        self.mob_until = -1
        self.collect = None
        self.cooldown = -1
        self.next_check = 0

    def mobilising(self, sim):
        """The enemy is assembling a strike: its biggest stack off its
        producers is large next to our home defence, or grew fast lately."""
        p = self.p
        top = core36.mobile_top(sim, 1 - p)
        self.mob_hist.append(top)
        if len(self.mob_hist) > MOB_WINDOW:
            self.mob_hist.pop(0)
        g = sim.gen[p]
        home = sim.army[g]
        dist = self.ctx.dist(g)
        home += sum(sim.army[c] - 1 for c in core36.top_stacks(sim, p, 8, 3) if c != g and dist[c] <= 8)
        old = self.mob_hist[0]
        if top >= max(20, 0.5 * home) or (top >= 15 and top >= 1.5 * max(1, old)):
            self.mob_until = sim.step + MOB_LATCH
        return sim.step <= self.mob_until

    def guard_pull(self, sim):
        """Strongest of our stacks that can reach home before the gathered
        strike could, one step toward the general."""
        p = self.p
        g = sim.gen[p]
        prof = guard_profile(self.ctx, sim, p)
        if not prof:
            return 0
        T_bad = min((T for T, m, thr in prof if m <= self.econ.guard_pad(thr)), default=None)
        if T_bad is None:
            return 0
        dist = self.ctx.dist(g)
        best, best_s = None, 0.0
        for s in core36.top_stacks(sim, p, 12, 3):
            if s == g or dist[s] >= T_bad + 10:
                continue
            score = (sim.army[s] - dist[s]) / (1 + dist[s])
            if score > best_s:
                best, best_s = s, score
        if best is None:
            return 0
        path = descent(self.ctx, sim, p, best, g)
        if path is None or len(path) < 2:
            return 0
        a = mv(self.ctx.b, path[0], path[1])
        return a if sim.legal(p, a) else 0

    # -- attack force: collect onto the forward castle, then v36's strike ----
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
        p = self.p
        g = sim.gen[p]
        out = []
        for it in _v44._v42._items_us(self.ctx, sim, p, S):
            if it[2] == g:
                path = descent(self.ctx, sim, p, g, S)
                if path is None or not _v44.general_move_ok(self.ctx, sim, p, mv(self.ctx.b, g, path[1]), pad):
                    continue
            out.append(it)
        return out

    def strike_estimate(self, sim, S, pad):
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        path = descent(self.ctx, sim, p, S, tg)
        if path is None or len(path) < 2:
            return None
        d = len(path) - 1
        us = self.items_to(sim, S, pad)
        home = _v44._v42._items_us(self.ctx, sim, o, tg)
        for T in BUDGETS:
            got = _v44.deliver_once(us, T, sim, p)
            x = core36.walk(sim, p, path, start=sim.army[S] + got)
            need = sim.army[tg] + growth(sim, sim.step, T + d) + _v44.deliver_once(home, T + d - 2, sim, o)
            if x > (1 + KAPPA) * need + 5:
                return T
        return None

    def try_collect(self, sim):
        p = self.p
        econ = self.econ
        if "collect" in OFF or econ.kind == "rush" or econ.guard_on:
            self.collect = None
            return 0
        _thr, m0 = econ.margin_of(sim)
        pad = econ.pad + econ.pad_frac * core36.mobile_top(sim, 1 - p)
        if m0 <= pad:
            return 0
        if self.collect is None:
            if sim.step < max(self.cooldown, self.next_check) or self.phase == "opening":
                return 0
            self.next_check = sim.step + 5
            path, _m = econ.attack_plan(sim, econ.aggression)
            if path is not None:
                return 0
            S = self.staging(sim)
            T = self.strike_estimate(sim, S, pad)
            if T is None:
                return 0
            STATS["collect_start"] = STATS.get("collect_start", 0) + 1
            self.collect = (S, sim.step + T + 6)
        S, deadline = self.collect
        if sim.owner[S] != p or sim.step > deadline:
            self.collect, self.cooldown = None, sim.step + 40
            return 0
        g = sim.gen[p]
        for d, x, s, ps in sorted(self.items_to(sim, S, pad), key=lambda t: -t[1])[:4]:
            path = descent(self.ctx, sim, p, s, S)
            if path is None or len(path) < 2:
                continue
            a = mv(self.ctx.b, path[0], path[1])
            econ.checks = 0
            if (s == g or econ.safe_after(sim, a, m0, pad)) and sim.legal(p, a):
                return a
        self.collect, self.cooldown = None, sim.step + 20
        return 0

    def act(self, sim):
        p = self.p
        if sim.gen[p] >= 0 and sim.gen[1 - p] >= 0:
            self.econ.guard_on = "guard" not in OFF and self.mobilising(sim)
            if not self.econ.guard_on:
                a = self.try_collect(sim)
                if a:
                    self.opp.update(sim)
                    self.observe(sim)
                    self.why = f"{self.gates()}:collect"
                    return a
        if self.econ.guard_on and sim.gen[p] >= 0 and sim.gen[1 - p] >= 0 \
                and self.econ.kind != "rush":
            # capture-in-one and verified rushes keep priority (v36 urgent play)
            tg = sim.gen[1 - p]
            for v in self.ctx.b.nbrs[tg]:
                if sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg]:
                    return super().act(sim)
            m, thr = guard_margin(self.ctx, sim, p)
            if m <= self.econ.guard_pad(thr):
                path, _m = self.econ.attack_plan(sim, self.econ.aggression)
                if path is None:
                    a = self.guard_pull(sim)
                    if a:
                        STATS["guard_pull"] += 1
                        self.opp.update(sim)
                        self.observe(sim)
                        self.why = f"{self.gates()}:guard"
                        return a
        return super().act(sim)


Planner = Planner48


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
        mem = [int(state.tick), Planner48(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
