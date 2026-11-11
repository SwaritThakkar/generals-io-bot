"""v47: v46 + a wider Bayesian opponent classifier, army-spread phases, and a
(type x phase) policy.

1. Opponent classifier (naive Bayes, updated on every opponent move; after
   Synnaeve & Bessiere 2011, Weber & Mateas 2009). Eight types:
       rusher           early all-in walks at our general
       gather_rusher    expands, then sweeps into one stack and strikes
       castle_rusher    takes castles near us, then rushes
       expander         land grab with small stacks
       hoarder          banks in its general, little else
       spread_collector castles + parked stacks, collects later (the human)
       general_walker   walks its own general out to attack
       passive          mostly passes
   Each tick the opponent's move is read off the board change (source cell,
   destination, size) and turned into features: kind (pass / expand / castle /
   attack / gather), toward our general or not, from their general or not,
   big or small. Every 10 ticks a state feature is added: how concentrated
   their army is and whether they hold castles. Log-posterior with
   exponential forgetting (lambda = 0.95, about a 20-move memory), tempered
   evidence (x0.5) and a 5% floor per type, so the
   type can change during the game.
2. Phases from army spread, not ticks (hysteresis: 3 ticks to switch):
       clash   a big stack (>= 20% of its side's army) within 10 steps of the
               other side's general
       end     one side holds >= 65% of all land, or the cap is < 60 ticks off
       mass    the opponent is concentrated (top-3 stacks >= 60% of its army)
       spread  the opponent is dispersed (top-3 < 35%)
       expand  neutral land is still >= 30% of the board
       build   otherwise
3. Policy = posterior-weighted type policy, then phase modifiers:
       reserve (general keeps this share of every threat), castle mode
       (full / cheap / off), counter-strike on/off, collect-and-strike on/off
       (v43's gather onto the forward castle, then v36's verified strike).
   Everything still runs inside v44's safety layer (race margin, reserve,
   verify()), as in safe opponent exploitation (Ganzfried & Sandholm 2015).
"""
from __future__ import annotations

import math
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

_spec = _ilu.spec_from_file_location("_v46_for_v47", pathlib.Path(core36.__file__).with_name("v46_opponent_learner.py"))
_v46 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v46)
_v44 = _v46._v44
_v42 = _v44._v42

_MEM = {}
STATS = {"collect_start": 0, "collect_step": 0, "collect_abort": 0}
OFF = set(filter(None, os.environ.get("V47_OFF", "").split(",")))
LOG = os.environ.get("V47_LOG")

KINDS = ("pass", "expand", "castle", "attack", "gather")
#: P(kind), P(toward our general), P(from their general), P(big) per type
LIK = {
    "rusher":           ((0.05, 0.15, 0.05, 0.35, 0.40), 0.85, 0.50, 0.60),
    "gather_rusher":    ((0.05, 0.35, 0.05, 0.15, 0.40), 0.65, 0.30, 0.40),
    "castle_rusher":    ((0.05, 0.25, 0.25, 0.15, 0.30), 0.60, 0.30, 0.40),
    "expander":         ((0.05, 0.75, 0.05, 0.05, 0.10), 0.50, 0.15, 0.05),
    "hoarder":          ((0.35, 0.35, 0.02, 0.03, 0.25), 0.45, 0.05, 0.05),
    "spread_collector": ((0.05, 0.40, 0.15, 0.05, 0.35), 0.50, 0.15, 0.25),
    "general_walker":   ((0.05, 0.30, 0.05, 0.25, 0.35), 0.75, 0.80, 0.60),
    "passive":          ((0.80, 0.10, 0.02, 0.02, 0.06), 0.50, 0.10, 0.05),
}
#: state likelihoods every 10 ticks: P(concentrated), P(holds a castle)
STATE_LIK = {
    "rusher": (0.6, 0.2), "gather_rusher": (0.6, 0.2), "castle_rusher": (0.5, 0.8),
    "expander": (0.2, 0.3), "hoarder": (0.9, 0.1), "spread_collector": (0.2, 0.9),
    "general_walker": (0.6, 0.3), "passive": (0.5, 0.05),
}
#: type policy: reserve, castle mode, counter, collect
TYPE_POLICY = {
    "rusher":           (0.7, "cheap", True, False),
    "gather_rusher":    (0.7, "cheap", True, False),
    "castle_rusher":    (0.6, "full", True, False),
    "expander":         (0.4, "full", False, True),
    "hoarder":          (0.4, "full", False, True),
    "spread_collector": (0.6, "full", True, True),
    "general_walker":   (0.7, "cheap", True, False),
    "passive":          (0.55, "full", False, True),   # a banking rusher looks passive until it launches
}
LAMBDA = 0.95
TEMPER = 0.5            # power posterior: per-move evidence is not independent
FLOOR = 0.05            # every type keeps this share, so the type can change
PHASES = ("expand", "build", "spread", "mass", "clash", "end")
BUDGETS = (8, 16, 24, 32, 48)


# --------------------------------------------------------------------------- #
# 1. Bayesian opponent classifier
# --------------------------------------------------------------------------- #
class Classifier:
    def __init__(self, ctx, p):
        self.ctx, self.p = ctx, p
        self.logp = {t: 0.0 for t in LIK}
        self.prev = None
        self.last_move = None

    def post(self):
        m = max(self.logp.values())
        e = {t: math.exp(v - m) for t, v in self.logp.items()}
        z = sum(e.values())
        k = len(e)
        return {t: (1 - FLOOR) * v / z + FLOOR / k for t, v in e.items()}

    def map_type(self):
        post = self.post()
        t = max(post, key=post.get)
        return t, post[t]

    def infer_move(self, prev, sim):
        """(kind, toward, from_general, big) of the opponent's last move."""
        o, p = 1 - self.p, self.p
        b = self.ctx.b
        best, best_drop = None, 0
        for c in range(len(prev.owner)):
            if prev.owner[c] == o and prev.army[c] >= 2:
                drop = prev.army[c] - (sim.army[c] if sim.owner[c] == o else 0)
                if drop > best_drop:
                    best, best_drop = c, drop
        if best is None:
            return ("pass", False, False, False)
        src = best
        dst = None
        for v in b.nbrs[src]:
            if sim.owner[v] == o and (prev.owner[v] != o or sim.army[v] > prev.army[v] + 1):
                dst = v
                break
        if dst is None:            # bounced: it went at something it did not take
            dst = max(b.nbrs[src], key=lambda v: prev.army[v] if prev.owner[v] != o else -1)
        if prev.owner[dst] == p:
            kind = "attack"
        elif prev.owner[dst] == o:
            kind = "gather"
        elif b.castle[dst]:
            kind = "castle"
        else:
            kind = "expand"
        g = sim.gen[p]
        toward = g >= 0 and self.ctx.dist(g)[dst] < self.ctx.dist(g)[src]
        return (kind, toward, src == prev.gen[o], best_drop >= 20)

    def update(self, sim):
        prev = self.prev
        self.prev = sim.copy()
        if prev is None or prev.step + 1 != sim.step:
            return
        feat = self.infer_move(prev, sim)
        self.last_move = feat
        kind, toward, gen, big = feat
        o = 1 - self.p
        state = None
        if sim.step % 10 == 0:
            _land, tot = sim.totals()
            top3 = sum(sim.army[c] for c in top_stacks(sim, o, 3, 1))
            conc = top3 >= 0.6 * max(1, tot[o])
            held = any(sim.owner[c] == o for c in self.ctx.castles)
            state = (conc, held)
        for t, (pk, pt, pg, pb) in LIK.items():
            ll = math.log(pk[KINDS.index(kind)])
            if kind != "pass":
                ll += math.log(pt if toward else 1 - pt)
                ll += math.log(pg if gen else 1 - pg)
                ll += math.log(pb if big else 1 - pb)
            if state is not None:
                pc, ph = STATE_LIK[t]
                ll += math.log(pc if state[0] else 1 - pc) + math.log(ph if state[1] else 1 - ph)
            self.logp[t] = LAMBDA * self.logp[t] + TEMPER * ll


# --------------------------------------------------------------------------- #
# 2. phases from army spread
# --------------------------------------------------------------------------- #
def spread_stats(sim, q):
    _land, tot = sim.totals()
    stacks = top_stacks(sim, q, 3, 1)
    top3 = sum(sim.army[c] for c in stacks)
    return top3 / max(1, tot[q]), stacks, tot[q]


def raw_phase(ctx, sim, p):
    o = 1 - p
    land, tot = sim.totals()
    g, tg = sim.gen[p], sim.gen[o]
    for a, d in ((o, g), (p, tg)):
        if d < 0:
            continue
        dist = ctx.dist(d)
        for s in top_stacks(sim, a, 3, 5):
            if s != sim.gen[a] and sim.army[s] >= 0.2 * max(1, tot[a]) and dist[s] <= 10:
                return "clash"
    if sim.max_steps - sim.step < 60 or max(land) >= 0.65 * max(1, land[0] + land[1]):
        return "end"
    share_o, _s, _t = spread_stats(sim, o)
    if share_o >= 0.6 and tot[o] >= 40:
        return "mass"
    if share_o < 0.35 and tot[o] >= 40:
        return "spread"
    free = sum(1 for c in range(len(sim.owner)) if sim.owner[c] < 0 and sim.b.passable[c]
               and not sim.b.castle[c] and sim.army[c] == 0)
    if free >= 0.3 * sum(1 for c in range(len(sim.owner)) if sim.b.passable[c]):
        return "expand"
    return "build"


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
class Planner47(_v46.Planner46):
    def __init__(self, ctx, p, arm="base"):
        super().__init__(ctx, p, arm)
        self.clf = Classifier(ctx, p)
        self.phase47 = "expand"
        self.pending = (None, 0)
        self.castle_mode = "full"
        self.counter_on = True
        self.collect_on = False
        self.kappa = 0.15
        self.collect = None
        self.cooldown = -1
        self.next_check = 0

    # -- phase and policy ------------------------------------------------------
    def update_phase(self, sim):
        ph = raw_phase(self.ctx, sim, self.p)
        if ph == self.phase47:
            self.pending = (None, 0)
            return
        cand, n = self.pending
        n = n + 1 if cand == ph else 1
        if n >= 3 or ph == "clash":
            self.phase47, self.pending = ph, (None, 0)
        else:
            self.pending = (ph, n)

    def update_policy(self, sim):
        post = self.clf.post()
        reserve = sum(post[t] * TYPE_POLICY[t][0] for t in post)
        votes = {"full": 0.0, "cheap": 0.0, "off": 0.0}
        counter = collect = 0.0
        for t, w in post.items():
            _r, cm, ct, co = TYPE_POLICY[t]
            votes[cm] += w
            counter += w * ct
            collect += w * co
        castle = max(votes, key=votes.get)
        counter_on, collect_on = counter > 0.3, collect > 0.5
        kappa = 0.15
        ph = self.phase47
        if ph == "clash":
            reserve += 0.1
            counter_on = True
            castle = "cheap" if castle == "full" else castle
        elif ph == "mass":
            reserve += 0.15
            collect_on = False
            castle = "cheap" if castle == "full" else castle
        elif ph == "spread":
            collect_on = True
            kappa -= 0.05
        elif ph == "expand":
            collect_on = False
            castle = "cheap" if castle == "off" else castle
        elif ph == "end":
            land, _tot = sim.totals()
            collect_on = land[self.p] > land[1 - self.p]
        reserve += _v46.ARMS[self.arm][0]
        _v44.RESERVE = max(0.3, min(0.85, reserve))
        self.castle_mode, self.counter_on, self.collect_on, self.kappa = castle, counter_on, collect_on, kappa
        self.castle_gate = castle == "full"
        self.threat_gate = counter_on

    def castle_options(self, sim):
        opts = _v44.Planner44.castle_options(self, sim)
        if self.castle_mode == "full":
            return opts
        if self.castle_mode == "off":
            return []
        g = sim.gen[self.p]
        cheap = _v46.ARMS[self.arm][1]
        return [o for o in opts if o[3][0] != g and sim.army[o[2][-1]] + len(o[3]) - 1 <= cheap]

    def counter(self, sim):
        if not self.counter_on:
            return 0
        return _v44.Planner44.counter(self, sim)

    # -- collect-and-strike (from v43) ----------------------------------------
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
        for it in _v42._items_us(self.ctx, sim, p, S):
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
        home = _v42._items_us(self.ctx, sim, o, tg)
        for T in BUDGETS:
            got = _v44.deliver_once(us, T, sim, p)
            x = walk(sim, p, path, start=sim.army[S] + got)
            need = sim.army[tg] + growth(sim, sim.step, T + d) + _v44.deliver_once(home, T + d - 2, sim, o)
            if x > (1 + self.kappa) * need + 5:
                return T
        return None

    def collect_step(self, sim, m0, pad):
        S, deadline = self.collect
        p = self.p
        if sim.owner[S] != p or sim.step > deadline or not self.collect_on:
            if sim.step > deadline:
                STATS["collect_abort"] += 1
            self.collect = None
            self.cooldown = sim.step + 40
            return 0
        g = sim.gen[p]
        for d, x, s, ps in sorted(self.items_to(sim, S, pad), key=lambda t: -t[1])[:4]:
            path = descent(self.ctx, sim, p, s, S)
            if path is None or len(path) < 2:
                continue
            a = mv(self.ctx.b, path[0], path[1])
            self.econ.checks = 0
            if s == g or self.econ.safe_after(sim, a, m0, pad):
                STATS["collect_step"] += 1
                return a
        self.collect = None
        self.cooldown = sim.step + 20
        return 0

    def try_collect(self, sim):
        p = self.p
        if "collect" in OFF or self.econ.kind == "rush":
            return 0
        econ = self.econ
        _thr, m0 = econ.margin_of(sim)
        pad = econ.pad + econ.pad_frac * core36.mobile_top(sim, 1 - p)
        if m0 <= pad:
            return 0
        if self.collect is None:
            if not self.collect_on or sim.step < max(self.cooldown, self.next_check):
                return 0
            self.next_check = sim.step + 5
            path, _m = econ.attack_plan(sim, econ.aggression)
            if path is not None:
                return 0            # a verified strike is already on: v36 takes it
            S = self.staging(sim)
            T = self.strike_estimate(sim, S, pad)
            if T is None:
                return 0
            STATS["collect_start"] += 1
            self.collect = (S, sim.step + T + 6)
        a = self.collect_step(sim, m0, pad)
        return a if a and sim.legal(p, a) else 0

    # -- tick ------------------------------------------------------------------
    def act(self, sim):
        p = self.p
        if sim.gen[p] < 0 or sim.gen[1 - p] < 0:
            return 0
        if "bayes" not in OFF:
            self.clf.update(sim)
        self.update_phase(sim)
        self.update_policy(sim)
        a = self.try_collect(sim)
        if a:
            self.observe(sim)
            self.why = "collect"
        else:
            a = _v44.Planner44.act(self, sim)
        t, pr = self.clf.map_type()
        self.why = f"{t}:{pr:.2f}|{self.phase47}|{self.castle_mode}:{self.why}"
        if LOG and sim.step % 10 == 0:
            with open(LOG, "a") as f:
                f.write(f"{sim.step} p{p} {self.why} reserve={_v44.RESERVE:.2f}\n")
        return a


Planner = Planner47


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
        mem = [int(state.tick), Planner47(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
