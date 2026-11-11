"""v51: v46 + committed fast defence + a learned, wider opponent classifier.

Loss traced (v46 vs GatherRusher t0=200, seed 93003): one 103-army stack
walked 22 -> 1 tiles onto the general over 20 ticks. v46 kept expanding with
2-army tiles, defended for single ticks whenever the race margin dipped and
returned to expanding when it recovered, and only collected from distance 4,
one stack per tick; 400 army sat spread over the board and two stacks (18,
26) were still 3 tiles out when the general (77) fell to 82.

1. Approach detector. A hostile mobile stack (not on a producer, >= max(15,
   0.2 x our general)) whose distance to our general fell by >= 2 over the
   last 5 ticks is INCOMING, with ETA = its distance.
2. Committed defence. While anything is incoming, need = max over T <= ETA+5
   of D_them(g, T) - G - growth(T) + 0.15 x the attacker. If need > 0 every
   move collects INTO the general: a rooted prize-collecting Steiner tree
   (v50, Human.exe) rooted at the general with a budget of ETA - 2 moves, run
   leaf-first; stacks next to the general step in immediately once the
   attacker is within 6 (no parking beside the general). No expansion or
   castles until the attack is beaten or turns away (commit 6 ticks past the
   last approach).
3. Opponent classifier (models/v51_classifier.json, trained on the Mac GPU by
   train_v51_classifier.py): an MLP over 14 per-tick board features of the
   opponent -> 10 types (rusher, castle rusher, expander, turtle, gather
   rusher early / mid / late, spread collector, economy bot, planner bot).
   Online posterior: log p_t = 0.95 log p_{t-1} + 0.4 log P(type | x_t),
   combined with v46's move likelihood. The 10 types map onto the planner's
   four lookahead scripts and open the threat gate from tick 60 when the
   aggressive types dominate; collectors (gather / spread) also lower the
   approach threshold.
V51_OFF=defence,classifier for ablation.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from collections import deque

for _root in [pathlib.Path(__file__).resolve().parent, *map(pathlib.Path, sys.path)]:
    for _cand in (_root, _root / "models"):
        if (_cand / "core37.py").exists():
            sys.path.insert(0, str(_cand))
            break
    else:
        continue
    break

import importlib.util as _ilu  # noqa: E402

import core36  # noqa: E402
from core36 import growth, mv, top_stacks  # noqa: E402
from sim36 import Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v46_for_v51", pathlib.Path(core36.__file__).with_name("v46_opponent_learner.py"))
_v46 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v46)
_v44 = _v46._v44
_v42 = _v44._v42

_MEM = {}
OFF = set(filter(None, os.environ.get("V51_OFF", "").split(",")))
STATS = {"incoming_ticks": 0, "defence_moves": 0, "merge_in": 0}
CLS_TYPES = ("rusher", "castle_rusher", "expander", "turtle", "gather_early", "gather_mid", "gather_late",
             "spread", "econ", "planner")
#: which of v46's four lookahead scripts each type behaves like
TO_SCRIPT = {"rusher": "rusher", "castle_rusher": "castle_rusher", "expander": "expander", "turtle": "hold",
             "gather_early": "rusher", "gather_mid": "rusher", "gather_late": "expander", "spread": "expander",
             "econ": "expander", "planner": "expander"}
AGGRESSIVE = {"rusher", "castle_rusher", "gather_early", "gather_mid", "gather_late", "spread"}
COLLECTORS = {"gather_early", "gather_mid", "gather_late", "spread"}
try:
    CLS = json.loads(pathlib.Path(__file__).with_name("v51_classifier.json").read_text())
except Exception:
    CLS = None


# --------------------------------------------------------------------------- #
# rooted prize-collecting Steiner tree gather (as in v50, Human.exe)
# --------------------------------------------------------------------------- #
def _pcst_tree(ctx, sim, p, root, lam, cand):
    owner, army, nbrs = sim.owner, sim.army, ctx.b.nbrs
    parent = {root: None}
    while True:
        dist, via = {c: 0 for c in parent}, {}
        q = deque(parent)
        while q:
            u = q.popleft()
            for v in nbrs[u]:
                if v not in dist and owner[v] == p:
                    dist[v] = dist[u] + 1
                    via[v] = u
                    q.append(v)
        best, best_gain = None, 0.0
        for v in cand:
            if v in parent or v not in dist:
                continue
            gain, u = 0.0, v
            while u not in parent:
                gain += max(0, army[u] - 1)
                u = via[u]
            gain -= lam * dist[v]
            if gain > best_gain:
                best, best_gain = v, gain
        if best is None:
            return parent
        u = best
        while u not in parent:
            parent[u] = via[u]
            u = via[u]


def pcst_moves(ctx, sim, p, root, budget):
    army = sim.army
    cand = [c for c in top_stacks(sim, p, 30, 2) if c != root]
    if not cand or budget < 1:
        return []
    lo, hi, best = 0.0, float(max(army[c] for c in cand)), None
    for _ in range(7):
        lam = (lo + hi) / 2
        tree = _pcst_tree(ctx, sim, p, root, lam, cand)
        if len(tree) - 1 <= budget:
            best, hi = tree, lam
        else:
            lo = lam
    if best is None:
        return []
    depth = {}

    def d(c):
        if c not in depth:
            depth[c] = 0 if best[c] is None else d(best[c]) + 1
        return depth[c]

    return sorted(((c, best[c]) for c in best if best[c] is not None), key=lambda e: -d(e[0]))


# --------------------------------------------------------------------------- #
# opponent features and classifier
# --------------------------------------------------------------------------- #
class OppFeatures:
    """Per-tick features of the opponent's play (from the board only)."""

    N = 14

    def __init__(self, ctx, p):
        self.ctx, self.p = ctx, p
        self.hist = deque(maxlen=21)
        self.drains = 0
        self.prev_gen = None

    def update(self, sim):
        ctx, p, o = self.ctx, self.p, 1 - self.p
        owner, army = sim.owner, sim.army
        mine = [c for c in range(len(owner)) if owner[c] == o]
        land = len(mine)
        tot = sum(army[c] for c in mine) or 1
        tg, g = sim.gen[o], sim.gen[p]
        gen = army[tg] if tg >= 0 else 0
        stacks = sorted((c for c in mine if c != tg), key=lambda c: -army[c])
        top = stacks[0] if stacks else -1
        topa = army[top] if top >= 0 else 0
        dist = ctx.dist(g) if g >= 0 else None
        dtop = dist[top] if (top >= 0 and dist is not None and dist[top] < core36.INF) else 40
        big = sum(1 for c in mine if army[c] >= 5)
        cas = sum(1 for c in ctx.castles if owner[c] == o)
        if self.prev_gen is not None and self.prev_gen >= 15 and gen < 0.5 * self.prev_gen:
            self.drains += 1
        self.prev_gen = gen
        self.hist.append((sim.step, land, topa, dtop, tot))
        t0, land0, topa0, d0, tot0 = self.hist[0]
        _t5, _l5, topa5, d5, _tt5 = self.hist[-6] if len(self.hist) >= 6 else self.hist[0]
        return [
            sim.step / 500.0, land / 100.0, tot / 300.0, gen / tot, topa / tot, cas / 3.0,
            dtop / 30.0, (d5 - dtop) / 5.0, (land - land0) / 20.0, (topa - topa5) / 30.0,
            big / 20.0, min(self.drains, 5) / 5.0, (tot - tot0) / 60.0, 1.0 if dtop <= 10 else 0.0,
        ]


def mlp_logp(x):
    """log P(type | x) from the trained 2-layer MLP (weights in v51_classifier.json)."""
    mu, sd = CLS["mu"], CLS["sd"]
    h = [(xi - m) / s for xi, m, s in zip(x, mu, sd)]
    for W, b, act in ((CLS["W1"], CLS["b1"], True), (CLS["W2"], CLS["b2"], True), (CLS["W3"], CLS["b3"], False)):
        h = [sum(w * v for w, v in zip(row, h)) + bb for row, bb in zip(W, b)]
        if act:
            h = [max(0.0, v) for v in h]
    m = max(h)
    z = math.log(sum(math.exp(v - m) for v in h)) + m
    return [v - z for v in h]


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
class Planner51(_v46.Planner46):
    def __init__(self, ctx, p, arm="base"):
        super().__init__(ctx, p, arm)
        self.move_w = dict(self.weights)
        self.feat = OppFeatures(ctx, p)
        self.cls_lp = [0.0] * len(CLS_TYPES)
        self.type_p = {k: 1.0 / len(CLS_TYPES) for k in CLS_TYPES}
        self.track = {}           # enemy stack cell -> deque of (tick, dist)
        self.incoming = None      # (eta, attacker army, last tick seen approaching)
        self.def_moves = []
        self.def_t0 = -99

    # -- classifier --------------------------------------------------------------
    def classify(self, sim):
        x = self.feat.update(sim)
        if CLS is None or "classifier" in OFF:
            return
        lp = mlp_logp(x)
        self.cls_lp = [0.95 * a + 0.4 * b for a, b in zip(self.cls_lp, lp)]
        m = max(self.cls_lp)
        e = [math.exp(v - m) for v in self.cls_lp]
        z = sum(e)
        self.type_p = {k: v / z for k, v in zip(CLS_TYPES, e)}

    def observe(self, sim):
        self.weights = self.move_w
        super().observe(sim)
        self.move_w = dict(self.weights)
        if CLS is None or "classifier" in OFF:
            return
        script = {k: 0.02 for k in self.move_w}
        for ty, pr in self.type_p.items():
            script[TO_SCRIPT[ty]] += pr
        comb = {k: math.sqrt(self.move_w[k] / sum(self.move_w.values())) * script[k] for k in self.move_w}
        z = sum(comb.values())
        self.weights = {k: v / z * len(comb) for k, v in comb.items()}

    def update_gates(self, sim):
        super().update_gates(sim)
        if CLS is None or "classifier" in OFF:
            return
        aggr = sum(self.type_p[k] for k in AGGRESSIVE)
        if sim.step >= 60 and aggr > 0.5:
            self.threat_gate = True

    # -- approach detector -------------------------------------------------------
    def detect(self, sim):
        p, o = self.p, 1 - self.p
        g = sim.gen[p]
        dist = self.ctx.dist(g)
        prods = set(sim.producers)
        G = sim.army[g]
        coll = sum(self.type_p.get(k, 0) for k in COLLECTORS)
        thresh = max(12 if coll > 0.5 else 15, (0.15 if coll > 0.5 else 0.2) * G)
        seen = {}
        best = None
        for s in top_stacks(sim, o, 6, 5):
            if s in prods or dist[s] >= core36.INF or sim.army[s] < thresh:
                continue
            # follow a stack across ticks: it moved to an adjacent cell or stayed
            prev = self.track.get(s)
            if prev is None:
                for v in self.ctx.b.nbrs[s]:
                    if v in self.track:
                        prev = self.track[v]
                        break
            h = deque(prev or [], maxlen=6)
            h.append((sim.step, dist[s]))
            seen[s] = h
            if len(h) >= 3 and h[0][1] - dist[s] >= 2 and sim.step - h[0][0] <= 6:
                if best is None or dist[s] < best[0]:
                    best = (dist[s], sim.army[s], sim.step)
        self.track = seen
        if best is not None:
            self.incoming = best
        elif self.incoming is not None and sim.step - self.incoming[2] > 6:
            self.incoming = None
        return self.incoming

    def defence_need(self, sim, eta, att):
        p = self.p
        g = sim.gen[p]
        them = _v42._items_them(self.ctx, sim, 1 - p, g)
        G = sim.army[g]
        need = -1e9
        for T in range(1, eta + 6):
            need = max(need, _v44.deliver_once(them, T, sim, 1 - p) - G - growth(sim, sim.step, T))
        return need + 0.15 * att

    def defend_move(self, sim):
        inc = self.detect(sim)
        if inc is None:
            return 0
        STATS["incoming_ticks"] += 1
        eta, att, _t = inc
        if self.defence_need(sim, eta, att) <= 0:
            return 0
        p = self.p
        g = sim.gen[p]
        owner, army, b = sim.owner, sim.army, self.ctx.b
        # close in: stacks next to the general step in now, no parking beside it
        best = max((v for v in b.nbrs[g] if owner[v] == p and army[v] >= (2 if eta <= 6 else 10)),
                   key=lambda v: army[v], default=None)
        if best is not None:
            if True:
                STATS["merge_in"] += 1
                return mv(b, best, g)
        if not self.def_moves or sim.step - self.def_t0 >= 4:
            self.def_moves = pcst_moves(self.ctx, sim, p, g, max(1, eta - 2))
            self.def_t0 = sim.step
        while self.def_moves:
            c, par = self.def_moves.pop(0)
            if owner[c] == p and army[c] >= 2 and owner[par] == p:
                STATS["defence_moves"] += 1
                return mv(b, c, par)
        return 0

    def act(self, sim):
        self.classify(sim)
        p = self.p
        g, tg = sim.gen[p], sim.gen[1 - p]
        if "defence" not in OFF and g >= 0 and tg >= 0:
            if not any(sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg] for v in self.ctx.b.nbrs[tg]):
                a = self.defend_move(sim)
                if a and sim.legal(p, a):
                    self.opp.update(sim)
                    self.update_gates(sim)
                    _v44.RESERVE = (_v46.RES_HIGH if self.threat_gate else _v46.RES_LOW) + _v46.ARMS[self.arm][0]
                    self.observe(sim)
                    self.econ.kind, self.cwalk = None, None
                    self.why = f"{self.gates()}:fast_defence"
                    return a
        return super().act(sim)


Planner = Planner51
live_ctx = _v46.live_ctx


def act(state, player):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        mem = [int(state.tick), Planner51(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
