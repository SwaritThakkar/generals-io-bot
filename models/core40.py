"""v40: paper-guided planner pieces.

* EconLite: v36/v37-like opponent model without nested safety rollouts.
* PGS+ style rollouts (Lelis 2017): the candidate plan acts only for the
  commit interval K, then a default policy plays the rest of the horizon,
  so "hold"/"gather" plans are not judged as if held forever.
* Regret matching over the plan x opponent-reply matrix (Lisy et al. 2013:
  Hannan-consistent selection converges in simultaneous-move games; UCB
  does not). Replaces v37's 0.5*mean + 0.5*worst blend.
* Board planes + plan features for the learned ranker (Barriga et al. 2017
  style policy over abstract actions), trained on the Mac GPU (MPS).
* Asymmetric tactical step (Moraes & Lelis 2018): on contact, all raw moves
  of up to N=4 critical stacks against likely simultaneous replies; kept
  only if it beats the plan's own move under the same evaluation.
"""
from __future__ import annotations

import math

import numpy as np

import core36
import core37
from core36 import INF, Econ, mv, threats, top_stacks
from adversaries38 import GatherRusher

PLAN_TYPES = ("expand", "hold", "castle", "gather_home", "strike", "rush")
N_PLANES = 14


class EconLite(Econ):
    """v36/v37-like economy and defence, no nested rollouts (cheap)."""

    swap_safe = True

    def __init__(self, ctx, p):
        super().__init__(ctx, p, aggression=10 ** 9)

    def safe_after(self, sim, a, m0, pad):
        if not a:
            return True
        cell = a // 9
        if m0 >= INF and sim.army[cell] < 8 and cell != sim.gen[self.p]:
            return True
        s2 = self.after(sim, a)
        if s2.done:
            return s2.winner == self.p
        _thr, m1 = self.margin_of(s2)
        return m1 > pad or (cell != sim.gen[self.p] and m1 >= m0)


def replies(ctx, o):
    """Opponent plan set: v37's four models + v36-like economy + the
    human-style gather rush that beat v37 in recorded games."""
    m = dict(core37.opponent_models(ctx, o))
    m["econ"] = lambda: EconLite(ctx, o)
    m["gather_rush"] = lambda: GatherRusher(ctx, o, t0=0)
    return m


def plan_type(name):
    for t in PLAN_TYPES:
        if name.startswith(t):
            return t
    return "expand"


# --------------------------------------------------------------------------- #
# rollouts and the matrix game
# --------------------------------------------------------------------------- #
class Then(core36.Script):
    """Run `first` for k ticks, then `default` (PGS+ evaluation)."""

    def __init__(self, ctx, p, first, default, k):
        super().__init__(ctx, p)
        self.first, self.default, self.k = first, default, k
        self.t = 0

    def act(self, sim):
        self.t += 1
        return self.first.act(sim) if self.t <= self.k else self.default.act(sim)


def rollout(sim, p, ours, theirs, horizon):
    s2 = sim.copy()
    a_s, o_s = ours(), theirs()
    for _ in range(horizon):
        if s2.done:
            break
        a = a_s.act(s2)
        b = o_s.act(s2)
        a = a if a and a > 0 else 0
        b = b if b and b > 0 else 0
        if p == 0:
            s2.step2(a, b)
        else:
            s2.step2(b, a)
    return s2


def regret_matching(M, iters=400, gamma=0.05):
    """Row player's average strategy, column average strategy and value of
    the zero-sum matrix game M (row maximises)."""
    M = np.asarray(M, float)
    m, n = M.shape
    rr, rc = np.zeros(m), np.zeros(n)
    sr, sc = np.zeros(m), np.zeros(n)
    for _ in range(iters):
        pr = np.maximum(rr, 0); pr = pr / pr.sum() if pr.sum() > 0 else np.full(m, 1 / m)
        pc = np.maximum(rc, 0); pc = pc / pc.sum() if pc.sum() > 0 else np.full(n, 1 / n)
        pr = (1 - gamma) * pr + gamma / m
        pc = (1 - gamma) * pc + gamma / n
        u_rows = M @ pc
        u_cols = pr @ M
        v = pr @ u_rows
        rr += u_rows - v
        rc += v - u_cols            # column player minimises
        sr += pr; sc += pc
    sr /= sr.sum(); sc /= sc.sum()
    return sr, sc, float(sr @ M @ sc)


def plan_matrix(ctx, sim, p, phase, value, cands, models, horizon, k):
    """Exact simultaneous rollouts: rows = our plans (PGS+ style), cols = replies."""
    o_default = None
    M = np.zeros((len(cands), len(models)))
    names = list(models)
    for i, (name, fac) in enumerate(cands):
        ours = (lambda fac=fac: Then(ctx, p, fac(), EconLite(ctx, p), k))
        for j, oname in enumerate(names):
            leaf = rollout(sim, p, ours, models[oname], horizon)
            M[i, j] = value(leaf, p, phase)
    return M, names


# --------------------------------------------------------------------------- #
# ranker inputs
# --------------------------------------------------------------------------- #
def planes(ctx, sim, p):
    """(N_PLANES, H, W) float32 board encoding from p's perspective."""
    h, w = ctx.b.h, ctx.b.w
    owner = np.asarray(sim.owner).reshape(h, w)
    army = np.asarray(sim.army, np.float32).reshape(h, w)
    o = 1 - p
    la = np.log1p(army) / 6.0
    castle = np.asarray(ctx.b.castle).reshape(h, w)
    mountain = np.asarray(ctx.b.mountain).reshape(h, w)
    gen = np.zeros((2, h, w), np.float32)
    for q in (0, 1):
        if sim.gen[q] >= 0:
            gen[q].flat[sim.gen[q]] = 1.0
    dg = np.asarray(ctx.dist(sim.gen[p]), np.float32).reshape(h, w) if sim.gen[p] >= 0 else np.full((h, w), 99.0)
    do = np.asarray(ctx.dist(sim.gen[o]), np.float32).reshape(h, w) if sim.gen[o] >= 0 else np.full((h, w), 99.0)
    t = sim.step
    out = np.stack([
        np.where(owner == p, la, 0), np.where(owner == o, la, 0), np.where(owner < 0, la, 0),
        (owner == p), (owner == o), castle, gen[p], gen[o], mountain,
        np.minimum(dg, 60) / 40.0, np.minimum(do, 60) / 40.0,
        np.full((h, w), (t % sim.land_int) / sim.land_int),     # tempo (Straka et al.)
        np.full((h, w), max(0, sim.max_steps - t) / 1000.0),
        np.full((h, w), t / 1000.0),
    ]).astype(np.float32)
    return out


def plan_vec(ctx, sim, p, name, m0):
    t = plan_type(name)
    onehot = [1.0 if t == u else 0.0 for u in PLAN_TYPES]
    cf = [0.0, 0.0, 0.0, 0.0]
    if t == "castle":
        c = int(name[len("castle"):])
        g, og = sim.gen[p], sim.gen[1 - p]
        cf = [sim.army[c] / 50.0, ctx.dist(c)[g] / 20.0,
              (ctx.dist(c)[og] / 20.0) if og >= 0 else 1.0,
              1.0 if sim.owner[c] == 1 - p else 0.0]
    mm = max(-2.0, min(2.0, (m0 if m0 < INF else 100) / 50.0))
    ph = core37.detect_phase(ctx, sim, p, m0)
    return onehot + cf + [mm] + [1.0 if ph == u else 0.0 for u in core37.PHASES]


PLAN_DIM = len(PLAN_TYPES) + 4 + 1 + 3


def margin(ctx, sim, p):
    thr = threats(ctx, sim, p)
    return core36.home_margin(ctx, sim, p, thr=thr, reinf=False) if thr else INF


# --------------------------------------------------------------------------- #
# asymmetric tactical step
# --------------------------------------------------------------------------- #
def critical_cells(sim, p, n=4):
    """Our stacks that touch an enemy stack, our general, or their general."""
    o = 1 - p
    b = sim.b
    owner, army = sim.owner, sim.army
    hot = set()
    for c in range(b.n):
        if owner[c] != p or army[c] < 2:
            continue
        for v in b.nbrs[c]:
            if (owner[v] == o and army[v] >= 2) or v == sim.gen[o]:
                hot.add(c)
                break
    g = sim.gen[p]
    if g >= 0 and any(owner[v] == o and army[v] >= 2 for v in b.nbrs[g]):
        hot.add(g)
    return sorted(hot, key=lambda c: -army[c])[:n]


def raw_moves(sim, cells):
    out = []
    for c in cells:
        for d in range(4):
            if sim.b.step_to[c][d] >= 0:
                out.append(c * 9 + 1 + d)
                out.append(c * 9 + 5 + d)
    return out


def enemy_replies(sim, p, limit=8):
    """Plausible simultaneous enemy moves: pass, plus every move of enemy
    stacks adjacent to our stacks/general (incl. into cells we vacate)."""
    o = 1 - p
    return [0] + raw_moves(sim, critical_cells(sim, o, 3))[:limit]
