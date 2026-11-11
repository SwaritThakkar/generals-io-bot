"""v50: v46 with the algorithms of its source papers implemented as published.

v46 cited its papers for ideas; v50 implements their algorithms. Each block
below names the paper and the algorithm, and is switchable with V50_OFF.

 1. Human.exe army consolidation (Straka, Lisy & Schmid, arXiv 2606.23348,
    2026, sec. VI-A): a ROOTED PRIZE-COLLECTING STEINER TREE (prize = army - 1
    of a tile, cost = one move per tree edge), solved approximately, with a
    BINARY SEARCH OVER THE COST BASIS lambda so the tree fits a turn budget.
    Used for every live gather (gather home, gather to the strike base).
                                                                [off: pcst]
 2. Human.exe expansion: MIN-COST FLOW on a CONTRACTED GRAPH whose nodes are
    contiguous regions of neutral tiles (successive shortest paths,
    Bellman-Ford), sources = our stacks.                       [off: mcf]
 3. Human.exe plan combination: MULTIPLE-CHOICE KNAPSACK DP over the
    candidate paths (one option per stack, shared turn budget). [off: mckp]
 4. Human.exe short engagements: SIMULTANEOUS-MOVE MINIMAX on the local
    matrix game, solved by REGRET MATCHING (Hart & Mas-Colell 2000), whose
    convergence in simultaneous-move games is shown by Lisy, Kovarik, Lanctot
    & Bosansky, NeurIPS 2013.                                  [off: skirmish]
 5. Puppet Search (Barriga, Stanescu & Buro, AIIDE 2015), Algorithm 1
    PUPPET ABCD with alpha-beta, over the macro scripts (our puppet moves) and
    the opponent scripts (theirs), with ABCD's ALTERNATING move-serialization
    policy (Churchill, Saffidine & Buro, AIIDE 2012).           [off: abcd]
 6. Safe exploitation (Ganzfried & Sandholm, ACM TEAC 2015, "Safe Opponent
    Exploitation", the SES "risk what you've won" rule): play the best
    response to the opponent model among plans whose worst-case value is at
    least v* - k, where v* is the safe (Puppet ABCD) value and k the
    accumulated gift; k grows by the realised value above what the safe plan
    guaranteed. The opponent model is DBBR-style (Ganzfried & Sandholm,
    AAMAS 2011): observed behaviour moves the prior.            [off: ses]
 7. Opponent typing:
    a. Synnaeve & Bessiere, AIIDE 2011: Bayesian model
       P(type | X, T) ∝ P(type) prod_e P(X_e, T_e | type) with BELL-SHAPED
       (Gaussian) timing P(T_e | type) per observed event, and the survival
       term for events not yet seen.                            [off: bayes]
    b. Weber & Mateas, CIG 2009: k-NEAREST-NEIGHBOUR strategy prediction on
       event-time feature vectors, with features later than the current
       time set to "not yet seen".                              [off: knn]
    Both are combined with v46's per-move likelihood posterior.
    Parameters (Gaussians, kNN set) come from models/v50_model.json, fitted
    by models/fit_v50.py on the Mac GPU (torch MPS).
 8. Build-order search (Churchill & Buro, AIIDE 2011): DEPTH-FIRST BRANCH AND
    BOUND over castle sequences (macro actions), maximising production to
    the horizon, with an optimistic bound for pruning; the first castle of
    the best sequence is tried first.                           [off: bnb]
 9. Strategy selection (Tavares, Azpurua, Santos & Chaimowicz, AIIDE 2016):
    the parameter arms vs opponent types form a normal-form game with
    empirical payoffs; at game start play its NASH EQUILIBRIUM (solved by
    regret matching), switch to the BEST RESPONSE once the type is
    predicted; across games EXP3 (Auer et al. 2002) when V46_MEMORY is set,
    exactly as in v46.                                          [off: nash]

v46's safety layer (race margin, reserve, survives(), verify()) is unchanged
and still vetoes every move these produce.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import random
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
import core37  # noqa: E402
from core36 import INF, descent, growth, mv, top_stacks  # noqa: E402
from sim36 import Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v46_for_v50", pathlib.Path(core36.__file__).with_name("v46_opponent_learner.py"))
_v46 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v46)
_v44 = _v46._v44
_v42 = _v44._v42
_v39 = _v42._v39
_v37 = _v42._v37

_MEM = {}
#: min-cost-flow expansion (+ its knapsack) measured worse than v46's greedy
#: one-tile-per-move expansion here (6-14 vs v46 alone); kept, off unless V50_ON=mcf
DEFAULT_OFF = {"mcf", "mckp"} - set(filter(None, os.environ.get("V50_ON", "").split(",")))
OFF = set(filter(None, os.environ.get("V50_OFF", "").split(","))) | DEFAULT_OFF
STATS = {k: 0 for k in ("pcst", "mcf", "skirmish", "abcd_exploit", "abcd_safe", "bnb_reorder", "br_switch")}
MODEL_PATH = pathlib.Path(__file__).with_name("v50_model.json")
TYPES = ("rusher", "castle_rusher", "expander", "hold")
EVENTS = ("castle", "attack", "drained", "land30", "stack40")
try:
    MODEL = json.loads(MODEL_PATH.read_text())
except Exception:
    MODEL = None


# =========================================================================== #
# 1. rooted prize-collecting Steiner tree, binary search on the cost basis
# =========================================================================== #
def _pcst_tree(ctx, sim, p, root, lam, cand):
    """Greedy PCST augmentation: grow the tree from the root by the own-land
    path whose collected prize minus lam x (new edges) is largest."""
    owner, army, nbrs = sim.owner, sim.army, ctx.b.nbrs
    parent = {root: None}
    while True:
        # multi-source BFS from the tree over our own land
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


def pcst_plan(ctx, sim, p, root, budget):
    """(post-order move list [(child, parent)], army delivered) of the tree
    with at most `budget` moves; lambda found by binary search."""
    owner, army = sim.owner, sim.army
    cand = [c for c in top_stacks(sim, p, 30, 2) if c != root]
    if not cand:
        return [], 0
    lo, hi = 0.0, float(max(army[c] for c in cand))
    best = None
    for _ in range(7):
        lam = (lo + hi) / 2
        tree = _pcst_tree(ctx, sim, p, root, lam, cand)
        if len(tree) - 1 <= budget:
            best, hi = tree, lam          # fits: try a cheaper basis (bigger tree)
        else:
            lo = lam
    if best is None:
        best = _pcst_tree(ctx, sim, p, root, hi, cand)
        if len(best) - 1 > budget:
            return [], 0
    depth = {}

    def d(c):
        if c not in depth:
            depth[c] = 0 if best[c] is None else d(best[c]) + 1
        return depth[c]

    moves = sorted(((c, best[c]) for c in best if best[c] is not None), key=lambda e: -d(e[0]))
    gathered = sum(max(0, army[c] - 1) for c in best if c != root)
    return moves, gathered


class PCSTGather:
    """Follows a PCST schedule leaf-first, recomputing every 8 ticks or when it breaks."""

    def __init__(self, ctx, p, budget):
        self.ctx, self.p, self.budget = ctx, p, budget
        self.target, self.moves, self.t0 = None, [], -99

    def move(self, sim, target):
        owner, army = sim.owner, sim.army
        if target != self.target or sim.step - self.t0 >= 8 or not self.moves:
            self.moves, _g = pcst_plan(self.ctx, sim, self.p, target, self.budget)
            self.target, self.t0 = target, sim.step
        while self.moves:
            c, par = self.moves.pop(0)
            if owner[c] == self.p and army[c] >= 2 and owner[par] == self.p:
                STATS["pcst"] += 1
                return mv(self.ctx.b, c, par)
        return 0


# =========================================================================== #
# 2. min-cost flow on contracted neutral regions, 3. multiple-choice knapsack
# =========================================================================== #
def regions(ctx, sim, p, max_size=5, limit=15):
    """Contract the neutral land next to our frontier into regions of up to
    max_size contiguous tiles: [(entry, tiles)]."""
    owner, army, b = sim.owner, sim.army, ctx.b
    free = lambda c: owner[c] < 0 and army[c] == 0 and not b.castle[c]  # noqa: E731
    used, out = set(), []
    frontier = [c for c in range(b.n) if free(c) and any(owner[v] == p for v in b.nbrs[c])]
    anchors = top_stacks(sim, p, 3, 2)
    if anchors:
        dists = [ctx.dist(a) for a in anchors]
        frontier.sort(key=lambda c: min(d[c] for d in dists))
    for e in frontier:
        if e in used:
            continue
        tiles, q = [e], deque([e])
        used.add(e)
        while q and len(tiles) < max_size:
            u = q.popleft()
            for v in b.nbrs[u]:
                if v not in used and free(v) and len(tiles) < max_size:
                    used.add(v)
                    tiles.append(v)
                    q.append(v)
        out.append((e, tiles))
        if len(out) >= limit:
            break
    return out


def _bfs_own(ctx, sim, p, src):
    """Distances and predecessors from src over our land, stepping onto one
    non-own tile at the end (the region entry)."""
    owner, nbrs = sim.owner, ctx.b.nbrs
    prods = set(sim.producers)
    dist, prev = {src: 0}, {}
    q = deque([src])
    while q:
        u = q.popleft()
        if owner[u] != p and u != src:
            continue
        for v in nbrs[u]:
            # never route through a producer: the general / castle would pass it back
            if v not in dist and v not in prods:
                dist[v] = dist[u] + 1
                prev[v] = u
                q.append(v)
    return dist, prev


def min_cost_flow(n, edges, s, t):
    """Successive shortest paths (Bellman-Ford); augments while the path
    cost is negative (profitable). edges: [u, v, cap, cost]. Returns flows."""
    graph = [[] for _ in range(n)]
    E = []
    for u, v, cap, cost in edges:
        graph[u].append(len(E)); E.append([u, v, cap, cost, 0])
        graph[v].append(len(E)); E.append([v, u, 0, -cost, 0])
    while True:
        dist = [INF] * n
        pe = [-1] * n
        dist[s] = 0
        for _ in range(n - 1):
            changed = False
            for i, (u, v, cap, cost, f) in enumerate(E):
                if cap - f > 0 and dist[u] < INF and dist[u] + cost < dist[v] - 1e-9:
                    dist[v] = dist[u] + cost
                    pe[v] = i
                    changed = True
            if not changed:
                break
        if dist[t] >= 0 or pe[t] < 0:
            break
        push, v = INF, t
        while v != s:
            i = pe[v]
            push = min(push, E[i][2] - E[i][4])
            v = E[i][0]
        v = t
        while v != s:
            i = pe[v]
            E[i][4] += push
            E[i ^ 1][4] -= push
            v = E[i][0]
    return [(E[i][0], E[i][1], E[i][4]) for i in range(0, len(E), 2)]


def mckp(groups, budget):
    """Multiple-choice knapsack DP: at most one (cost, value, item) per group,
    total cost <= budget, maximum value. Returns the chosen items."""
    best = [(0.0, [])] + [(-INF, [])] * budget
    for opts in groups:
        nxt = list(best)
        for c in range(budget + 1):
            if best[c][0] == -INF:
                continue
            for cost, val, item in opts:
                cc = c + cost
                if cc <= budget and best[c][0] + val > nxt[cc][0]:
                    nxt[cc] = (best[c][0] + val, best[c][1] + [item])
        best = nxt
    return max(best, key=lambda x: x[0])[1]


def productive_move(ctx, sim, p, budget=20, plan=None):
    """Human.exe's productive default: expansion candidates from the min-cost
    flow, combined by the multiple-choice knapsack; returns one move. `plan`
    (a dict) keeps the chosen path until it is walked: the flow is re-solved
    only when the committed path is finished or broken."""
    owner, army = sim.owner, sim.army
    if plan is not None and plan.get("path"):
        path = plan["path"]
        if len(path) >= 2 and owner[path[0]] == p and army[path[0]] >= 2 and \
                (owner[path[1]] == p or (owner[path[1]] < 0 and army[path[1]] == 0)):
            plan["path"] = path[1:]
            return mv(ctx.b, path[0], path[1])
        plan["path"] = None
    prods = set(sim.producers)
    b = ctx.b
    stacks = [c for c in top_stacks(sim, p, 14, 2) if c not in prods][:8]
    border = [c for c in range(b.n) if owner[c] == p and army[c] >= 2 and c not in prods and c not in stacks
              and any(owner[v] < 0 and army[v] == 0 and not b.castle[v] for v in b.nbrs[c])]
    border.sort(key=lambda c: -army[c])
    stacks += border[:10]
    regs = regions(ctx, sim, p)
    if not stacks or not regs:
        return 0
    left = max(1, sim.max_steps - sim.step)
    tile_val = 1.0 + min(3.0, left / 100.0)         # land + its production to the cap, in ticks
    S, T = 0, 1
    sn = {s: 2 + i for i, s in enumerate(stacks)}
    rn = {i: 2 + len(stacks) + i for i in range(len(regs))}
    edges, paths = [], {}
    for s in stacks:
        edges.append([S, sn[s], army[s] - 1, 0.0])
        dist, prev = _bfs_own(ctx, sim, p, s)
        for i, (e, tiles) in enumerate(regs):
            if e not in dist or dist[e] > 12:
                continue
            d = dist[e] - 1                      # moves that capture nothing
            edges.append([sn[s], rn[i], len(tiles), 1.0 + d / len(tiles) - tile_val])
            path, u = [e], e
            while u != s:
                u = prev[u]
                path.append(u)
            paths[(s, i)] = path[::-1]
    for i, (_e, tiles) in enumerate(regs):
        edges.append([rn[i], T, len(tiles), 0.0])
    flows = min_cost_flow(2 + len(stacks) + len(regs), edges, S, T)
    used = {}
    inv_s = {v: k for k, v in sn.items()}
    inv_r = {v: k for k, v in rn.items()}
    for u, v, f in flows:
        if f > 0 and u in inv_s and v in inv_r:
            used[(inv_s[u], inv_r[v])] = f
    if not used:
        return 0
    STATS["mcf"] += 1
    groups = {}
    for (s, i), path in paths.items():
        tiles = min(len(regs[i][1]), army[s] - 1)
        cost = len(path) - 2 + tiles
        bonus = 1.0 if (s, i) in used else 0.85          # flow paths first, alternatives kept
        groups.setdefault(s, []).append((max(1, cost), bonus * tiles * tile_val, (s, path, tiles)))
    for s in groups:
        groups[s] = sorted(groups[s], key=lambda o: -o[1] / o[0])[:3]
    chosen = mckp(list(groups.values()), budget) if "mckp" not in OFF else \
        [max((o for g in groups.values() for o in g), key=lambda o: o[1] / o[0])[2]]
    if not chosen:
        return 0
    # first the chosen item that captures the most land per move it spends
    s, path, tiles = max(chosen, key=lambda it: it[2] / max(1, len(it[1]) - 2 + it[2]))
    if len(path) < 2:
        return 0
    if plan is not None:
        # walk on through the region after its entry, one neutral tile at a time
        region = next(t for e, t in regs if e == path[-1])
        plan["path"] = path[1:] + [v for v in region if v != path[-1]][:max(0, tiles - 1)]
        plan["path"] = _connect(ctx, plan["path"])
    return mv(ctx.b, path[0], path[1])


def _connect(ctx, cells):
    """Keep the longest prefix of cells that is a walk (each step adjacent)."""
    out = cells[:1]
    for c in cells[1:]:
        if c in ctx.b.nbrs[out[-1]]:
            out.append(c)
        else:
            break
    return out


# =========================================================================== #
# 4. simultaneous-move minimax for short engagements, by regret matching
# =========================================================================== #
def regret_matching(M, iters=400):
    """Equilibrium of the zero-sum matrix game M (row player maximises)."""
    m, n = len(M), len(M[0])
    rr, rc = [0.0] * m, [0.0] * n
    sr, sc = [0.0] * m, [0.0] * n

    def strat(r):
        pos = [max(0.0, x) for x in r]
        z = sum(pos)
        return [x / z for x in pos] if z > 0 else [1.0 / len(r)] * len(r)

    for _ in range(iters):
        x, y = strat(rr), strat(rc)
        ur = [sum(M[i][j] * y[j] for j in range(n)) for i in range(m)]
        uc = [-sum(M[i][j] * x[i] for i in range(m)) for j in range(n)]
        vr = sum(x[i] * ur[i] for i in range(m))
        vc = sum(y[j] * uc[j] for j in range(n))
        for i in range(m):
            rr[i] += ur[i] - vr
            sr[i] += x[i]
        for j in range(n):
            rc[j] += uc[j] - vc
            sc[j] += y[j]
    zr, zc = sum(sr), sum(sc)
    return [v / zr for v in sr], [v / zc for v in sc]


def _local_options(ctx, sim, q, cells):
    out = [0]
    for c in cells:
        if sim.owner[c] == q and sim.army[c] >= 2:
            for d in range(4):
                v = ctx.b.step_to[c][d]
                if v >= 0:
                    out.append(c * 9 + 1 + d)
    return out


def skirmish(ctx, sim, p):
    """If our and their stacks are in contact, solve the local simultaneous
    matrix game (2 ticks: the chosen moves, then each side's best capture)."""
    o = 1 - p
    owner, army = sim.owner, sim.army
    ours = top_stacks(sim, p, 6, 5)
    theirs = top_stacks(sim, o, 6, 5)
    contact = []
    for a in ours:
        da = ctx.dist(a)
        for t in theirs:
            if da[t] <= 2:
                contact.append((a, t))
    if not contact:
        return 0
    mine = list(dict.fromkeys(a for a, _t in contact))[:2]
    foes = list(dict.fromkeys(t for _a, t in contact))[:2]
    A = _local_options(ctx, sim, p, mine)
    B = _local_options(ctx, sim, o, foes)
    region = set()
    for c in mine + foes:
        dc = ctx.dist(c)
        region |= {x for x in range(len(owner)) if dc[x] <= 4}

    def score(s2):
        if s2.done:
            return 1000.0 if s2.winner == p else -1000.0
        m = 0.0
        for x in region:
            if s2.owner[x] == p:
                m += s2.army[x] + 0.5
            elif s2.owner[x] == o:
                m -= s2.army[x] + 0.5
        return m

    def greedy(s2, q):
        best, gain = 0, 0
        for c in top_stacks(s2, q, 6, 2):
            for v in ctx.b.nbrs[c]:
                if s2.owner[v] == 1 - q and s2.army[c] - 1 > s2.army[v] and s2.army[v] > gain:
                    best, gain = mv(ctx.b, c, v), s2.army[v]
        return best

    base = score(sim)
    M = []
    for a in A:
        row = []
        for b in B:
            s2 = sim.copy()
            s2.step2(*((a, b) if p == 0 else (b, a)))
            if not s2.done:
                x, y = greedy(s2, p), greedy(s2, o)
                s2.step2(*((x, y) if p == 0 else (y, x)))
            row.append(score(s2) - base)
        M.append(row)
    x, y = regret_matching(M)
    i = max(range(len(A)), key=lambda k: x[k])
    val = lambda k: sum(M[k][j] * y[j] for j in range(len(B)))  # noqa: E731
    if A[i] == 0 or val(i) < val(0) + 2:
        return 0
    STATS["skirmish"] += 1
    return A[i]


# =========================================================================== #
# 5. Puppet ABCD (Barriga 2015, Alg. 1) and 6. SES (Ganzfried & Sandholm 2015)
# =========================================================================== #
def puppet_abcd(V, ours, theirs, h, m1, alpha, beta, first, depth0):
    """Algorithm 1 on a two-level puppet game: our scripts (MAX) and the
    opponent's (MIN); EXEC_CHOICES and EVAL are V(a, b) (a rollout + leaf
    value, memoised). `first` is the serialization policy's player to move."""
    if h == 0:
        return V(*m1), None
    to_move = first if h == depth0 else 1 - first
    best_m = None
    choices = ours if to_move == 0 else theirs
    for m2 in choices:
        if m1 is None:
            v, _ = puppet_abcd(V, ours, theirs, h - 1, (m2,), alpha, beta, first, depth0)
        else:
            pair = (m1[0], m2) if to_move == 1 else (m2, m1[0])
            v, _ = puppet_abcd(V, ours, theirs, h - 1, pair, alpha, beta, first, depth0)
        if to_move == 0 and v > alpha:
            alpha, best_m = v, m2
        if to_move == 1 and v < beta:
            beta, best_m = v, m2
        if alpha >= beta:
            break
    return (alpha if to_move == 0 else beta), best_m


# =========================================================================== #
# 7. opponent typing: Bayesian timing model and kNN
# =========================================================================== #
def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def bayes_timing(events, t):
    """Synnaeve & Bessiere: P(type | events seen by time t), uniform prior."""
    if MODEL is None:
        return {k: 1.0 for k in TYPES}
    out = {}
    for ty in TYPES:
        lp = 0.0
        for e in EVENTS:
            pm = MODEL["timing"][ty][e]
            ph, mu, sd = max(0.02, min(0.98, pm["p"])), pm["mu"], max(15.0, pm["sd"])
            if e in events:
                z = (events[e] - mu) / sd
                lp += math.log(ph) - 0.5 * z * z - math.log(sd)
            else:
                lp += math.log((1 - ph) + ph * (1 - _phi((t - mu) / sd)) + 1e-9)
        out[ty] = lp
    m = max(out.values())
    return {k: math.exp(v - m) for k, v in out.items()}


def knn_type(events, t, k=7):
    """Weber & Mateas: k-NN on event-time vectors, unseen features masked."""
    if MODEL is None:
        return {ty: 1.0 for ty in TYPES}
    cap = MODEL["never"]

    def vec(ev):
        return [ev[e] if e in ev and ev[e] <= t else cap for e in EVENTS]

    q = vec(events)
    rows = []
    for ty, ev in MODEL["knn"]:
        r = vec({e: v for e, v in zip(EVENTS, ev) if v < cap})
        rows.append((sum((a - b) ** 2 for a, b in zip(q, r)), ty))
    rows.sort()
    out = {ty: 0.2 for ty in TYPES}
    for _d, ty in rows[:k]:
        out[ty] += 1.0
    return out


def track_events(ctx, sim, p, ev, prev_gen):
    """Opponent events with the tick they were first seen; returns the new
    prev_gen (their general's army last tick)."""
    o = 1 - p
    t = sim.step
    owner, army = sim.owner, sim.army
    if "castle" not in ev and any(owner[c] == o for c in ctx.castles):
        ev["castle"] = t
    if "land30" not in ev and sum(1 for c in range(len(owner)) if owner[c] == o) >= 30:
        ev["land30"] = t
    tg, g = sim.gen[o], sim.gen[p]
    if "stack40" not in ev and any(c != tg for c in top_stacks(sim, o, 2, 40)):
        ev["stack40"] = t
    if "attack" not in ev and g >= 0:
        dist = ctx.dist(g)
        for s in top_stacks(sim, o, 4, 5):
            if s != tg and dist[s] <= 8 and army[s] >= 0.25 * max(10, army[g]):
                ev["attack"] = t
                break
    if tg >= 0:
        if "drained" not in ev and prev_gen is not None and prev_gen >= 20 and army[tg] < 0.5 * prev_gen:
            ev["drained"] = t
        return army[tg]
    return prev_gen


# =========================================================================== #
# 8. build-order DFS branch and bound over castles (Churchill & Buro 2011)
# =========================================================================== #
def castle_bnb(cands, horizon, depth=3):
    """cands: [(castle, cost_army, ticks)]. Value of a sequence: income 0.5
    per tick from capture to the horizon, minus army spent. Returns the best
    sequence of castles."""
    best = [0.0, []]

    def bound(t, rest):
        return sum(max(0.0, 0.5 * (horizon - t - d) - c) for _k, c, d in rest)

    def dfs(t, val, seq, rest):
        if val > best[0]:
            best[0], best[1] = val, list(seq)
        if len(seq) >= depth or val + bound(t, rest) <= best[0]:
            return
        for i, (k, c, d) in enumerate(rest):
            gain = 0.5 * (horizon - t - d) - c
            if gain <= 0:
                continue
            dfs(t + d, val + gain, seq + [k], rest[:i] + rest[i + 1:])

    dfs(0, 0.0, [], sorted(cands, key=lambda x: -(0.5 * (horizon - x[2]) - x[1])))
    return best[1]


# =========================================================================== #
# planner
# =========================================================================== #
def _nash_arm(seed):
    """Tavares et al.: Nash mix of the arm-vs-type meta-game, sampled per map."""
    if MODEL is None or "nash" in OFF:
        return "base"
    mix = MODEL["nash"]
    r, acc = random.Random(seed).random(), 0.0
    for a, pr in mix.items():
        acc += pr
        if r <= acc:
            return a
    return "base"


class Planner50(_v46.Planner46):
    def __init__(self, ctx, p, arm=None):
        super().__init__(ctx, p, arm or _nash_arm(hash(tuple(ctx.castles)) + p))
        self.exp_plan = {}
        self.move_w = dict(self.weights)
        self.events = {}
        self.prev_gen = None
        self.gift = 0.0
        self.last_safe = None          # (worst-case value guaranteed at the last replan)
        self.serial = 0                # ABCD alternating serialization
        self.gather_home = PCSTGather(ctx, p, 15)
        self.gather_strike = PCSTGather(ctx, p, 25)
        self.br_done = False

    # -- 7. opponent typing --------------------------------------------------
    def track_events(self, sim):
        self.prev_gen = track_events(self.ctx, sim, self.p, self.events, self.prev_gen)

    def observe(self, sim):
        # v37's move likelihood runs on its own weights; the planner sees the
        # product of experts: move likelihood x Bayesian timing x kNN
        self.weights = self.move_w
        super().observe(sim)
        self.move_w = dict(self.weights)
        bt = bayes_timing(self.events, sim.step) if "bayes" not in OFF else {k: 1.0 for k in TYPES}
        kn = knn_type(self.events, sim.step) if "knn" not in OFF else {k: 1.0 for k in TYPES}
        zk = sum(kn.values())
        comb = {k: (self.move_w[k] ** 0.5) * bt[k] * (kn[k] / zk) ** 0.5 for k in TYPES}
        z = sum(comb.values()) or 1.0
        self.weights = {k: v / z * len(TYPES) for k, v in comb.items()}

    def update_gates(self, sim):
        super().update_gates(sim)
        # 9. best response once the opponent type is predicted
        if "nash" in OFF or MODEL is None or self.br_done or sim.step < 150:
            return
        tot = sum(self.weights.values()) or 1.0
        ty, pr = max(((k, v / tot) for k, v in self.weights.items()), key=lambda x: x[1])
        if pr >= 0.8:
            br = MODEL["best_response"].get(ty)
            if br and br != self.arm and _v46._BANDIT is None:
                self.arm = br
                STATS["br_switch"] += 1
            self.br_done = True

    # -- 8. castle build order -------------------------------------------------
    def castle_options(self, sim):
        opts = super().castle_options(sim)
        if "bnb" in OFF or len(opts) < 2:
            return opts
        best_by_castle = {}
        for o in opts:
            c = o[2][-1]
            if c not in best_by_castle:
                need = sim.army[c] + len(o[3]) - 1
                best_by_castle[c] = (c, need, len(o[3]) - 1)
        horizon = min(250, sim.max_steps - sim.step)
        seq = castle_bnb(list(best_by_castle.values())[:6], horizon)
        if not seq:
            return []
        if seq[0] != opts[0][2][-1]:
            STATS["bnb_reorder"] += 1
        return [o for o in opts if o[2][-1] == seq[0]] + [o for o in opts if o[2][-1] != seq[0]]

    # -- 5 + 6. plan choice ----------------------------------------------------
    def replan(self, sim, phase):
        if "abcd" in OFF:
            super().replan(sim, phase)
            self.wrap_macro()
            return
        ctx, p = self.ctx, self.p
        cands = _v39.plan_set39(ctx, sim, p, phase, self.margin, self.econ.pad)
        models = core37.opponent_models(ctx, 1 - p)
        horizon = min(core37.HORIZON[phase], sim.max_steps - sim.step)
        facs = dict(cands)
        names = list(facs)
        cache = {}

        def V(a, b):
            if (a, b) not in cache:
                leaf = core37.rollout(sim, p, facs[a], models[b], horizon)
                cache[(a, b)] = _v37._VALUE(leaf, p, phase)
            return cache[(a, b)]

        # SES gift: realised value now minus what the last safe choice guaranteed
        now = _v37._VALUE(sim, p, phase)
        if self.last_safe is not None:
            self.gift = max(0.0, min(0.15, self.gift + now - self.last_safe))
        first = self.serial % 2
        self.serial += 1
        v_star, m = puppet_abcd(V, names, list(models), 2, None, -INF, INF, first, 2)
        if first == 1:
            # MIN was serialized first: m is their script; our move is the best reply to it
            safe = max(names, key=lambda a: V(a, m)) if m is not None else names[0]
        else:
            safe = m if m is not None else names[0]
        choice = safe
        if "ses" not in OFF:
            tot = sum(self.weights.values()) or 1.0
            best_br = -INF
            for a in names:
                worst = min(V(a, b) for b in models)
                if worst < v_star - self.gift - 1e-9:
                    continue
                br = sum(self.weights[b] * V(a, b) for b in models) / tot
                if br > best_br:
                    best_br, choice = br, a
        STATS["abcd_exploit" if choice != safe else "abcd_safe"] += 1
        self.last_safe = min(V(choice, b) for b in models)
        self.macro, self.macro_name = facs[choice](), choice
        self.until = sim.step + core37.COMMIT[phase]
        self.wrap_macro()

    def wrap_macro(self):
        """Live macro uses Human.exe's gather (PCST) and expansion (MCF + MCKP)."""
        e = getattr(self.macro, "econ", None)
        if e is None:
            return
        ctx, p = self.ctx, self.p
        orig_gather, orig_exp = e.gather, e.expansion
        home, strike = self.gather_home, self.gather_strike
        self.exp_plan = {}

        def gather(sim, target):
            if "pcst" not in OFF:
                pc = home if target == sim.gen[p] else strike
                a = pc.move(sim, target)
                if a:
                    return a
            return orig_gather(sim, target)

        def expansion(sim, allow_producer):
            if not allow_producer and "mcf" not in OFF:
                a = productive_move(ctx, sim, p)
                if a and sim.legal(p, a):
                    return a
            return orig_exp(sim, allow_producer)

        e.gather, e.expansion = gather, expansion

    # -- 4. engagements ------------------------------------------------------------
    def act(self, sim):
        self.track_events(sim)
        p = self.p
        if "skirmish" not in OFF and sim.gen[p] >= 0 and sim.gen[1 - p] >= 0 and self.econ.kind != "rush":
            tg = sim.gen[1 - p]
            if not any(sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg] for v in self.ctx.b.nbrs[tg]):
                _thr, m0 = self.econ.margin_of(sim)
                pad = self.econ.pad + self.econ.pad_frac * core36.mobile_top(sim, 1 - p)
                if m0 > pad:
                    a = skirmish(self.ctx, sim, p)
                    if a and sim.legal(p, a):
                        self.econ.checks = 0
                        if self.econ.safe_after(sim, a, m0, pad):
                            self.opp.update(sim)
                            self.update_gates(sim)
                            self.observe(sim)
                            self.why = f"{self.gates()}:skirmish"
                            return a
        return super().act(sim)


Planner = Planner50
live_ctx = _v46.live_ctx


def act(state, player):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        last = _v46._LAST.pop(player, None)
        if last is not None and _v46._BANDIT is not None:
            _v46._BANDIT.update(last[0].arm, _v46._reward(last[1], player))
        arm = _v46._BANDIT.pick() if _v46._BANDIT is not None else None
        mem = [int(state.tick), Planner50(ctx, player, arm)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    _v46._LAST[player] = (mem[1], sim)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
