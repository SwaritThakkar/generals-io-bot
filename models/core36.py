"""Shared model for the v36+ candidates: routes, threat invariant, macro scripts.

Everything here works on `sim36.Sim`, so the same script code drives the live
bot and both players inside simultaneous rollouts.

Model (see RESEARCH_IDEAS.md, "v36"):

* One action per player per tick, so a plan is "which single stack moves".
* An all-in walk from s to t is simulated exactly on the current board:
  every departure leaves one unit, own cells are picked up, neutral/hostile
  cells are paid for. `walk()` returns the force that reaches t.
* The defender's general grows while the attacker walks:
  growth(t, d) = producer ticks + land ticks in (t, t+d].
* Safety invariant: for every hostile stack, its walk arrival must stay below
  our general after growth plus one reinforcement we could bring in time.
  The invariant decides how much the general may send; it never forbids
  expansion with other stacks.
"""
from __future__ import annotations

import heapq
from collections import deque

from sim36 import Board, Sim  # noqa: F401

INF = 10 ** 9


# --------------------------------------------------------------------------- #
# static per-match context
# --------------------------------------------------------------------------- #
class Ctx:
    """Static distance maps, built lazily, shared by every rollout of a match."""

    def __init__(self, board):
        self.b = board
        self._dist = {}
        self.castles = [i for i in range(board.n) if board.castle[i]]

    def dist(self, target):
        """BFS steps to `target` over passable cells."""
        d = self._dist.get(target)
        if d is not None:
            return d
        b = self.b
        d = [INF] * b.n
        d[target] = 0
        q = deque([target])
        while q:
            u = q.popleft()
            du = d[u] + 1
            for v in b.nbrs[u]:
                if d[v] == INF:
                    d[v] = du
                    q.append(v)
        self._dist[target] = d
        return d


_CTX = {}


def ctx_for(board_key, board_factory):
    c = _CTX.get(board_key)
    if c is None:
        if len(_CTX) > 8:
            _CTX.clear()
        c = Ctx(board_factory())
        _CTX[board_key] = c
    return c


def route_tree(ctx, sim, att, target, mode="garrison", cap=None):
    """Reverse Dijkstra to `target` for attacker `att` on the live board.

    Edge cost of stepping onto cell v is 1 (the unit left behind) plus the
    neutral/hostile army on v; own cells are free (pickups are counted exactly
    by `walk`, but are not subtracted here so costs stay non-negative).
    mode "income" adds 0.5 per step for the defender's production while the
    walk is under way (Track A income-aware edge cost); "steps" is plain BFS.
    Returns (cost, next_hop) lists.
    """
    b = ctx.b
    owner, army = sim.owner, sim.army
    cost = [INF] * b.n
    nxt = [-1] * b.n
    cost[target] = 0
    heap = [(0, target)]
    extra = 0.5 if mode == "income" else 0.0
    steps = mode == "steps"
    while heap:
        cu, u = heapq.heappop(heap)
        if cu > cost[u]:
            continue
        # entering u from v costs: 1 + hostile(u), except the target itself,
        # whose garrison is compared separately
        if steps:
            w = 1
        else:
            w = 1 + extra
            if u != target and owner[u] != att:
                ou = owner[u]
                w += army[u] if (cap is None or ou < 0) else min(army[u], cap)
        cv = cu + w
        for v in b.nbrs[u]:
            if cv < cost[v]:
                cost[v] = cv
                nxt[v] = u
                heapq.heappush(heap, (cv, v))
    return cost, nxt


def tree_path(nxt, s, target):
    path = [s]
    u = s
    guard = 0
    while u != target:
        u = nxt[u]
        if u < 0 or guard > 2000:
            return None
        path.append(u)
        guard += 1
    return path


# --------------------------------------------------------------------------- #
# exact static walks
# --------------------------------------------------------------------------- #
def growth(sim, t0, d, cell_is_producer=True):
    """Units a producer (or plain owned cell) gains in ticks (t0, t0+d]."""
    g = 0
    if cell_is_producer:
        g += (t0 + d) // sim.gil - t0 // sim.gil
    g += (t0 + d) // sim.land_int - t0 // sim.land_int
    return g


def descent(ctx, sim, p, s, t):
    """Shortest-step path s -> t (list of cells incl. both ends), choosing the
    cheapest next cell for player p among equal-distance options."""
    dist = ctx.dist(t)
    if dist[s] >= INF:
        return None
    owner, army, nbrs = sim.owner, sim.army, ctx.b.nbrs
    path = [s]
    u = s
    while u != t:
        want = dist[u] - 1
        best, best_c = -1, INF
        for v in nbrs[u]:
            if dist[v] == want:
                c = -army[v] if owner[v] == p else army[v]
                if v == t:
                    c = -INF
                if c < best_c:
                    best, best_c = v, c
        if best < 0:
            return None
        path.append(best)
        u = best
    return path


def walk(sim, p, path, start=None, cap=None):
    """Force that reaches path[-1] if player p walks all-in along `path` now.

    `cap` limits what a defender-owned cell is assumed to hold: mobile
    defender stacks may leave the path, land tiles stay. Returns -1 if the
    stack dies or cannot move before the end."""
    owner, army = sim.owner, sim.army
    x = army[path[0]] if start is None else start
    for v in path[1:-1]:
        if x < 2:
            return -1
        x -= 1
        o = owner[v]
        if o == p:
            x += army[v]
        else:
            x -= army[v] if (cap is None or o < 0) else min(army[v], cap)
            if x <= 0:
                return -1
    if x < 2:
        return -1
    return x - 1


# --------------------------------------------------------------------------- #
# threat model and reserve
# --------------------------------------------------------------------------- #
def top_stacks(sim, p, k=4, min_army=3):
    owner, army = sim.owner, sim.army
    cells = [c for c in range(len(owner)) if owner[c] == p and army[c] >= min_army]
    cells.sort(key=lambda c: -army[c])
    return cells[:k]


ROUTE_MODE = {"mode": "garrison"}
#: what a defender cell is assumed to hold in threat walks (land, not stacks)
STATIC_CAP = 3


def best_walk(ctx, sim, att, s, target, tree=None, cap=None):
    """(arrival, path) of the better of the cheapest-cost and shortest walks."""
    cands = []
    if tree is not None:
        path = tree_path(tree[1], s, target)
        if path is not None:
            cands.append(path)
    path = descent(ctx, sim, att, s, target)
    if path is not None:
        cands.append(path)
    best, best_x = None, -1
    for path in cands:
        if len(path) < 2:
            continue
        x = walk(sim, att, path, cap=cap)
        # a longer path also gives the defender more growth
        score = x - growth(sim, sim.step, len(path) - 1)
        if best is None or score > best_x:
            best, best_x = path, score
    if best is None:
        return -1, None
    return walk(sim, att, best, cap=cap), best


def threats(ctx, sim, defender, k=4, tree=None):
    """[(arrival_force, steps, source, path)] of hostile all-in walks on our general."""
    g = sim.gen[defender]
    att = 1 - defender
    out = []
    if g < 0:
        return out
    if tree is None and ROUTE_MODE["mode"] != "steps":
        tree = route_tree(ctx, sim, att, g, ROUTE_MODE["mode"], cap=STATIC_CAP)
    dist = ctx.dist(g)
    stacks = top_stacks(sim, att, 12, 3)
    stacks.sort(key=lambda c: -(sim.army[c] - dist[c]))
    for s in stacks[:k]:
        x, path = best_walk(ctx, sim, att, s, g, tree, cap=STATIC_CAP)
        if x > 0:
            out.append((x, len(path) - 1, s, path))
    return out


def reinforcement(ctx, sim, p, within, exclude=()):
    """Largest force one of our non-general stacks could walk home in `within` ticks."""
    g = sim.gen[p]
    dist = ctx.dist(g)
    best = 0
    owner, army = sim.owner, sim.army
    for s in top_stacks(sim, p, 6):
        if s == g or s in exclude or dist[s] > within:
            continue
        path = descent(ctx, sim, p, s, g)
        if path is None:
            continue
        x = walk(sim, p, path)
        if x > best:
            best = x
    return best


def home_margin(ctx, sim, p, general_army=None, thr=None, reinf=True):
    """min over threats of (defence at arrival - attack arrival). >0 is safe."""
    g = sim.gen[p]
    G = sim.army[g] if general_army is None else general_army
    thr = threats(ctx, sim, p) if thr is None else thr
    m = INF
    for x, d, s, _path in thr:
        r = reinforcement(ctx, sim, p, d - 1) if reinf else 0
        m = min(m, G + growth(sim, sim.step, d) + r - x)
    return m


# --------------------------------------------------------------------------- #
# action helpers
# --------------------------------------------------------------------------- #
def mv(b, src, dst, half=False):
    d = b.direction(src, dst)
    if d < 0:
        return 0
    return src * 9 + 1 + d + (4 if half else 0)


def sent_after(sim, action):
    cell, k = divmod(action, 9)
    a = sim.army[cell]
    return a // 2 if k >= 5 else a - 1


# --------------------------------------------------------------------------- #
# scripts
# --------------------------------------------------------------------------- #
class Script:
    """Deterministic macro policy. One instance per player per simulation."""

    name = "script"

    def __init__(self, ctx, p):
        self.ctx, self.p = ctx, p
        self.path = None
        self.kind = None

    def follow(self, sim, need=0):
        """Step the stored path if it is still ours and still strong enough."""
        path = self.path
        if not path or len(path) < 2:
            self.path = None
            return 0
        s = path[0]
        if sim.owner[s] != self.p or sim.army[s] < 2:
            self.path = None
            return 0
        if need:
            x = walk(sim, self.p, path)
            if x <= need(sim, len(path) - 1, path[-1]):
                self.path = None
                return 0
        self.path = path[1:]
        return mv(self.ctx.b, s, path[1])

    def act(self, sim):
        return 0


def need_general(p):
    def f(sim, d, t):
        return sim.army[t] + growth(sim, sim.step, d)
    return f


def need_cell(p):
    def f(sim, d, t):
        o = sim.owner[t]
        if o == p:
            return -INF
        return sim.army[t] + (growth(sim, sim.step, d, sim.b.castle[t]) if o >= 0 else 0)
    return f


class Pass(Script):
    name = "hold"

    def act(self, sim):
        return 0


class Rusher(Script):
    """A common human stress strategy: bank at the general, then all-in the
    opposing general by the shortest path when the walk would beat its
    current garrison; replan every tick (like models/general_rusher.py)."""

    name = "rusher"

    def __init__(self, ctx, p, extra=0):
        super().__init__(ctx, p)
        self.extra = extra

    def act(self, sim):
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        best, best_x = None, 0
        for s in top_stacks(sim, p, 3):
            path = descent(self.ctx, sim, p, s, tg)
            if path is None:
                continue
            x = walk(sim, p, path)
            if x > sim.army[tg] + self.extra and x > best_x:
                best, best_x = path, x
        if best is None:
            return 0
        return mv(self.ctx.b, best[0], best[1])


class CastleRusher(Script):
    """Bank, take the nearest castle when affordable, then rush the general."""

    name = "castle_rusher"

    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.rush = Rusher(ctx, p)

    def act(self, sim):
        a = self.rush.act(sim)
        if a:
            return a
        p = self.p
        g = sim.gen[p]
        if self.path:
            a = self.follow(sim, need_cell(p))
            if a:
                return a
        best, best_d = None, INF
        for c in self.ctx.castles:
            if sim.owner[c] == p:
                continue
            d = self.ctx.dist(c)[g]
            if d < best_d:
                best, best_d = c, d
        if best is None:
            return 0
        path = descent(self.ctx, sim, p, g, best)
        if path is None:
            return 0
        x = walk(sim, p, path)
        if x > sim.army[best]:
            self.path = path
            return self.follow(sim)
        return 0


class PathRush(Script):
    """Follow a fixed all-in path, then keep hitting the general."""

    name = "pathrush"

    def __init__(self, ctx, p, path):
        super().__init__(ctx, p)
        self.path = list(path)
        self.fallback = Rusher(ctx, p, extra=-10 ** 6)

    def act(self, sim):
        a = self.follow(sim)
        return a or self.fallback.act(sim)


def mobile_top(sim, p):
    """Largest stack of p that is not banked on a producer. Banked generals
    grow predictably (the margin trend covers them); mobile stacks can merge."""
    prods = set(sim.producers)
    for c in top_stacks(sim, p, 6, 1):
        if c not in prods:
            return sim.army[c]
    return 0


def merge_potential(ctx, sim, att, thr, radius=10):
    """Extra force the strongest threat can absorb from nearby hostile stacks
    before launch: max over other stacks c of army[c] - dist(c, s)."""
    if not thr:
        return 0
    x, _d, s, _path = max(thr, key=lambda t: t[0])
    dist = ctx.dist(s)
    best = 0
    for c in top_stacks(sim, att, 6, 3):
        if c != s and dist[c] <= radius:
            best = max(best, sim.army[c] - dist[c])
    return best


class Defender(Script):
    """Cheap home defence used as our continuation inside safety rollouts:
    hit an adjacent attacker we can beat, else pull the stack that reaches
    home soonest with the most force, else wait (the general grows)."""

    name = "defender"

    def __init__(self, ctx, p, swap_safe=False):
        super().__init__(ctx, p)
        #: v39+: never step the general toward an adjacent enemy stack. Moves
        #: are simultaneous, so the two stacks can pass each other and the
        #: attacker lands on the emptied general (seen in v37 losses).
        self.swap_safe = swap_safe

    def act(self, sim):
        p = self.p
        g = sim.gen[p]
        if g < 0:
            return 0
        owner, army, b = sim.owner, sim.army, self.ctx.b
        o = 1 - p
        dist = self.ctx.dist(g)
        for s in top_stacks(sim, o, 2, 3):
            for v in b.nbrs[s]:
                if self.swap_safe and v == g:
                    continue
                if owner[v] == p and army[v] - 1 > army[s] and dist[s] <= dist[v] + 1:
                    return mv(b, v, s)
        best, best_x = None, 1
        for s in top_stacks(sim, p, 6, 3):
            if s == g or dist[s] > 30:
                continue
            path = descent(self.ctx, sim, p, s, g)
            if path is None:
                continue
            x = walk(sim, p, path) - 0.5 * len(path)
            if x > best_x:
                best, best_x = path, x
        if best is not None:
            return mv(b, best[0], best[1])
        return 0


def launch_paths(ctx, sim, att, target, k=3):
    """Plausible all-in routes of the attacker's largest stacks."""
    out = []
    tree = route_tree(ctx, sim, att, target)
    dist = ctx.dist(target)
    # rank by force left after the shortest walk, not by raw size
    stacks = top_stacks(sim, att, 10, 3)
    stacks.sort(key=lambda c: -(sim.army[c] - dist[c]))
    for s in stacks[:k]:
        seen = set()
        for path in (descent(ctx, sim, att, s, target), tree_path(tree[1], s, target)):
            if path is None or len(path) < 2 or tuple(path) in seen:
                continue
            seen.add(tuple(path))
            if walk(sim, att, path) > 0:
                out.append(path)
    return out


def survives(ctx, sim, p, first):
    """Play `first` now, then Defender, against every immediate all-in launch."""
    o = 1 - p
    g = sim.gen[p]
    if g < 0:
        return False
    after = sim.copy()
    if p == 0:
        after.step2(first, 0)
    else:
        after.step2(0, first)
    if after.done:
        return after.winner == p
    paths, seen = [], set()
    # routes are judged on the board our move creates (a captured castle
    # can open a short cut) and on the current board
    for path in launch_paths(ctx, after, o, g) + launch_paths(ctx, sim, o, g):
        if tuple(path) not in seen:
            seen.add(tuple(path))
            paths.append(path)
    for path in paths:
        s2 = sim.copy()
        att = PathRush(ctx, o, path)
        dfn = Defender(ctx, p)
        a, b = first, att.act(s2)
        for _ in range(len(path) + 4):
            if p == 0:
                s2.step2(a, b)
            else:
                s2.step2(b, a)
            if s2.done:
                break
            a, b = dfn.act(s2), att.act(s2)
        if s2.done and s2.winner != p:
            return False
    return True


def responses(ctx, p):
    """Opponent replies used to verify a committed plan for player 1-p."""
    o = 1 - p
    return [lambda: Rusher(ctx, o), lambda: CastleRusher(ctx, o),
            lambda: Pass(ctx, o), lambda: Econ(ctx, o, turtle=True)]


def verify(ctx, sim, p, ours, horizon):
    """Play `ours` (a fresh script factory) against every reply for
    `horizon` ticks; True if we never lose and win against all."""
    for reply in responses(ctx, p):
        s2 = sim.copy()
        a_s, o_s = ours(), reply()
        for _ in range(horizon):
            if s2.done:
                break
            a = a_s.act(s2)
            b = o_s.act(s2)
            if p == 0:
                s2.step2(a, b)
            else:
                s2.step2(b, a)
        if s2.winner != p:
            return False
    return True


class Econ(Script):
    """Economy policy built on the standalone-general invariant.

    Fundamentals (RESEARCH_IDEAS.md, "v36"):
      * the general alone must beat every hostile all-in walk after the
        growth it gets during that walk (margin m0 > pad);
      * moves that do not lower m0 are always allowed, so expansion with
        other stacks never freezes;
      * a drained enemy general is attacked by any stack whose exact walk
        wins the race.

    Parameters give macro postures for the search:
      aggression   surplus demanded before an all-in general attack
      castle_wait  max ticks to bank at a producer for a castle
      pad          base units kept above the strongest hostile walk
      expand       whether the general feeds expansion
      rush_all     consolidate home instead of expanding
      turtle       only gather home and hold
    """

    name = "econ"

    def __init__(self, ctx, p, aggression=2, castle_wait=20, pad=3,
                 expand=True, rush_all=False, turtle=False, label=None,
                 stack_castles=False, anchor=False, finisher=True, pad_frac=0.08):
        super().__init__(ctx, p)
        self.pad_frac = pad_frac
        self.use_stack_castles = stack_castles
        self.use_anchor = anchor
        self.use_finisher = finisher
        self.aggression = aggression
        self.castle_wait = castle_wait
        self.pad = pad
        self.expand = expand
        self.rush_all = rush_all
        self.turtle = turtle
        if label:
            self.name = label
        self.hist = deque(maxlen=24)
        self.corridor = set()
        self.last_gather = None
        self.sticky = None
        self.anchor = None
        self.why = "none"
        self.margin = INF

    # -- invariant -----------------------------------------------------------
    def dyn_pad(self, sim, m0):
        """Pad plus lead time when the margin shrinks over time (an opponent
        banking faster than our general grows). The base pad scales with the
        largest hostile stack, so late-game stacks of hundreds are not held
        to a three-unit tolerance."""
        if m0 >= INF:
            self.hist.clear()
            return self.pad
        base = self.pad + self.pad_frac * mobile_top(sim, 1 - self.p)
        self.hist.append((sim.step, m0))
        t_old, m_old = self.hist[0]
        if sim.step - t_old < 8:
            return base
        slope = (m0 - m_old) / (sim.step - t_old)
        return base + max(0.0, -slope * 16.0)

    def margin_of(self, sim):
        thr = threats(self.ctx, sim, self.p)
        if not thr:
            return thr, INF
        m = home_margin(self.ctx, sim, self.p, thr=thr, reinf=False)
        return thr, m - merge_potential(self.ctx, sim, 1 - self.p, thr)

    def margin_r(self, sim, thr=None):
        thr = threats(self.ctx, sim, self.p) if thr is None else thr
        if not thr:
            return INF
        return home_margin(self.ctx, sim, self.p, thr=thr, reinf=True)

    def launched(self, sim, thr):
        """A threatening stack that has left its producer is on the way."""
        prods = set(sim.producers)
        return any(s not in prods and x > 0 for x, _d, s, _p in thr)

    def after(self, sim, a):
        s2 = sim.copy()
        if self.p == 0:
            s2.step2(a, 0)
        else:
            s2.step2(0, a)
        return s2

    def safe_after(self, sim, a, m0, pad):
        if not a:
            if m0 > pad:
                return True
            self.checks += 1
            return survives(self.ctx, sim, self.p, 0)
        cell = a // 9
        if m0 >= INF and sim.army[cell] < 8 and cell != sim.gen[self.p]:
            return True
        s2 = self.after(sim, a)
        if s2.done:
            return s2.winner == self.p
        thr1, m1 = self.margin_of(s2)
        if m1 > pad:
            return True
        if cell != sim.gen[self.p] and m1 >= m0 and (cell, a) == self.sticky:
            return True
        if cell == sim.gen[self.p]:
            # the general never relies on being refilled later
            return False
        self.checks += 1
        return survives(self.ctx, sim, self.p, a)

    # -- offence -----------------------------------------------------------
    def attack_plan(self, sim, margin):
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        if tg < 0:
            return None, -INF
        best, best_m = None, -INF
        tree = None
        if ROUTE_MODE["mode"] != "steps":
            tree = route_tree(self.ctx, sim, p, tg, ROUTE_MODE["mode"])
        dist_o = self.ctx.dist(tg)
        for s in top_stacks(sim, p, 12, 3):
            if sim.army[s] - dist_o[s] <= sim.army[tg]:
                continue
            x, path = best_walk(self.ctx, sim, p, s, tg, tree)
            if path is None or x <= 0:
                continue
            d = len(path) - 1
            defence = sim.army[tg] + growth(sim, sim.step, d)
            defence += reinforcement(self.ctx, sim, o, d - 1)
            m = x - defence
            if m <= margin or m <= best_m:
                continue
            if not self.race_safe(sim, path, d):
                continue
            ok = verify(self.ctx, sim, p, lambda: PathRush(self.ctx, p, path), d + 4)
            if not ok:
                continue
            best, best_m = path, m
        return best, best_m

    def race_safe(self, sim, path, d):
        """Pessimistic race: hostile stacks take their BFS-shortest time and
        their strongest walk force, and dodge our attacking stack."""
        p = self.p
        g = sim.gen[p]
        a = mv(self.ctx.b, path[0], path[1])
        s2 = self.after(sim, a)
        # the attacker routes around our marching stack: treat it as gone
        if s2.owner[path[1]] == p:
            s2.army[path[1]] = 1
        G = s2.army[g]
        dist = self.ctx.dist(g)
        for x, _td, s, _tp in threats(self.ctx, s2, p, k=6):
            t_fast = dist[s]
            if t_fast <= d + 1 and x > G + growth(s2, s2.step, t_fast):
                return False
        return True

    # -- defence -----------------------------------------------------------
    def defend(self, sim, thr):
        p = self.p
        g = sim.gen[p]
        dist = self.ctx.dist(g)
        owner, army, b = sim.owner, sim.army, self.ctx.b
        for x, d, s, path in sorted(thr, key=lambda t: -t[0]):
            for v in b.nbrs[s]:
                if getattr(self, "swap_safe", False) and v == g:
                    continue
                if owner[v] == p and army[v] - 1 > army[s]:
                    return mv(b, v, s)
        horizon = min(t[1] for t in thr) if thr else INF
        best, best_x = None, 0
        for s in top_stacks(sim, p, 8, 2):
            if s == g or dist[s] >= horizon:
                continue
            path = descent(self.ctx, sim, p, s, g)
            if path is None:
                continue
            x = walk(sim, p, path)
            if x > best_x:
                best, best_x = path, x
        if best is not None:
            return mv(b, best[0], best[1])
        return 0

    # -- economy -----------------------------------------------------------
    def castle_choice(self, sim):
        p = self.p
        best = (INF, None, None)
        sources = [c for c in self.ctx.castles if sim.owner[c] == p]
        if sim.gen[p] >= 0:
            sources.append(sim.gen[p])
        if self.use_stack_castles:
            sources += [c for c in top_stacks(sim, p, 3, 20) if c not in sources]
        for c in self.ctx.castles:
            if sim.owner[c] == p:
                continue
            dist = self.ctx.dist(c)
            for s in sources:
                if dist[s] >= 30:
                    continue
                path = descent(self.ctx, sim, p, s, c)
                if path is None:
                    continue
                d = len(path) - 1
                x = walk(sim, p, path)
                need = sim.army[c] + (growth(sim, sim.step, d) if sim.owner[c] >= 0 else 0)
                deficit = need + 1 - x
                wait = 0 if deficit <= 0 else 2 * deficit + 2
                if sim.owner[c] >= 0:
                    wait += 2
                eta = wait + d
                if eta < best[0]:
                    best = (eta, path, s)
        return best

    def expansion(self, sim, allow_producer):
        p = self.p
        owner, army, b = sim.owner, sim.army, self.ctx.b
        corridor = self.corridor
        castle = b.castle
        best, best_s = 0, -INF
        prod = set(c for c in sim.producers if owner[c] == p)
        o = 1 - p
        dist_o = self.ctx.dist(sim.gen[o]) if sim.gen[o] >= 0 else None
        for c in range(b.n):
            if owner[c] != p or army[c] < 2:
                continue
            if c in prod and not allow_producer:
                continue
            for v in b.nbrs[c]:
                if owner[v] == p or castle[v]:
                    continue
                if owner[v] < 0 and army[v] == 0:
                    gain = 1.0
                elif owner[v] == o and army[v] < army[c] - 1:
                    gain = 2.0
                else:
                    continue
                score = gain * 10 + army[c] * 0.1
                score += sum(1 for w in b.nbrs[v] if owner[w] < 0 and not castle[w]) * 0.5
                if dist_o is not None and dist_o[v] < INF:
                    score -= dist_o[v] * 0.05
                if c in prod:
                    score -= 5
                if v in corridor:
                    score += 4
                if score > best_s:
                    best, best_s = mv(b, c, v), score
        return best

    def to_frontier(self, sim):
        p = self.p
        owner, army, b = sim.owner, sim.army, self.ctx.b
        prod = set(c for c in sim.producers if owner[c] == p)
        for s in top_stacks(sim, p, 6, 3):
            if s in prod:
                continue
            seen = {s: None}
            q = deque([s])
            hit = None
            while q and hit is None:
                u = q.popleft()
                for v in b.nbrs[u]:
                    if v in seen or b.castle[v]:
                        continue
                    seen[v] = u
                    if owner[v] != p:
                        if owner[v] < 0 and army[v] == 0:
                            hit = v
                            break
                        continue
                    q.append(v)
            if hit is None:
                continue
            v = hit
            while seen[v] != s:
                v = seen[v]
            return mv(b, s, v)
        return 0

    def general_out(self, sim):
        """Send the general's stack one step toward the nearest neutral land."""
        p = self.p
        g = sim.gen[p]
        owner, army, b = sim.owner, sim.army, self.ctx.b
        if army[g] < 3:
            return 0
        seen = {g: None}
        q = deque([g])
        while q:
            u = q.popleft()
            for v in b.nbrs[u]:
                if v in seen or b.castle[v]:
                    continue
                seen[v] = u
                if owner[v] != p:
                    if owner[v] < 0 and army[v] == 0:
                        while seen[v] != g:
                            v = seen[v]
                        return mv(b, g, v, half=army[g] > 12)
                    continue
                q.append(v)
        return 0

    def gather(self, sim, target):
        p = self.p
        owner, army, b = sim.owner, sim.army, self.ctx.b
        dist = self.ctx.dist(target)
        # keep walking the stack we moved last tick until it arrives
        last = self.last_gather
        if last is not None and last[0] == target:
            c = last[1]
            if owner[c] == p and army[c] >= 2 and c != target:
                for v in b.nbrs[c]:
                    if dist[v] == dist[c] - 1:
                        a = mv(b, c, v)
                        self.sticky = (c, a)
                        self.last_gather = (target, v)
                        return a
        best, best_s = 0, -INF
        for c in range(b.n):
            if owner[c] != p or army[c] < 2 or c == target or dist[c] >= INF:
                continue
            for v in b.nbrs[c]:
                if dist[v] == dist[c] - 1 and owner[v] == p:
                    score = army[c] - 0.01 * dist[c]
                    if score > best_s:
                        best, best_s = mv(b, c, v), score
                        best_v = v
                    break
        if best:
            self.last_gather = (target, best_v)
            self.sticky = (best // 9, best)
        return best

    def finishing(self, sim):
        """Clear lead: stop widening, start concentrating for the kill."""
        land, tot = sim.totals()
        p, o = self.p, 1 - self.p
        return (land[p] >= 2 * land[o] + 20 and tot[p] >= 2 * tot[o] + 40
                and self.strike_base(sim) is not None)

    def strike_base(self, sim):
        """Our producer (not the general) closest to the enemy general."""
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        if tg < 0:
            return None
        dist = self.ctx.dist(tg)
        best, best_d = None, INF
        for c in self.ctx.castles:
            if sim.owner[c] == p and c != sim.gen[p] and dist[c] < best_d:
                best, best_d = c, dist[c]
        if best is not None or not self.use_anchor:
            return best
        # no forward castle: a stable own cell near the enemy general, a few
        # steps back from contact so the stack is not traded on arrival
        anchor = self.anchor
        if anchor is not None and sim.owner[anchor[0]] == p and sim.step - anchor[1] < 25:
            return anchor[0]
        g = sim.gen[p]
        dg = self.ctx.dist(g)
        best, best_s = None, INF
        for c in range(len(sim.owner)):
            if sim.owner[c] != p or c == g or dist[c] >= INF:
                continue
            sc = dist[c] + 0.15 * dg[c]
            if dist[c] >= 4 and sc < best_s:
                best, best_s = c, sc
        self.anchor = (best, sim.step) if best is not None else None
        return best

    # -- policy ------------------------------------------------------------
    def candidates(self, sim, thr, m0, pad):
        p = self.p
        g = sim.gen[p]
        G = sim.army[g]

        path, _m = self.attack_plan(sim, self.aggression)
        if path is not None:
            self.kind, self.path = "rush", path
            a = self.follow(sim)
            if a:
                yield "rush", a, False

        if m0 <= pad and self.launched(sim, thr):
            self.path, self.kind = None, None
            yield "defend", self.defend(sim, thr), True
            yield "defender", Defender(self.ctx, p, getattr(self, "swap_safe", False)).act(sim), True
            yield "expand_tight", self.expansion(sim, allow_producer=False), True
            yield "hold", 0, True
            return

        if m0 <= pad:
            # not launched yet, but the general alone would fall: consolidate
            # home now, while there is still time to walk stacks in
            yield "consolidate_home", self.gather(sim, g), True

        if self.turtle:
            yield "turtle", self.gather(sim, g), False
            yield "hold", 0, True
            return

        if self.kind == "castle" and self.path:
            a = self.follow(sim, need_cell(p))
            if a:
                yield "castle_walk", a, True
            self.kind, self.path = None, None

        eta, cpath, src = self.castle_choice(sim)
        waiting_at = None
        if cpath is not None:
            wait_needed = eta - (len(cpath) - 1)
            if wait_needed <= 0:
                self.kind, self.path = "castle", cpath
                a = self.follow(sim)
                if a:
                    yield "castle", a, True
                self.kind, self.path = None, None
            elif wait_needed <= self.castle_wait:
                waiting_at = src

        if self.rush_all:
            yield "consolidate", self.gather(sim, g), True

        base = self.strike_base(sim)
        if self.use_finisher and self.finishing(sim):
            yield "gather_finish", self.gather(sim, base), True

        yield "expand", self.expansion(sim, allow_producer=False), True
        yield "frontier", self.to_frontier(sim), True
        base = self.strike_base(sim)
        if base is not None and m0 > pad + 10:
            yield "gather_strike", self.gather(sim, base), True
        if waiting_at is not None:
            yield "gather_wait", self.gather(sim, waiting_at), True
            yield "wait", 0, True
            return
        if self.expand and G >= 2:
            a = self.expansion(sim, allow_producer=True)
            if a and a // 9 == g and G > 12:
                yield "general_expand_half", a + 4, True
            yield "general_expand", a, True
            yield "general_frontier", self.general_out(sim), True
        yield "gather_home", self.gather(sim, g), True

    def act(self, sim):
        p, o = self.p, 1 - self.p
        g = sim.gen[p]
        self.why = "none"
        if g < 0:
            return 0
        b = self.ctx.b
        owner, army = sim.owner, sim.army
        tg = sim.gen[o]

        if tg >= 0:
            for v in b.nbrs[tg]:
                if owner[v] == p and army[v] - 1 > army[tg]:
                    self.why = "capture"
                    return mv(b, v, tg)

        if self.kind == "rush" and self.path:
            a = self.follow(sim, need_general(p))
            if a:
                self.why = "rush_walk"
                return a
            self.kind = None

        thr, m0 = self.margin_of(sim)
        self.corridor = set(c for t in thr for c in t[3])
        pad = self.dyn_pad(sim, m0)
        self.margin = m0
        self.checks = 0
        for why, a, check in self.candidates(sim, thr, m0, pad):
            if not a and why not in ("hold", "wait"):
                continue
            if self.checks >= 6:
                break
            if check and not self.safe_after(sim, a, m0, pad):
                if why.startswith("castle"):
                    self.kind, self.path = None, None
                continue
            self.why = why
            return a
        self.why = "fallback_defender"
        return Defender(self.ctx, p, getattr(self, "swap_safe", False)).act(sim)
