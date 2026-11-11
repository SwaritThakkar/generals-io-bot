"""v49: v46 + a kill window: attack the moment a win is guaranteed.

Human games against v46 showed it attacks very late: v46 only charges when
ONE stack's walk beats the enemy general (core36 attack_plan), so an army
spread over castles and land never adds up to an attack in its eyes, while
its home race already counts several hostile stacks collected together.

v49 runs the same collection race on the enemy's home, from our side, every
tick (before v46's macro layer):

    attack(T)  = D_us(tg, T)      our walks onto their general, lengths
                                  summing to <= T (one move per tick), each
                                  tile once, their cells at FULL army plus
                                  the land growth they get on the way
    defence(T) = G_o + growth(T) + D_them(tg, T)
                                  their general plus everything they can
                                  collect home in T moves (their walks
                                  through our cells capped at 3: optimistic
                                  for them)

The smallest T with attack(T) > defence(T) + buffer is the kill window. It
is taken only if
  * our home still holds over T + slack with every committed stack treated
    as gone (v44 race profile with `exclude`), and the general keeps the
    v44 reserve when it sends half;
  * a rollout of the plan wins against every core36 reply (rusher, castle
    rusher, pass, and a turtle Econ that gathers home) within T + 6 ticks.
Then the bot commits: the walks run one after another (hits on a general
add up: each one removes its force from the garrison), re-checked every
tick; two failed re-checks in a row or home in danger cancel the plan.

Research: this is the attacker's side of the budgeted collection race of
Straka, Lisy & Schmid 2026 (rooted prize-collecting Steiner tree), checked
by rollouts against a fixed reply portfolio as in Puppet Search (Barriga
et al. 2015) and core36 verify().

Env: V49_OFF=kill (ablation), V49_TMAX (default 70), V49_BUF (default 0.1).
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

import core36  # noqa: E402
from core36 import Rusher, Script, best_walk, growth, mv, route_tree, top_stacks, verify, walk  # noqa: E402
from sim36 import Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v46_for_v49", pathlib.Path(core36.__file__).with_name("v46_opponent_learner.py"))
_v46 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v46)
_v44 = _v46._v44
_v42 = _v44._v42

_MEM = {}
STATS = {"kill_found": 0, "kill_verified": 0, "kill_rejected": 0, "kill_step": 0, "kill_abort": 0}
OFF = set(filter(None, os.environ.get("V49_OFF", "").split(",")))
T_MAX = int(os.environ.get("V49_TMAX", "70"))
BUF = float(os.environ.get("V49_BUF", "0.1"))
SLACK = 3            # our home must hold this many ticks past the last hit
RETRY = 4            # ticks between verify() attempts after a rejection


def pick(items, T, sim, q):
    """deliver_once that also returns the chosen walks, in launch order."""
    used = tot = 0
    cells, srcs, out = set(), set(), []
    owner, army = sim.owner, sim.army
    for it in sorted(items, key=lambda t: -t[1]):
        d, x, s, ps = it[:4]
        if used + d > T or s in cells or (ps & srcs):
            continue
        x -= sum(army[v] - 1 for v in ps & cells if owner[v] == q and army[v] > 1)
        if x <= 0:
            continue
        used += d
        tot += x
        cells |= ps
        srcs.add(s)
        out.append(it)
    return tot, out


def attack_items(ctx, sim, p, tg):
    """Our all-in walks onto their general: (d, x, s, cells, path, half).
    Their cells are paid in full, plus the land growth they get while we walk."""
    o = 1 - p
    g = sim.gen[p]
    owner, t, li = sim.owner, sim.step, sim.land_int
    tree = route_tree(ctx, sim, p, tg, "garrison")
    out = []

    def add(s, x, path, half):
        d = len(path) - 1
        theirs = sum(1 for v in path[1:-1] if owner[v] == o)
        x -= theirs * ((t + d) // li - t // li)
        if x > 0:
            out.append((d, x, s, frozenset(path), path, half))

    for s in top_stacks(sim, p, 10, 3):
        if s == g:
            continue
        x, path = best_walk(ctx, sim, p, s, tg, tree)
        if path is not None and x > 0:
            add(s, x, path, False)
    if g >= 0 and sim.army[g] >= 20:
        _x, path = best_walk(ctx, sim, p, g, tg, tree)
        if path is not None:
            add(g, walk(sim, p, path, start=sim.army[g] // 2 + 1), path, True)
    return out


HS = (4, 8, 12, 16, 24, 32, 45)


def staged_plans(ctx, sim, p, tg, k=3):
    """Collect-then-strike: walks of our land army onto a forward stack s
    for H moves (v42 home walks with s as home: only through our land, each
    tile once), then one all-in walk s -> their general.
    Yields (T, x, sources, walks)."""
    o = 1 - p
    g = sim.gen[p]
    owner, army, t, li = sim.owner, sim.army, sim.step, sim.land_int
    tree = route_tree(ctx, sim, p, tg, "garrison")
    dist_t = ctx.dist(tg)
    cand = [s for s in top_stacks(sim, p, 12, 3) if s != g and dist_t[s] < core36.INF]
    cand.sort(key=lambda s: dist_t[s] - 0.5 * army[s])
    for s in cand[:k]:
        _x, path = best_walk(ctx, sim, p, s, tg, tree)
        if path is None:
            continue
        d = len(path) - 1
        feeders = [it for it in _v42._items_us(ctx, sim, p, s) if it[2] != g]
        theirs = sum(1 for v in path[1:-1] if owner[v] == o)
        for H in HS:
            mass, chosen = pick(feeders, H, sim, p)
            if not chosen:
                continue
            used = sum(it[0] for it in chosen)
            start = army[s] + mass + ((t + used) // li - t // li)
            x = walk(sim, p, path, start=start) - theirs * ((t + used + d) // li - t // li)
            if x <= 0:
                continue
            walks = [(descent_path(ctx, sim, p, it[2], s), False) for it in chosen] + [(path, False)]
            if any(w is None for w, _h in walks):
                continue
            yield used + d, x, {it[2] for it in chosen} | {s}, walks


def descent_path(ctx, sim, p, a, b):
    return core36.descent(ctx, sim, p, a, b)


class KillScript(Script):
    """Run the committed walks one after another, then keep hitting."""

    name = "kill"

    def __init__(self, ctx, p, walks):
        super().__init__(ctx, p)
        self.queue = [(list(path), half) for path, half in walks]
        self.fallback = Rusher(ctx, p, extra=-10 ** 6)

    def act(self, sim):
        p, b = self.p, self.ctx.b
        tg = sim.gen[1 - p]
        if tg >= 0:
            for v in b.nbrs[tg]:
                if sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg]:
                    return mv(b, v, tg)
        while self.queue:
            path, half = self.queue[0]
            if len(path) >= 2 and sim.owner[path[0]] == p and sim.army[path[0]] >= 2:
                self.queue[0] = (path[1:], False)
                return mv(b, path[0], path[1], half=half)
            self.queue.pop(0)
        return self.fallback.act(sim)


class Planner49(_v46.Planner46):
    def __init__(self, ctx, p, arm="base"):
        super().__init__(ctx, p, arm)
        self.kill = None          # committed walks [(path, half)]
        self.kill_misses = 0
        self.next_verify = -1
        self.kill_info = None

    # -- the kill window ---------------------------------------------------
    def kill_plan(self, sim, pad):
        """(T, surplus, walks) of the earliest guaranteed hit, or None."""
        p, o = self.p, 1 - self.p
        ctx = self.ctx
        g, tg = sim.gen[p], sim.gen[o]
        dfn_items = _v42._items_them(ctx, sim, o, tg)
        G_o = sim.army[tg]

        def defence(T):
            return G_o + growth(sim, sim.step, T) + _v44.deliver_once(dfn_items, T - 1, sim, o)

        def beats(att, T):
            dfn = defence(T)
            return att > dfn + 2 + BUF * dfn, att - dfn

        best = None
        items = attack_items(ctx, sim, p, tg)
        if items and sum(it[1] for it in items) > G_o:
            for T in range(min(it[0] for it in items), T_MAX + 1):
                att, chosen = pick(items, T, sim, p)
                ok, s = beats(att, T)
                if ok and self.home_holds(sim, [(it[2], it[5]) for it in chosen], T, pad):
                    best = (T, s, [(it[4], it[5]) for it in chosen])
                    break
        if "staged" not in OFF:
            for T, x, srcs, walks in staged_plans(ctx, sim, p, tg):
                if T > T_MAX or (best is not None and T >= best[0]):
                    continue
                ok, s = beats(x, T)
                if ok and self.home_holds(sim, [(c, False) for c in srcs], T, pad):
                    best = (T, s, walks)
        return best

    def home_holds(self, sim, chosen, T, pad):
        p = self.p
        g = sim.gen[p]
        s2, srcs = sim, set()
        for src, half in chosen:
            if half:
                # the general sends half: judge home on the halved garrison
                if not _v44.reserve_ok(self.ctx, sim, p, sim.army[g] - sim.army[g] // 2):
                    return False
                s2 = sim.copy()
                s2.army[g] -= sim.army[g] // 2
            else:
                srcs.add(src)
        _thr, prof = _v44.race_profile(self.ctx, s2, p, t_max=T + SLACK, exclude=srcs)
        return not prof or min(m for _T, m in prof) > pad

    def still_winning(self, sim):
        """The remaining walks, chained, still beat their general now plus
        its growth until the last one lands (each hit removes its force)."""
        p, tg = self.p, sim.gen[1 - self.p]
        left, force, ticks, merged = 0, 0, 0, {}
        for path, half in self.kill:
            if len(path) < 2:
                continue
            ticks += len(path) - 1
            if path[-1] == tg:
                start = merged.pop(path[0], 0) + sim.army[path[0]]
                if half:
                    start = sim.army[path[0]] // 2 + 1
                x = walk(sim, p, path, start=start)
                force += max(0, x)
            else:
                # a feeder: what it brings to the staging stack
                x = walk(sim, p, path)
                merged[path[-1]] = merged.get(path[-1], 0) + max(0, x)
            left += 1
        return left and force > sim.army[tg] + growth(sim, sim.step, ticks) + 1

    def kill_step(self, sim, m0, pad):
        p = self.p
        if m0 <= pad:
            if self.kill:
                STATS["kill_abort"] += 1
            self.kill = None
            return 0
        if self.kill and not self.still_winning(sim):
            STATS["kill_abort"] += 1
            self.kill = None
        if not self.kill:
            if sim.step < self.next_verify:
                return 0
            plan = self.kill_plan(sim, pad)
            if plan is None:
                return 0
            T, surplus, walks = plan
            STATS["kill_found"] += 1
            if not verify(self.ctx, sim, p, lambda: KillScript(self.ctx, p, walks), T + 6):
                STATS["kill_rejected"] += 1
                self.next_verify = sim.step + RETRY
                return 0
            STATS["kill_verified"] += 1
            self.kill_info = (T, round(surplus))
            # committed: the rollouts already answered their replies, so the
            # walks run to the end unless home is in danger or the hit is lost
            self.kill = [(list(path), half) for path, half in walks]
        while self.kill:
            path, half = self.kill[0]
            if len(path) >= 2 and sim.owner[path[0]] == p and sim.army[path[0]] >= 2:
                a = mv(self.ctx.b, path[0], path[1], half=half)
                if sim.legal(p, a):
                    self.kill[0] = (path[1:], False)
                    STATS["kill_step"] += 1
                    return a
            self.kill.pop(0)
        self.kill = None
        return 0

    def act(self, sim):
        p = self.p
        g, tg = sim.gen[p], sim.gen[1 - p]
        if "kill" in OFF or g < 0 or tg < 0:
            return super().act(sim)
        b = self.ctx.b
        for v in b.nbrs[tg]:
            if sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg]:
                return super().act(sim)        # v42 takes the capture
        econ = self.econ
        _thr, m0 = econ.margin_of(sim)
        pad = econ.pad + econ.pad_frac * core36.mobile_top(sim, 1 - p)
        a = self.kill_step(sim, m0, pad)
        if a:
            self.opp.update(sim)
            self.update_gates(sim)
            self.observe(sim)
            self.margin = m0
            T, s = self.kill_info or (0, 0)
            self.why = f"{self.gates()}:kill(T={T},+{s})"
            return a
        return super().act(sim)


Planner = Planner49
live_ctx = _v46.live_ctx


def act(state, player):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        mem = [int(state.tick), Planner49(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
