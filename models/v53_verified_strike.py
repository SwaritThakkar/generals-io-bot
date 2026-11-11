"""v53: v51 + intercepts with non-general stacks, and every attack verified
against every reply.

Problem (human games vs v51): when the opponent parks stacks next to our
general, v51 only feeds the general (fast defence / fallback Defender) and
never hits those stacks with the army it has elsewhere. And only moves OUT
OF THE GENERAL were safety-gated as attacks: the planner's `strike` and
`rush` plans charged the enemy general from any other stack after a bare
size check (walk > garrison + growth), with no look at the replies.

1. Intercept. Enemy stacks (not their general) within INTERCEPT_R of our
   general are targets; our stacks other than the general are attackers.
   A strike s -> e is launched only if, against EVERY reply script
   (stay, flee, reinforce the target, charge our general with it, hit back
   at our stack, all-in rusher), the rollout (our stack chases the target,
   then the Defender) never loses the general, never ends with our race
   margin under the pad when standing still would not, and either removes
   the target (enemy army near home ends at least half the target lower
   than if we had stood still) or the reply was a retreat that cost us
   nothing. The strike is re-verified every tick and dropped as soon as it
   no longer wins.
2. Every charge at the enemy general, whatever stack it comes from (plan
   `strike` / `rush`, v36 `rush_walk`, v44 counter), is re-verified each
   tick with verify(): the walk must capture against all four replies
   (rusher, castle rusher, pass, turtle). Otherwise it is not played.
3. Every contact move into an enemy stack (>= CONTACT_MIN army) or the
   enemy general, from any of our cells, is checked against every legal
   reply of the enemy cells within 2 steps (full and half moves, plus
   pass): after the simultaneous step we must hold the target cell and the
   game must not be lost.
A vetoed move is replaced by v36's rule list, then the Defender, then pass,
each passing the same checks.
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import pathlib
import sys
import time

for _root in [pathlib.Path(__file__).resolve().parent, *map(pathlib.Path, sys.path)]:
    for _cand in (_root, _root / "models"):
        if (_cand / "core37.py").exists():
            sys.path.insert(0, str(_cand))
            break
    else:
        continue
    break

import core36  # noqa: E402
from core36 import (INF, Defender, Econ, PathRush, Pass, Rusher, Script, descent,  # noqa: E402
                    mv, top_stacks, verify, walk)
from sim36 import Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v51_for_v53", pathlib.Path(core36.__file__).with_name("v51_fast_defence.py"))
_v51 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v51)
_v46 = _v51._v46
_v44 = _v51._v44

_v44.verify = None        # replaced below (counter-strike uses the cheap verify)
_MEM = {}
OFF = set(filter(None, os.environ.get("V53_OFF", "").split(",")))
STATS = {"intercept": 0, "veto_charge": 0, "veto_contact": 0, "intercept_checks": 0}
INTERCEPT_R = 8         # enemy stacks this close to our general are targets
CHASE_R = 11            # stop chasing once the target is this far from home
MAX_WALK = 10
CONTACT_MIN = 3
CHARGE_TAGS = ("rush", "strike", "rush_walk", "counter")
INTERCEPT_BUDGET = 0.12  # seconds per tick for intercept checks
ATTACK_VERIFY, ATTACK_BUDGET = 3, 0.15


# --------------------------------------------------------------------------- #
# rollout scripts for intercept verification
# --------------------------------------------------------------------------- #
def _near_target(sim, o, pos, ctx):
    """Biggest enemy cell within 2 steps of `pos` (follows a stack that moved)."""
    best, best_a = None, 1
    for c in [pos] + list(ctx.b.nbrs[pos]) + [w for v in ctx.b.nbrs[pos] for w in ctx.b.nbrs[v]]:
        if sim.owner[c] == o and c != sim.gen[o] and sim.army[c] > best_a:
            best, best_a = c, sim.army[c]
    return best


class Chase(Script):
    """Our attacker walks at the (moving) target, then defends."""

    def __init__(self, ctx, p, src, target):
        super().__init__(ctx, p)
        self.cur, self.tgt = src, target
        self.dfn = Defender(ctx, p, swap_safe=True)
        self.done = False

    def act(self, sim):
        p, o = self.p, 1 - self.p
        if not self.done:
            cur = self.cur
            t = _near_target(sim, o, self.tgt, self.ctx) if self.tgt is not None else None
            g = sim.gen[p]
            if (t is None or sim.owner[cur] != p or sim.army[cur] < 2
                    or self.ctx.dist(g)[t] > CHASE_R):
                self.done = True
            else:
                path = descent(self.ctx, sim, p, cur, t)
                if path is None or len(path) < 2 or g in path:
                    self.done = True
                else:
                    self.cur, self.tgt = path[1], t
                    return mv(self.ctx.b, cur, path[1])
        return self.dfn.act(sim)


class _Reply(Script):
    """Opponent reply anchored on the target stack (tracked as it moves)."""

    def __init__(self, ctx, p, target, chase=None):
        super().__init__(ctx, p)
        self.tgt, self.chase = target, chase

    def track(self, sim):
        if self.tgt is None:
            return None
        self.tgt = _near_target(sim, self.p, self.tgt, self.ctx)
        return self.tgt


class Flee(_Reply):
    def act(self, sim):
        t = self.track(sim)
        if t is None:
            return 0
        tg = sim.gen[self.p]
        path = descent(self.ctx, sim, self.p, t, tg) if tg >= 0 else None
        if path and len(path) >= 2:
            self.tgt = path[1]
            return mv(self.ctx.b, t, path[1])
        return 0


class Reinforce(_Reply):
    def __init__(self, ctx, p, target, chase=None):
        super().__init__(ctx, p, target, chase)
        self.econ = Econ(ctx, p)

    def act(self, sim):
        t = self.track(sim)
        return self.econ.gather(sim, t) if t is not None else 0


class ChargeHome(_Reply):
    """The target goes all-in at our general (we may have left it thin)."""

    def act(self, sim):
        t = self.track(sim)
        g = sim.gen[1 - self.p]
        if t is None or g < 0:
            return 0
        path = descent(self.ctx, sim, self.p, t, g)
        if path and len(path) >= 2:
            self.tgt = path[1]
            return mv(self.ctx.b, t, path[1])
        return 0


class HitBack(_Reply):
    """The target walks at our attacking stack."""

    def act(self, sim):
        t = self.track(sim)
        if t is None or self.chase is None:
            return 0
        c = self.chase.cur
        if sim.owner[c] != 1 - self.p:
            return 0
        path = descent(self.ctx, sim, self.p, t, c)
        if path and len(path) >= 2:
            self.tgt = path[1]
            return mv(self.ctx.b, t, path[1])
        return 0


REPLIES = {
    "stay": lambda ctx, o, e, ch: Pass(ctx, o),
    "flee": lambda ctx, o, e, ch: Flee(ctx, o, e, ch),
    "reinforce": lambda ctx, o, e, ch: Reinforce(ctx, o, e, ch),
    "charge": lambda ctx, o, e, ch: ChargeHome(ctx, o, e, ch),
    "hit_back": lambda ctx, o, e, ch: HitBack(ctx, o, e, ch),
    "rusher": lambda ctx, o, e, ch: Rusher(ctx, o),
}


def _run(sim, p, ours, theirs, horizon):
    s2 = sim.copy()
    for _ in range(horizon):
        if s2.done:
            break
        a, b = ours.act(s2), theirs.act(s2)
        a = a if a and a > 0 else 0
        b = b if b and b > 0 else 0
        s2.step2(*((a, b) if p == 0 else (b, a)))
    return s2


def _home(ctx, sim, p):
    """(enemy army near our general, our army near our general)."""
    g = sim.gen[p]
    if g < 0:
        return INF, 0
    dist = ctx.dist(g)
    o = 1 - p
    them = us = 0
    for c in range(len(sim.owner)):
        if dist[c] <= INTERCEPT_R + 2:
            if sim.owner[c] == o and c != sim.gen[o]:
                them += sim.army[c]
            elif sim.owner[c] == p:
                us += sim.army[c]
    return them, us


def strike_ok(ctx, sim, p, src, target, econ, pad):
    """True if the strike src -> target wins against every reply."""
    o = 1 - p
    a_e = sim.army[target]
    d = ctx.dist(target)[src]
    horizon = d + 5
    for name, rep in REPLIES.items():
        ch = Chase(ctx, p, src, target)
        att = _run(sim, p, ch, rep(ctx, o, target, ch), horizon)
        if att.done and att.winner != p:
            return False
        if att.done:
            continue
        base = _run(sim, p, Defender(ctx, p, swap_safe=True), rep(ctx, o, target, None), horizon)
        if base.done and base.winner != p:
            # standing still loses too: the strike must at least survive (checked above)
            continue
        them_a, us_a = _home(ctx, att, p)
        them_b, us_b = _home(ctx, base, p)
        them_0, _us0 = _home(ctx, sim, p)
        removed = them_a <= them_b - 0.5 * a_e
        retreat = them_b <= them_0 - 0.5 * a_e and us_a >= us_b - 0.3 * sim.army[src]
        if not (removed or retreat):
            return False
        _t, m_a = econ.margin_of(att)
        if m_a <= pad:
            _t, m_b = econ.margin_of(base)
            if m_a < m_b:
                return False
    return True


# --------------------------------------------------------------------------- #
# one-step contact check and charge check
# --------------------------------------------------------------------------- #
def contact_ok(ctx, sim, p, a):
    """A move into an enemy stack / general holds the cell against every
    legal reply of the enemy cells within 2 steps."""
    o = 1 - p
    cell, k = divmod(a, 9)
    dst = ctx.b.step_to[cell][(k - 1) % 4]
    near = set([dst, cell])
    for v in list(near):
        for w in ctx.b.nbrs[v]:
            near.add(w)
            near.update(ctx.b.nbrs[w])
    replies = [0]
    for c in near:
        if sim.owner[c] == o and sim.army[c] >= 2:
            for kk in range(1, 9):
                b = c * 9 + kk
                if sim.legal(o, b):
                    replies.append(b)
    for b in replies:
        s2 = sim.copy()
        s2.step2(*((a, b) if p == 0 else (b, a)))
        if s2.done:
            if s2.winner != p:
                return False
            continue
        if s2.owner[dst] != p:
            return False
    return True


class Turtle(Script):
    """Cheap turtle reply: pull the nearest-best stack into the general every
    tick (core36's turtle Econ runs its whole rule list, attack search and
    safety rollouts inside every simulated tick)."""

    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.econ = Econ(ctx, p)

    def act(self, sim):
        g = sim.gen[self.p]
        return self.econ.gather(sim, g) if g >= 0 else 0


def verify53(ctx, sim, p, ours, horizon):
    """core36.verify with the cheap turtle: our script must capture the
    general against rusher, castle rusher, pass and turtle."""
    o = 1 - p
    for reply in (lambda: Rusher(ctx, o), lambda: core36.CastleRusher(ctx, o),
                  lambda: Pass(ctx, o), lambda: Turtle(ctx, o)):
        s2 = sim.copy()
        a_s, o_s = ours(), reply()
        for _ in range(horizon):
            if s2.done:
                break
            a, b = a_s.act(s2), o_s.act(s2)
            s2.step2(*((a, b) if p == 0 else (b, a)))
        if s2.winner != p:
            return False
    return True


def charge_ok(ctx, sim, p, src):
    """The stack at src walking all-in at the enemy general captures it
    against every reply (verify(): rusher, castle rusher, pass, turtle)."""
    tg = sim.gen[1 - p]
    path = descent(ctx, sim, p, src, tg) if tg >= 0 else None
    if path is None or len(path) < 2:
        return False
    return verify53(ctx, sim, p, lambda: PathRush(ctx, p, path), len(path) - 1 + 4)


def attack_plan_budget(self, sim, margin):
    """core36 Econ.attack_plan with the same tests, but the cheap margin is
    computed for every stack first and the expensive race / verify() checks
    run best-first on at most ATTACK_VERIFY stacks within ATTACK_BUDGET s
    (the arena keeps ticking while the bot thinks)."""
    t0 = time.perf_counter()
    p, o = self.p, 1 - self.p
    tg = sim.gen[o]
    if tg < 0:
        return None, -INF
    tree = None
    if core36.ROUTE_MODE["mode"] != "steps":
        tree = core36.route_tree(self.ctx, sim, p, tg, core36.ROUTE_MODE["mode"])
    dist_o = self.ctx.dist(tg)
    cands = []
    for s in top_stacks(sim, p, 12, 3):
        if sim.army[s] - dist_o[s] <= sim.army[tg]:
            continue
        x, path = core36.best_walk(self.ctx, sim, p, s, tg, tree)
        if path is None or x <= 0:
            continue
        d = len(path) - 1
        defence = sim.army[tg] + core36.growth(sim, sim.step, d)
        defence += core36.reinforcement(self.ctx, sim, o, d - 1)
        m = x - defence
        if m > margin:
            cands.append((m, path, d))
    cands.sort(key=lambda c: -c[0])
    for m, path, d in cands[:ATTACK_VERIFY]:
        if time.perf_counter() - t0 > ATTACK_BUDGET:
            break
        if not self.race_safe(sim, path, d):
            continue
        if verify53(self.ctx, sim, p, lambda path=path: PathRush(self.ctx, p, path), d + 4):
            return path, m
    return None, -INF


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
class Planner53(_v51.Planner51):
    def __init__(self, ctx, p, arm="base"):
        super().__init__(ctx, p, arm)
        self.strike = None          # (our stack cell, target cell) being executed
        self._charge_cache = {}
        self._inc = (-1, None)
        self.t0 = 0.0
        if "budget" not in OFF:
            import types
            self.econ.attack_plan = types.MethodType(attack_plan_budget, self.econ)

    def _refresh(self, sim):
        """Opponent bookkeeping the parent does on its own paths."""
        self.opp.update(sim)
        self.update_gates(sim)
        _v44.RESERVE = (_v46.RES_HIGH if self.threat_gate else _v46.RES_LOW) + _v46.ARMS[self.arm][0]
        self.observe(sim)

    # -- approach tracking, once per tick ----------------------------------------
    def detect(self, sim):
        if self._inc[0] != sim.step:
            self._inc = (sim.step, super().detect(sim))
        return self._inc[1]

    def under_attack(self, sim):
        """v51's fast defence will act this tick (an attack is incoming and
        the general would be short): defence comes first, no intercepts."""
        inc = self.detect(sim)
        if inc is None:
            return False
        eta, att, _t = inc
        return self.defence_need(sim, eta, att) > 0

    def home_holds_without(self, sim, s, pad):
        """Our race margin stays above the pad with stack s never coming home."""
        _thr, prof = _v44.race_profile(self.ctx, sim, self.p, exclude=(s,))
        return not prof or min(m for _T, m in prof) > pad

    # -- 1. intercept ------------------------------------------------------------
    def intercept(self, sim):
        if "intercept" in OFF:
            return 0
        if self.under_attack(sim):
            self.strike = None
            return 0
        p, o = self.p, 1 - self.p
        ctx = self.ctx
        g = sim.gen[p]
        dist_g = ctx.dist(g)
        targets = [c for c in top_stacks(sim, o, 8, 5) if c != sim.gen[o] and dist_g[c] <= INTERCEPT_R]
        if not targets:
            self.strike = None
            return 0
        thr, m0 = self.econ.margin_of(sim)
        pad = self.econ.pad + self.econ.pad_frac * core36.mobile_top(sim, o)
        # continue the current strike while it still wins
        if self.strike is not None:
            s, t = self.strike
            t2 = _near_target(sim, o, t, ctx)
            if s != g and sim.owner[s] == p and sim.army[s] >= 2 and t2 is not None:
                STATS["intercept_checks"] += 1
                if self.home_holds_without(sim, s, pad) and strike_ok(ctx, sim, p, s, t2, self.econ, pad):
                    path = descent(ctx, sim, p, s, t2)
                    if path and len(path) >= 2 and g not in path:
                        a = mv(ctx.b, s, path[1])
                        if self._contact_pass(sim, a):
                            self.strike = (path[1], t2)
                            return a
            self.strike = None
        attackers = [c for c in top_stacks(sim, p, 10, 3) if c != g]
        cands = []
        for e in targets:
            de = ctx.dist(e)
            for s in attackers:
                if de[s] > MAX_WALK:
                    continue
                path = descent(ctx, sim, p, s, e)
                if path is None or len(path) < 2 or g in path:
                    continue        # never walk through (and out of) the general
                x = walk(sim, p, path)
                if x <= sim.army[e] + 1:
                    continue
                cands.append((-(sim.army[e]), len(path), s, e, path))
        cands.sort(key=lambda t: (t[0], t[1]))
        for _k, _d, s, e, path in cands[:4]:
            if time.perf_counter() - self.t0 > INTERCEPT_BUDGET:
                break
            STATS["intercept_checks"] += 1
            if not self.home_holds_without(sim, s, pad):
                continue
            if not strike_ok(ctx, sim, p, s, e, self.econ, pad):
                continue
            a = mv(ctx.b, s, path[1])
            if not self._contact_pass(sim, a):
                continue
            self.econ.checks = 0
            if not self.econ.safe_after(sim, a, m0, pad):
                continue
            self.strike = (path[1], e)
            STATS["intercept"] += 1
            return a
        return 0

    # -- 2/3. vet every attack ---------------------------------------------------
    def _is_contact(self, sim, a):
        if not a:
            return False
        o = 1 - self.p
        cell, k = divmod(a, 9)
        dst = self.ctx.b.step_to[cell][(k - 1) % 4]
        return dst >= 0 and sim.owner[dst] == o and (sim.army[dst] >= CONTACT_MIN or dst == sim.gen[o])

    def _contact_pass(self, sim, a):
        if "contact" in OFF or not self._is_contact(sim, a):
            return True
        if contact_ok(self.ctx, sim, self.p, a):
            return True
        STATS["veto_contact"] += 1
        return False

    def _is_charge(self, sim, a):
        if not a:
            return False
        tag = str(self.why).split(":")[-1]
        if tag not in CHARGE_TAGS:
            return False
        if "v36:rush" in str(self.why) and not str(self.why).endswith("rush_walk"):
            return False        # launched this tick by attack_plan, which ran verify() on it
        if tag == "strike":
            path = getattr(self.macro, "path", None)
            return bool(path) and path[-1] == sim.gen[1 - self.p]
        tg = sim.gen[1 - self.p]
        if tg < 0:
            return False
        cell, k = divmod(a, 9)
        dst = self.ctx.b.step_to[cell][(k - 1) % 4]
        dist = self.ctx.dist(tg)
        return dist[dst] < dist[cell]

    def _vet(self, sim, a):
        """None if a is an attack that does not win against every reply."""
        if "charge" not in OFF and self._is_charge(sim, a):
            cell = a // 9
            # the stack keeps walking only while the whole walk still wins
            key = (sim.step, cell)
            if key not in self._charge_cache:
                self._charge_cache = {key: charge_ok(self.ctx, sim, self.p, cell)}
            if not self._charge_cache[key]:
                STATS["veto_charge"] += 1
                if self.econ.kind == "rush":
                    self.econ.kind, self.econ.path = None, None
                if getattr(self.macro, "path", None):
                    self.macro.path = None
                return None
        if not self._contact_pass(sim, a):
            return None
        return a

    def act(self, sim):
        self.t0 = time.perf_counter()
        p = self.p
        g, tg = sim.gen[p], sim.gen[1 - p]
        if g < 0:
            return 0
        if tg >= 0:
            for v in self.ctx.b.nbrs[tg]:
                if sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg]:
                    a = mv(self.ctx.b, v, tg)
                    if contact_ok(self.ctx, sim, p, a):
                        self.classify(sim)
                        self._refresh(sim)
                        self.why = "capture"
                        return a
        if tg >= 0:
            a = self.intercept(sim)
            if a and sim.legal(p, a):
                self.classify(sim)
                self._refresh(sim)
                self.econ.kind, self.cwalk = None, None
                self.why = f"{self.gates()}:intercept"
                return a
        a = super().act(sim)
        a = a if a and a > 0 else 0
        if self._vet(sim, a) is not None or not a:
            return a
        why = self.why
        was_charge = "charge" not in OFF and self._is_charge(sim, a)
        # replacement: v36's rule list, then the Defender, then pass
        self.econ.kind, self.econ.path, self.cwalk = None, None, None
        m0 = self.margin
        pad = self.econ.pad + self.econ.pad_frac * core36.mobile_top(sim, 1 - p)
        for name, alt in (("rules", lambda: self.econ.act(sim)),
                          ("expand", lambda: self.econ.expansion(sim, allow_producer=False)),
                          ("frontier", lambda: self.econ.to_frontier(sim)),
                          ("defender", lambda: Defender(self.ctx, p, swap_safe=True).act(sim))):
            if name == "rules" and was_charge:
                continue
            b = alt()
            b = b if b and b > 0 else 0
            if not b or not sim.legal(p, b):
                continue
            self.why = f"veto[{why}]->{name}:{self.econ.why if name == 'rules' else 'defender'}"
            if self._vet(sim, b) is None:
                continue
            self.econ.checks = 0
            if name == "defender" or self.econ.safe_after(sim, b, m0, pad):
                return b
        self.why = f"veto[{why}]->hold"
        return 0


_v44.verify = verify53
Planner = Planner53
live_ctx = _v51.live_ctx


def act(state, player):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        mem = [int(state.tick), Planner53(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
