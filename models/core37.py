"""v37 phase planner: macro plans, phase detection, features, lookahead.

Algorithm (RESEARCH_IDEAS.md, "v37"):

* Branching reduction. A decision chooses among ~6 macro plans instead of
  ~2,000 raw moves; plans whose H-tick rollouts end in the same board are
  merged (most opening moves are equivalent "take one neutral tile").
* Phases are detected from the board, not the clock:
    opening  no hostile walk reaches our general and neutral land remains
    mid      contact or a live threat
    end      decisive lead, or the tick cap is within the horizon
  Each phase sets the plan set, horizon H, commit interval K and value model.
* Lookahead. Every plan is rolled out H ticks against every opponent model
  with joint simultaneous steps in sim36; the leaf is scored by a learned
  per-phase logistic model whose features extrapolate income to the cap.
  When the cap falls inside H the rollout runs to the end and the exact
  tiebreak decides.
* The plan is committed for K ticks or until an event; v36's safety
  invariant still filters every live action.
"""
from __future__ import annotations

import json
import math
import pathlib

from core36 import (INF, CastleRusher, Econ, Pass, Rusher, Script, descent,
                    growth, mv, top_stacks, walk)

PHASES = ("opening", "mid", "end")
HORIZON = {"opening": 24, "mid": 40, "end": 60}
COMMIT = {"opening": 6, "mid": 10, "end": 8}


# --------------------------------------------------------------------------- #
# macro plans (cheap: no safety rollouts inside rollouts)
# --------------------------------------------------------------------------- #
class Macro(Script):
    """Base: a goal, then plain expansion once the goal is reached."""

    name = "macro"

    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.econ = Econ(ctx, p)

    def expand(self, sim):
        e = self.econ
        return (e.expansion(sim, allow_producer=False) or e.to_frontier(sim)
                or e.expansion(sim, allow_producer=True) or e.general_out(sim))

    def goal(self, sim):
        return 0

    def general_ok(self, sim, a):
        """Cheap rollout version of v36's strict invariant: what the general
        keeps must beat the strongest hostile stack's shortest walk."""
        p = self.p
        g = sim.gen[p]
        if a <= 0 or a // 9 != g:
            return True
        cell, k = divmod(a, 9)
        keep = sim.army[g] // 2 + sim.army[g] % 2 if k >= 5 else 1
        dist = self.ctx.dist(g)
        for s in top_stacks(sim, 1 - p, 3, 3):
            d = dist[s]
            if d < INF and sim.army[s] - d > keep + growth(sim, sim.step, d) - 3:
                return False
        return True

    def act(self, sim):
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        if tg >= 0:
            for v in self.ctx.b.nbrs[tg]:
                if sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg]:
                    return mv(self.ctx.b, v, tg)
        a = self.goal(sim)
        if a and a > 0 and not self.general_ok(sim, a):
            a = 0
            if self.path and self.path[0] == sim.gen[p]:
                pass
        if a:
            return a
        a = self.expand(sim)
        if a and not self.general_ok(sim, a):
            e = self.econ
            a = e.expansion(sim, allow_producer=False) or e.to_frontier(sim) or 0
        return a


class Expand(Macro):
    name = "expand"


class Hold(Macro):
    """Bank at producers; other stacks still expand."""

    name = "hold"

    def act(self, sim):
        e = self.econ
        return e.expansion(sim, allow_producer=False) or e.to_frontier(sim) or 0


class TakeCastle(Macro):
    """Bank at the best source until the walk takes castle c, then walk."""

    def __init__(self, ctx, p, castle):
        super().__init__(ctx, p)
        self.c = castle
        self.name = f"castle{castle}"
        self.done_goal = False

    def goal(self, sim):
        if self.done_goal or sim.owner[self.c] == self.p:
            self.done_goal = True
            return 0
        if self.path:
            a = self.follow(sim)
            if a:
                return a
        p = self.p
        sources = [s for s in self.ctx.castles if sim.owner[s] == p] + [sim.gen[p]]
        best, best_x = None, -INF
        for s in sources:
            path = descent(self.ctx, sim, p, s, self.c)
            if path is None or len(path) > 30:
                continue
            x = walk(sim, p, path) - sim.army[self.c]
            if x > best_x:
                best, best_x = path, x
        if best is None:
            self.done_goal = True
            return 0
        if best_x > 0:
            self.path = best
            return self.follow(sim)
        # bank at the source: let other stacks keep expanding
        e = self.econ
        return e.expansion(sim, allow_producer=False) or e.gather(sim, best[0]) or -1


class GatherHome(Macro):
    name = "gather_home"

    def goal(self, sim):
        return self.econ.gather(sim, sim.gen[self.p]) or -1


class Strike(Macro):
    """Concentrate on the producer nearest the enemy general, then charge
    when the exact walk beats its garrison plus growth."""

    name = "strike"

    def goal(self, sim):
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        if tg < 0:
            return 0
        if self.path:
            a = self.follow(sim)
            if a:
                return a
        for s in top_stacks(sim, p, 3, 10):
            if s == sim.gen[p]:
                continue
            path = descent(self.ctx, sim, p, s, tg)
            if path is None:
                continue
            x = walk(sim, p, path)
            if x > sim.army[tg] + growth(sim, sim.step, len(path) - 1) + 2:
                self.path = path
                return self.follow(sim)
        base = self.econ.strike_base(sim)
        if base is None:
            return 0
        return self.econ.gather(sim, base) or 0


class Rush(Macro):
    """All-in on the enemy general from any stack whose walk wins."""

    name = "rush"

    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.r = Rusher(ctx, p)

    def goal(self, sim):
        return self.r.act(sim)


def plan_set(ctx, sim, p, phase):
    """Candidate macro factories for this phase (the reduced branching)."""
    out = [("expand", lambda: Expand(ctx, p)), ("hold", lambda: Hold(ctx, p))]
    g = sim.gen[p]
    near = []
    for c in ctx.castles:
        if sim.owner[c] != p:
            d = ctx.dist(c)[g]
            if d < INF:
                near.append((d + 0.5 * sim.army[c], c))
    near.sort()
    k = 2 if phase == "opening" else 3
    for _s, c in near[:k]:
        out.append((f"castle{c}", (lambda c=c: TakeCastle(ctx, p, c))))
    if phase != "opening":
        out.append(("gather_home", lambda: GatherHome(ctx, p)))
        out.append(("strike", lambda: Strike(ctx, p)))
        out.append(("rush", lambda: Rush(ctx, p)))
    return out


def opponent_models(ctx, o):
    return {
        "rusher": lambda: Rusher(ctx, o),
        "castle_rusher": lambda: CastleRusher(ctx, o),
        "expander": lambda: Expand(ctx, o),
        "hold": lambda: Pass(ctx, o),
    }


# --------------------------------------------------------------------------- #
# phase detection and features
# --------------------------------------------------------------------------- #
def summary(sim):
    """Per-player land, army, producers, general, largest stack."""
    land = [0, 0]
    army = [0, 0]
    big = [0, 0]
    neutral_free = 0
    prod = [1 if sim.gen[q] >= 0 else 0 for q in (0, 1)]
    owner, a = sim.owner, sim.army
    castle = sim.b.castle
    passable = sim.b.passable
    for c in range(len(owner)):
        o = owner[c]
        if o >= 0:
            land[o] += 1
            army[o] += a[c]
            if a[c] > big[o]:
                big[o] = a[c]
            if castle[c]:
                prod[o] += 1
        elif passable[c] and not castle[c] and a[c] == 0:
            neutral_free += 1
    gen = [a[sim.gen[q]] if sim.gen[q] >= 0 else 0 for q in (0, 1)]
    return land, army, prod, gen, big, neutral_free


def detect_phase(ctx, sim, p, margin=None, summ=None):
    land, army, prod, gen, big, free = summ or summary(sim)
    o = 1 - p
    left = sim.max_steps - sim.step
    if left <= HORIZON["end"]:
        return "end"
    if (land[p] >= 2 * land[o] + 20 and army[p] >= 2 * army[o] + 40) or \
            (land[o] >= 2 * land[p] + 20 and army[o] >= 2 * army[p] + 40):
        return "end"
    contact = margin is not None and margin < INF
    if not contact and free > 0.35 * (free + land[0] + land[1]):
        return "opening"
    return "mid"


FEATURES = ("bias", "land", "army", "prod", "income_to_cap", "defence",
            "attack", "big", "left")


def features(sim, p, summ=None):
    land, army, prod, gen, big, _free = summ or summary(sim)
    o = 1 - p
    left = max(0, sim.max_steps - sim.step)
    inc = [0.5 * prod[q] + land[q] / sim.land_int for q in (0, 1)]
    return [
        1.0,
        (land[p] - land[o]) / 50.0,
        (army[p] - army[o]) / 100.0,
        float(prod[p] - prod[o]),
        (inc[p] - inc[o]) * left / 100.0,     # income extrapolated to the cap
        (gen[p] - big[o]) / 50.0,             # can our general absorb their stack
        (big[p] - gen[o]) / 50.0,             # can our stack break their general
        (big[p] - big[o]) / 50.0,
        left / 500.0,
    ]


class ValueModel:
    """Per-phase logistic win model, trained by train_value37.py."""

    def __init__(self, path=None):
        self.w = {}
        path = pathlib.Path(path) if path else pathlib.Path(__file__).with_name("value37.json")
        if path.exists():
            self.w = json.loads(path.read_text())["weights"]

    def __call__(self, sim, p, phase):
        if sim.done:
            return 1.0 if sim.winner == p else 0.0 if sim.winner >= 0 else 0.5
        summ = summary(sim)
        x = features(sim, p, summ)
        w = self.w.get(phase)
        if w is None:
            # untrained fallback: hand-set economy/safety weights
            w = [0.0, 1.0, 0.5, 0.6, 0.8, 0.4, 0.3, 0.1, 0.0]
        z = sum(a * b for a, b in zip(w, x))
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------------------- #
# lookahead
# --------------------------------------------------------------------------- #
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


def plan(ctx, sim, p, phase, value, weights, stats=None):
    """Choose a macro: robust blend of opponent-weighted mean and worst case."""
    horizon = HORIZON[phase]
    left = sim.max_steps - sim.step
    if left <= horizon:
        horizon = left            # play to the cap: exact tiebreak at the leaf
    cands = plan_set(ctx, sim, p, phase)
    models = opponent_models(ctx, 1 - p)
    table = {}
    seen = {}
    for name, fac in cands:
        vals = {}
        sig = []
        for oname, ofac in models.items():
            leaf = rollout(sim, p, fac, ofac, horizon)
            vals[oname] = value(leaf, p, phase)
            sig.append(hash((tuple(leaf.owner), tuple(leaf.army))))
        sig = tuple(sig)
        if sig in seen:             # same boards against every model: merge
            if stats is not None:
                stats["merged"] = stats.get("merged", 0) + 1
            continue
        seen[sig] = name
        table[name] = (fac, vals)
    tot = sum(weights.values()) or 1.0
    best, best_s = None, -1.0
    for name, (fac, vals) in table.items():
        mean = sum(weights[o] * v for o, v in vals.items()) / tot
        worst = min(vals.values())
        s = 0.5 * mean + 0.5 * worst
        if s > best_s:
            best, best_s = name, s
    if stats is not None:
        stats["branches"] = len(cands)
        stats["kept"] = len(table)
        stats["horizon"] = horizon
        stats["scores"] = {n: round(0.5 * sum(weights[o] * v for o, v in vals.items()) / tot
                                    + 0.5 * min(vals.values()), 3)
                           for n, (_f, vals) in table.items()}
    return best, table[best][0]
