"""Human-style stress opponents for v37+ (sim36 scripts and live wrappers).

GatherRusher: expand until tick `t0`, then sweep territory into its largest
stack (sticky gather) and go all-in on the enemy general as soon as the exact
walk beats that general plus its growth during the walk.
"""
from __future__ import annotations

from core36 import PathRush, Script, best_walk, growth, route_tree, top_stacks
from core37 import Expand


class GatherRusher(Script):
    name = "gather_rusher"

    def __init__(self, ctx, p, t0=120, margin=0):
        super().__init__(ctx, p)
        self.t0, self.margin = t0, margin
        self.exp = Expand(ctx, p)
        self.rush = None

    def act(self, sim):
        if self.rush is not None:
            return self.rush.act(sim)
        if sim.step < self.t0:
            return self.exp.act(sim)
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        if tg < 0:
            return 0
        tree = route_tree(self.ctx, sim, p, tg)
        for s in top_stacks(sim, p, 3, 5):
            x, path = best_walk(self.ctx, sim, p, s, tg, tree)
            if path and x > sim.army[tg] + growth(sim, sim.step, len(path) - 1) + self.margin:
                self.rush = PathRush(self.ctx, p, path)
                return self.rush.act(sim)
        big = top_stacks(sim, p, 1, 2)
        if not big:
            return self.exp.act(sim)
        return self.exp.econ.gather(sim, big[0]) or self.exp.act(sim)
