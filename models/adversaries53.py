"""Stress opponents for v53: stacks parked next to our general.

Besieger: expand until `t0`, then collect into one or two siege stacks, walk
them to `post` steps from the enemy general and park there, feeding them from
the rest of the army; all-in on the general as soon as an exact walk beats
it plus its growth (like a human building stacks beside the general and
waiting for a mistake).
"""
from __future__ import annotations

from core36 import INF, PathRush, Script, best_walk, descent, growth, mv, route_tree, top_stacks
from core37 import Expand


class Besieger(Script):
    name = "besieger"

    def __init__(self, ctx, p, t0=100, post=3, posts=1, margin=0):
        super().__init__(ctx, p)
        self.t0, self.post, self.posts, self.margin = t0, post, posts, margin
        self.exp = Expand(ctx, p)
        self.rush = None

    def act(self, sim):
        if self.rush is not None:
            a = self.rush.act(sim)
            if a:
                return a
            self.rush = None
        if sim.step < self.t0:
            return self.exp.act(sim)
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        if tg < 0:
            return 0
        tree = route_tree(self.ctx, sim, p, tg)
        for s in top_stacks(sim, p, 4, 5):
            x, path = best_walk(self.ctx, sim, p, s, tg, tree)
            if path and x > sim.army[tg] + growth(sim, sim.step, len(path) - 1) + self.margin:
                self.rush = PathRush(self.ctx, p, path)
                return self.rush.act(sim)
        dist = self.ctx.dist(tg)
        g = sim.gen[p]
        sieges = [s for s in top_stacks(sim, p, self.posts, 5) if s != g] or top_stacks(sim, p, 1, 5)
        # walk a siege stack towards its post (stop `post` steps short)
        for s in sieges:
            if dist[s] > self.post and dist[s] < INF:
                path = descent(self.ctx, sim, p, s, tg)
                if path and len(path) >= 2 and sim.army[path[1]] < sim.army[s] - 1 or (
                        path and len(path) >= 2 and sim.owner[path[1]] == p):
                    return mv(self.ctx.b, path[0], path[1])
        # feed the siege stacks, smallest first
        for s in sorted(sieges, key=lambda c: sim.army[c]):
            a = self.exp.econ.gather(sim, s)
            if a:
                return a
        return self.exp.act(sim)
