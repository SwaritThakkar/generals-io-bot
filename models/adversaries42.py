"""Human-style test opponent from the recorded wins against v37.

SpreadCollector: expand, then park stacks on castles (half from the general,
any stack that can take one), and from `t_collect` gather everything onto the
forward castle nearest the enemy general and strike once the walk wins.
Defence is crude: consolidate home when a hostile walk would beat the general.
"""
from __future__ import annotations

from core36 import INF, Econ, Script, descent, growth, home_margin, mv, threats, top_stacks, walk


class SpreadCollector(Script):
    name = "spread_collector"

    def __init__(self, ctx, p, t_castle=80, t_collect=300, margin=5):
        super().__init__(ctx, p)
        self.e = Econ(ctx, p)
        self.t_castle, self.t_collect, self.margin = t_castle, t_collect, margin

    def castle(self, sim):
        p = self.p
        g = sim.gen[p]
        sources = [g] + [c for c in self.ctx.castles if sim.owner[c] == p and sim.army[c] >= 4]
        sources += [s for s in top_stacks(sim, p, 4, 10) if s not in sources]
        best = None
        for c in self.ctx.castles:
            if sim.owner[c] == p:
                continue
            for s in sources:
                path = descent(self.ctx, sim, p, s, c)
                if path is None or len(path) < 2 or len(path) > 20:
                    continue
                half = s == g
                sent = sim.army[s] // 2 if half else sim.army[s] - 1
                need = sim.army[c] + (growth(sim, sim.step, len(path) - 1) if sim.owner[c] >= 0 else 0)
                if sent < 2 or walk(sim, p, path, start=sent + 1) <= need + 1:
                    continue
                key = need + len(path)
                if best is None or key < best[0]:
                    best = (key, mv(self.ctx.b, s, path[1], half=half), path)
        if best is None:
            return 0
        self.path, self.kind = best[2], "castle"
        return best[1]

    def strike(self, sim):
        p, o = self.p, 1 - self.p
        tg = sim.gen[o]
        for s in top_stacks(sim, p, 4, 10):
            path = descent(self.ctx, sim, p, s, tg)
            if path is None:
                continue
            if walk(sim, p, path) > sim.army[tg] + growth(sim, sim.step, len(path) - 1) + self.margin:
                self.path, self.kind = path, "strike"
                return mv(self.ctx.b, s, path[1])
        return 0

    def act(self, sim):
        p, o = self.p, 1 - self.p
        g, tg = sim.gen[p], sim.gen[o]
        if g < 0 or tg < 0:
            return 0
        b = self.ctx.b
        for v in b.nbrs[tg]:
            if sim.owner[v] == p and sim.army[v] - 1 > sim.army[tg]:
                return mv(b, v, tg)
        if self.path:
            a = self.follow(sim)
            if a:
                return a
            self.path = None
        thr = threats(self.ctx, sim, p)
        if thr and home_margin(self.ctx, sim, p, thr=thr) < 0:
            a = self.e.gather(sim, g)
            if a:
                return a
        a = self.strike(sim)
        if a:
            return a
        if sim.step >= self.t_castle:
            a = self.castle(sim)
            if a:
                return a
        if sim.step >= self.t_collect:
            base = self.e.strike_base(sim)
            if base is not None:
                a = self.e.gather(sim, base)
                if a:
                    return a
        e = self.e
        return (e.expansion(sim, allow_producer=False) or e.to_frontier(sim)
                or e.expansion(sim, allow_producer=True) or e.general_out(sim))
