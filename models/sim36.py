"""Exact, incremental pure-Python replica of `genmod.env.step` for two players.

The JAX step is the reference rules path. Calling it per tick from Python costs
dispatch overhead on every call and copies the whole board, which makes
hundreds of 200-tick rollouts per decision impossible. A tick changes at most
four cells plus production, so this replica updates flat lists in place.

`tests` in `models/tools36/check_sim36.py` replay thousands of random joint
actions through both implementations and require identical boards; run it after
any edit here.

Rollouts stop at the first general capture (the game is over), so the
"captor inherits cells at half strength" rule is not reproduced.
"""
from __future__ import annotations

import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Board:
    """Static map data shared by every simulated state of one match."""

    def __init__(self, mountain, castle, h, w):
        self.h, self.w = h, w
        self.n = h * w
        mountain = np.asarray(mountain, bool).reshape(-1)
        self.mountain = [bool(x) for x in mountain]
        self.castle = [bool(x) for x in np.asarray(castle, bool).reshape(-1)]
        # step_to[cell][d] -> destination cell or -1 (out of bounds / mountain)
        self.step_to = []
        self.nbrs = []
        for cell in range(self.n):
            r, c = divmod(cell, w)
            row, nb = [], []
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not self.mountain[rr * w + cc]:
                    row.append(rr * w + cc)
                    nb.append(rr * w + cc)
                else:
                    row.append(-1)
            self.step_to.append(tuple(row))
            self.nbrs.append(tuple(nb))
        self.passable = [not m for m in self.mountain]

    def direction(self, src, dst):
        row = self.step_to[src]
        for d in range(4):
            if row[d] == dst:
                return d
        return -1


class Sim:
    """Mutable game state. `owner`/`army` are flat lists; generals by player."""

    __slots__ = ("b", "owner", "army", "gen", "step", "max_steps", "done",
                 "winner", "producers", "gi_g", "gil", "ggi", "gl", "land_int")

    def __init__(self, board, owner, army, general_idx, step, max_steps,
                 general_interval=2, land_interval=50):
        self.b = board
        self.owner = [int(x) for x in np.asarray(owner).reshape(-1)]
        self.army = [int(x) for x in np.asarray(army).reshape(-1)]
        self.gen = [int(x) for x in general_idx]
        self.step = int(step)
        self.max_steps = int(max_steps)
        self.done = False
        self.winner = -1
        self.gil = general_interval
        self.land_int = land_interval
        prods = [i for i in range(board.n) if board.castle[i]]
        prods += [g for g in self.gen if g >= 0 and g not in prods]
        self.producers = prods

    @classmethod
    def from_state(cls, state, board=None, max_steps=None):
        cfg = getattr(state, "config", {}) or {}
        b = board or Board(state.mountain, state.castle, state.h, state.w)
        ms = int(cfg.get("maxTicks", 2000)) if max_steps is None else max_steps
        return cls(b, state.owner, state.army, list(np.asarray(state.general_idx)),
                   int(state.tick), ms,
                   int(cfg.get("generalGrowthInterval", 2)),
                   int(cfg.get("landGrowthInterval", 50)))

    def copy(self):
        s = Sim.__new__(Sim)
        s.b = self.b
        s.owner = self.owner[:]
        s.army = self.army[:]
        s.gen = self.gen[:]
        s.step = self.step
        s.max_steps = self.max_steps
        s.done = self.done
        s.winner = self.winner
        s.producers = self.producers
        s.gil = self.gil
        s.land_int = self.land_int
        return s

    # ------------------------------------------------------------------ #
    def legal(self, p, action):
        if action <= 0:
            return False
        cell, k = divmod(action, 9)
        if k == 0 or cell >= self.b.n:
            return False
        if self.owner[cell] != p or self.army[cell] < 2:
            return False
        return self.b.step_to[cell][(k - 1) % 4] >= 0

    def step2(self, a0, a1):
        """Advance one simultaneous tick with actions for players 0 and 1."""
        if self.done:
            return
        owner, army = self.owner, self.army
        moves = []
        for p, a in ((0, a0), (1, a1)):
            if a > 0 and self.gen[p] >= 0 and self.legal(p, a):
                cell, k = divmod(a, 9)
                m = k - 1
                dst = self.b.step_to[cell][m % 4]
                sent = army[cell] // 2 if m >= 4 else army[cell] - 1
                moves.append((p, cell, dst, sent))
        for p, cell, dst, sent in moves:
            army[cell] -= sent
        if moves:
            if len(moves) == 2 and moves[0][2] == moves[1][2]:
                dsts = {moves[0][2]: [0, 0]}
                dsts[moves[0][2]][moves[0][0]] += moves[0][3]
                dsts[moves[0][2]][moves[1][0]] += moves[1][3]
            else:
                dsts = {}
                for p, cell, dst, sent in moves:
                    inc = [0, 0]
                    inc[p] = sent
                    dsts[dst] = inc
            for dst, inc in dsts.items():
                o = owner[dst]
                f0 = inc[0] + (army[dst] if o == 0 else 0)
                f1 = inc[1] + (army[dst] if o == 1 else 0)
                strength = f0 - f1 if f0 >= f1 else f1 - f0
                challenger = 1 if f1 > f0 else 0
                if o < 0:
                    after = strength - army[dst]
                    if after > 0:
                        owner[dst] = challenger
                        army[dst] = after
                    else:
                        army[dst] = -after
                else:
                    if strength > 0:
                        owner[dst] = challenger
                    army[dst] = strength
        # general capture (checked before production)
        cap0 = self.gen[0] >= 0 and owner[self.gen[0]] != 0
        cap1 = self.gen[1] >= 0 and owner[self.gen[1]] != 1
        t = self.step + 1
        self.step = t
        if cap0 or cap1:
            self.done = True
            self.winner = -1 if (cap0 and cap1) else (1 if cap0 else 0)
            return
        if t % self.gil == 0:
            for c in self.producers:
                if owner[c] >= 0:
                    army[c] += 1
        if t % self.land_int == 0:
            for c in range(self.b.n):
                if owner[c] >= 0:
                    army[c] += 1
        if t >= self.max_steps:
            self.done = True
            self.winner = self.tiebreak()

    def tiebreak(self):
        land = [0, 0]
        tot = [0, 0]
        for o, a in zip(self.owner, self.army):
            if o >= 0:
                land[o] += 1
                tot[o] += a
        if land[0] != land[1]:
            return 0 if land[0] > land[1] else 1
        if tot[0] != tot[1]:
            return 0 if tot[0] > tot[1] else 1
        return -1

    def totals(self):
        land = [0, 0]
        tot = [0, 0]
        for o, a in zip(self.owner, self.army):
            if o >= 0:
                land[o] += 1
                tot[o] += a
        return land, tot
