"""v45: v44 + an opponent-gated policy tree.

The bot watches the opponent during the game and opens or closes gates that
switch v44's modules, so it answers what the opponent actually does:

    castle gate   OPEN  the opponent owns a castle, has taken one, or masses
                        a stack next to a neutral castle (>= 70% of its
                        garrison)
                        -> full castle layer, enemy-held castles first
                  SHUT  -> only cheap castles (garrison + walk <= CHEAP),
                           never sent from the general; land and defence first
    threat gate   OPEN  a hostile stack worth >= 25% of our general came within
                        8 steps, or their general walked out (dropped by half
                        while holding >= 20)
                        -> reserve RES_HIGH, counter-strike on
                  SHUT  (after OBSERVE ticks without either)
                        -> reserve RES_LOW, counter-strike off
Before OBSERVE ticks the threat gate starts open (v44 defaults).
Gates latch open (an opponent that took castles or attacked once will again)
except the castle gate, which shuts again if the opponent holds no castle
for SHUT_AFTER ticks.
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

import numpy as np  # noqa: E402

import core36  # noqa: E402
from core36 import ctx_for, top_stacks  # noqa: E402
from sim36 import Board, Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v44_for_v45", pathlib.Path(core36.__file__).with_name("v44_reserve_counter.py"))
_v44 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v44)

_MEM = {}
STATS = {"castle_gate_open": 0, "threat_gate_open": 0, "castle_filtered": 0}
OFF = set(filter(None, os.environ.get("V45_OFF", "").split(",")))
CHEAP = int(os.environ.get("V45_CHEAP", "30"))
OBSERVE = 150
SHUT_AFTER = 150
RES_HIGH, RES_LOW = 0.6, 0.4


class OpponentModel:
    """Online opponent profile: castle play and aggression."""

    def __init__(self, ctx, p):
        self.ctx, self.p = ctx, p
        self.castle_open = False
        self.last_castle_tick = -10 ** 9
        self.threat_open = True
        self.threat_seen = False
        self.prev_gen = None

    def update(self, sim):
        ctx, p = self.ctx, self.p
        o = 1 - p
        owner, army = sim.owner, sim.army
        # castle gate
        held = any(owner[c] == o for c in ctx.castles)
        massing = False
        for c in ctx.castles:
            if owner[c] < 0:
                for v in ctx.b.nbrs[c]:
                    if owner[v] == o and army[v] >= max(3, 0.7 * army[c]):
                        massing = True
                        break
        if held or massing:
            self.last_castle_tick = sim.step
            if not self.castle_open:
                STATS["castle_gate_open"] += 1
            self.castle_open = True
        elif self.castle_open and sim.step - self.last_castle_tick > SHUT_AFTER:
            self.castle_open = False
        # threat gate
        g, tg = sim.gen[p], sim.gen[o]
        if g >= 0:
            dist = ctx.dist(g)
            G = max(10, army[g])
            for s in top_stacks(sim, o, 4, 5):
                if s != tg and dist[s] <= 8 and army[s] >= 0.25 * G:
                    self.threat_seen = True
        if tg >= 0:
            if self.prev_gen is not None and self.prev_gen >= 20 and army[tg] < 0.5 * self.prev_gen:
                self.threat_seen = True
            self.prev_gen = army[tg]
        if self.threat_seen:
            if not self.threat_open:
                STATS["threat_gate_open"] += 1
            self.threat_open = True
        elif sim.step >= OBSERVE:
            self.threat_open = False


class Planner45(_v44.Planner44):
    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.opp = OpponentModel(ctx, p)

    def gates(self):
        return ("C" if self.opp.castle_open else "c") + ("T" if self.opp.threat_open else "t")

    def castle_options(self, sim):
        opts = super().castle_options(sim)
        if "castle_gate" in OFF or self.opp.castle_open:
            return opts
        g = sim.gen[self.p]
        out = []
        for opt in opts:
            _key, _a, path, full, _half = opt
            c = path[-1]
            need = sim.army[c]
            if full[0] != g and need + len(full) - 1 <= CHEAP:
                out.append(opt)
            else:
                STATS["castle_filtered"] += 1
        return out

    def counter(self, sim):
        if "threat_gate" not in OFF and not self.opp.threat_open:
            return 0
        return super().counter(sim)

    def act(self, sim):
        self.opp.update(sim)
        if "threat_gate" not in OFF:
            _v44.RESERVE = RES_HIGH if self.opp.threat_open else RES_LOW
        a = super().act(sim)
        self.why = f"{self.gates()}:{self.why}"
        return a


Planner = Planner45


def live_ctx(state):
    mountain = np.asarray(state.mountain, bool)
    castle = np.asarray(state.castle, bool)
    key = (state.h, state.w, mountain.tobytes(), castle.tobytes())
    return ctx_for(key, lambda: Board(mountain, castle, state.h, state.w))


def act(state, player):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        mem = [int(state.tick), Planner45(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
