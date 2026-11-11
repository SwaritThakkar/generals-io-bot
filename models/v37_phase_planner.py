"""v37: phase-aware macro lookahead on top of the v36 safety layer.

Per tick:
  1. v36 handles urgent play: an immediate capture, a verified rush, or
     defence when the general alone would no longer hold (`core36.Econ`).
  2. Otherwise the committed macro plan acts. A new plan is chosen by
     `core37.plan` when the commit interval of the current phase runs out or
     an event happens (phase change, castle changes hands, enemy general
     drained, home margin crosses the pad, plan finished).
  3. Every macro action still passes v36's `safe_after`; if not, v36 acts.

The opponent-model weights are updated each tick: each model predicts the
opponent's last move and is scored on whether the board change matches.

Set V37_LOG=path to append one JSON line per planning decision.
"""
from __future__ import annotations

import json
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

import numpy as np  # noqa: E402

import core37  # noqa: E402
from core36 import INF, Econ, ctx_for, mobile_top  # noqa: E402
from sim36 import Board, Sim  # noqa: E402

_MEM = {}
_VALUE = core37.ValueModel()
LOG = os.environ.get("V37_LOG")


def live_ctx(state):
    mountain = np.asarray(state.mountain, bool)
    castle = np.asarray(state.castle, bool)
    key = (state.h, state.w, mountain.tobytes(), castle.tobytes())
    return ctx_for(key, lambda: Board(mountain, castle, state.h, state.w))


class Planner:
    def __init__(self, ctx, p):
        self.ctx, self.p = ctx, p
        self.econ = Econ(ctx, p)
        self.macro = None
        self.macro_name = None
        self.until = -1
        self.phase = None
        self.sig = None
        self.weights = {k: 1.0 for k in core37.opponent_models(ctx, 1 - p)}
        self.prev = None
        self.why = "none"
        self.margin = INF

    # -- opponent modelling --------------------------------------------------
    def observe(self, sim):
        """Score each opponent model on whether its predicted move explains
        the opponent's cells in the new board."""
        prev = self.prev
        self.prev = sim.copy()
        if prev is None or prev.step + 1 != sim.step:
            return
        o = 1 - self.p
        t = sim.step
        prod_tick = t % sim.gil == 0
        land_tick = t % sim.land_int == 0
        prods = set(sim.producers)

        def expected(cell, sent):
            return prev.army[cell] - sent + (prod_tick and cell in prods) + land_tick

        moved = None
        for c in range(len(prev.owner)):
            if prev.owner[c] == o and prev.army[c] >= 2 and sim.owner[c] == o \
                    and sim.army[c] < expected(c, 0):
                moved = c
                break
        for name, fac in core37.opponent_models(self.ctx, o).items():
            a = fac().act(prev)
            a = a if a and a > 0 else 0
            if a == 0:
                ok = moved is None
            else:
                cell, k = divmod(a, 9)
                sent = prev.army[cell] // 2 if k >= 5 else prev.army[cell] - 1
                ok = moved == cell or (sim.owner[cell] == o and sim.army[cell] == expected(cell, sent))
            # decayed likelihood: consistent moves weigh 4:1
            self.weights[name] = 0.95 * self.weights[name] * (0.8 if ok else 0.2) / 0.5 + 0.05
        tot = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] = self.weights[k] / tot * len(self.weights)

    # -- planning ------------------------------------------------------------
    def replan(self, sim, phase):
        stats = {}
        name, fac = core37.plan(self.ctx, sim, self.p, phase, _VALUE, self.weights, stats)
        self.macro, self.macro_name = fac(), name
        self.until = sim.step + core37.COMMIT[phase]
        if LOG:
            with open(LOG, "a") as f:
                f.write(json.dumps({"tick": sim.step, "player": self.p, "phase": phase,
                                    "choice": name, "weights": {k: round(v, 2) for k, v in self.weights.items()},
                                    **stats}) + "\n")

    def act(self, sim):
        p = self.p
        self.observe(sim)
        econ = self.econ
        g = sim.gen[p]
        if g < 0:
            return 0
        thr, m0 = econ.margin_of(sim)
        pad = econ.pad + econ.pad_frac * mobile_top(sim, 1 - p)
        self.margin = m0
        # 1. urgent play is v36's: capture, rush, defence
        tg = sim.gen[1 - p]
        urgent = m0 <= pad or econ.kind == "rush"
        if not urgent and tg >= 0:
            path, _m = econ.attack_plan(sim, econ.aggression)
            urgent = path is not None
        if urgent:
            econ.kind = None if econ.kind != "rush" else econ.kind
            a = econ.act(sim)
            self.why = "v36:" + econ.why
            return a
        # 2. committed macro, re-planned on schedule or on events
        summ = core37.summary(sim)
        phase = core37.detect_phase(self.ctx, sim, p, m0, summ)
        og = sim.army[tg] if tg >= 0 else 0
        castles = tuple(sim.owner[c] for c in self.ctx.castles)
        sig = (castles, phase)
        drained = self.sig is not None and og < 0.5 * self.sig[2]
        finished = getattr(self.macro, "done_goal", False)
        if (self.macro is None or sim.step >= self.until or self.sig is None
                or sig != self.sig[:2] or drained or finished):
            self.replan(sim, phase)
        self.sig = (castles, phase, max(og, 1))
        self.phase = phase
        a = self.macro.act(sim)
        a = a if a and a > 0 else 0
        econ.checks = 0
        if econ.safe_after(sim, a, m0, pad) and (a == 0 or sim.legal(p, a)):
            self.why = f"{phase}:{self.macro_name}"
            return a
        a = econ.act(sim)
        self.why = "v36_fallback:" + econ.why
        return a


def act(state, player):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        mem = [int(state.tick), Planner(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
