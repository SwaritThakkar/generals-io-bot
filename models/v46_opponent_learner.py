"""v46: v44 + learning the opponent (in-game Bayes, cross-game bandit) and an
income-race castle gate.

Research backing (full list in RESEARCH_IDEAS.md, "v46"):
* Bayesian opponent typing during the game (Synnaeve & Bessiere, AIIDE 2011;
  Weber & Mateas, CIG 2009): v37's posterior over opponent scripts
  (rusher, castle_rusher, expander, hold), updated every tick from the
  opponent's observed move, plus observed facts (castles held, stacks sent at
  our general, their general walking out).
* Safe best response (Ganzfried & Sandholm, AAMAS 2011 and ACM TEAC 2015):
  deviate from the default toward the best answer to the inferred type, but
  only inside v44's safety layer (race margin, reserve, verify()).
* Cross-game strategy selection (Tavares et al., AIIDE 2016) with Exp3
  (Auer et al., SIAM J. Comput. 2002; Lisy et al. 2013 for simultaneous
  games): a small portfolio of parameter arms; one arm per game, rewarded by
  the game's outcome. Only through the live `act` entry point and only when
  V46_MEMORY names a file, so test harnesses stay independent.
* Income race (Churchill & Buro, AIIDE 2011 economic planning; v37's
  income_to_cap feature): castles are bought while our projected lead at the
  tick cap is not yet safe, and not once it is.

Gated tree, re-evaluated every tick:
    castle gate  OPEN if the income race is not safely won, or the opponent
                 holds / masses on castles (deny them);  SHUT = cheap castles
                 from non-general stacks only
    threat gate  OPEN if they attacked, walked their general out, or the
                 posterior puts > 50% on the two rush types -> reserve high +
                 counter;  SHUT -> reserve low, no counter
    arm          (cross-game) shifts reserve and castle appetite
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import random
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
from core36 import ctx_for  # noqa: E402
from sim36 import Board, Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v45_for_v46", pathlib.Path(core36.__file__).with_name("v45_gated_opponent.py"))
_v45 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v45)
_v44 = _v45._v44

_MEM = {}
_LAST = {}            # seat -> (planner, last board) of the game in progress
STATS = {"gate_C": 0, "gate_c": 0, "gate_T": 0, "gate_t": 0}
OFF = set(filter(None, os.environ.get("V46_OFF", "").split(",")))
MEMORY = os.environ.get("V46_MEMORY")
RES_HIGH, RES_LOW = 0.6, 0.4
SAFE_LEAD = 0.25          # projected army lead at the cap, as a share of theirs

#: portfolio for cross-game selection: (reserve shift, cheap-castle limit when shut)
ARMS = {"base": (0.0, 30), "greedy": (-0.1, 60), "safe": (0.1, 20), "castles": (0.0, 90)}


def income(sim, q, castles):
    owner = sim.owner
    land = sum(1 for c in range(len(owner)) if owner[c] == q)
    prod = (1 if sim.gen[q] >= 0 else 0) + sum(1 for c in castles if owner[c] == q)
    return 0.5 * prod + land / sim.land_int, land


class Exp3:
    """Cross-game arm selection (Auer et al. 2002), persisted as JSON."""

    def __init__(self, path, gamma=0.2):
        self.path, self.gamma = path, gamma
        self.w = {a: 1.0 for a in ARMS}
        try:
            self.w.update(json.loads(pathlib.Path(path).read_text())["w"])
        except Exception:
            pass

    def probs(self):
        tot = sum(self.w.values())
        k = len(self.w)
        return {a: (1 - self.gamma) * w / tot + self.gamma / k for a, w in self.w.items()}

    def pick(self):
        pr = self.probs()
        r, acc = random.random(), 0.0
        for a, p in pr.items():
            acc += p
            if r <= acc:
                return a
        return "base"

    def update(self, arm, reward):
        p = self.probs()[arm]
        self.w[arm] *= math.exp(self.gamma * reward / (p * len(self.w)))
        m = max(self.w.values())
        self.w = {a: w / m for a, w in self.w.items()}
        try:
            pathlib.Path(self.path).write_text(json.dumps({"w": self.w}))
        except Exception:
            pass


class Planner46(_v45.Planner45):
    def __init__(self, ctx, p, arm="base"):
        super().__init__(ctx, p)
        self.arm = arm
        self.castle_gate = True
        self.threat_gate = True

    def update_gates(self, sim):
        p, o = self.p, 1 - self.p
        castles = self.ctx.castles
        inc_p, land_p = income(sim, p, castles)
        inc_o, land_o = income(sim, o, castles)
        _land, tot = sim.totals()
        left = max(0, sim.max_steps - sim.step)
        lead = (tot[p] - tot[o]) + (inc_p - inc_o) * left
        race_won = lead > SAFE_LEAD * max(50, tot[o]) and inc_p >= inc_o and land_p > land_o
        self.castle_gate = (not race_won) or self.opp.castle_open
        w = self.weights
        tot_w = sum(w.values()) or 1.0
        rushy = (w.get("rusher", 0) + w.get("castle_rusher", 0)) / tot_w
        self.threat_gate = self.opp.threat_open or (sim.step >= _v45.OBSERVE and rushy > 0.5)
        STATS["gate_" + ("C" if self.castle_gate else "c")] += 1
        STATS["gate_" + ("T" if self.threat_gate else "t")] += 1

    def castle_options(self, sim):
        opts = _v44.Planner44.castle_options(self, sim)
        if "castle_gate" in OFF or self.castle_gate:
            return opts
        g = sim.gen[self.p]
        cheap = ARMS[self.arm][1]
        return [o for o in opts if o[3][0] != g and sim.army[o[2][-1]] + len(o[3]) - 1 <= cheap]

    def counter(self, sim):
        if not self.threat_gate:
            return 0
        return _v44.Planner44.counter(self, sim)

    def gates(self):
        return ("C" if self.castle_gate else "c") + ("T" if self.threat_gate else "t")

    def act(self, sim):
        self.opp.update(sim)
        self.update_gates(sim)
        shift = ARMS[self.arm][0]
        _v44.RESERVE = (RES_HIGH if self.threat_gate else RES_LOW) + shift
        a = _v44.Planner44.act(self, sim)
        self.why = f"{self.gates()}:{self.why}"
        return a


Planner = Planner46
_BANDIT = Exp3(MEMORY) if MEMORY and "bandit" not in OFF else None


def live_ctx(state):
    mountain = np.asarray(state.mountain, bool)
    castle = np.asarray(state.castle, bool)
    key = (state.h, state.w, mountain.tobytes(), castle.tobytes())
    return ctx_for(key, lambda: Board(mountain, castle, state.h, state.w))


def _reward(sim, p):
    """Outcome of a finished (or abandoned) game from our last view of it."""
    if sim.gen[p] < 0:
        return 0.0
    if sim.gen[1 - p] < 0:
        return 1.0
    land, tot = sim.totals()
    share = land[p] / max(1, land[p] + land[1 - p])
    return 0.25 + 0.5 * share


def act(state, player):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        # a new game: reward the previous one (any map) for this seat
        last = _LAST.pop(player, None)
        if last is not None and _BANDIT is not None:
            _BANDIT.update(last[0].arm, _reward(last[1], player))
        arm = _BANDIT.pick() if _BANDIT is not None else "base"
        mem = [int(state.tick), Planner46(ctx, player, arm)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    _LAST[player] = (mem[1], sim)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
