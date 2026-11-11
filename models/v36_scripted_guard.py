"""v36: scripted economy with an exact-walk home-safety invariant (no search).

Baseline for the v37 simultaneous macro search. Each tick it rebuilds the
board from the live state and runs `core36.Econ`, which
  * attacks the enemy general only when an exact all-in walk beats its
    garrison plus growth plus one reinforcement, and the home race is safe;
  * lets the general send units only while every hostile walk on our general
    still loses against general + growth + one reinforcement (`home_margin`);
  * otherwise takes the earliest castle, expands with non-producer stacks and
    gathers toward the producer it is banking at.
"""
from __future__ import annotations

import pathlib
import sys

# The local server imports agents from a temporary copy, so locate models/
# through the project root that is already on sys.path.
for _root in [pathlib.Path(__file__).resolve().parent, *map(pathlib.Path, sys.path)]:
    for _cand in (_root, _root / "models"):
        if (_cand / "core36.py").exists():
            sys.path.insert(0, str(_cand))
            break
    else:
        continue
    break

import numpy as np  # noqa: E402

from core36 import Econ, ctx_for  # noqa: E402
from sim36 import Board, Sim  # noqa: E402

_MEM = {}


def live_ctx(state):
    mountain = np.asarray(state.mountain, bool)
    castle = np.asarray(state.castle, bool)
    key = (state.h, state.w, mountain.tobytes(), castle.tobytes())
    return ctx_for(key, lambda: Board(mountain, castle, state.h, state.w))


def script_for(state, player, factory):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        mem = [int(state.tick), factory(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    return ctx, mem[1]


import json as _json  # noqa: E402
import os as _os  # noqa: E402

#: experiment hook: V36_PARAMS='{"castle_wait": 0}' overrides Econ arguments
PARAMS = _json.loads(_os.environ.get("V36_PARAMS", "{}"))


def act(state, player):
    ctx, script = script_for(state, player, lambda c, p: Econ(c, p, **PARAMS))
    sim = Sim.from_state(state, ctx.b)
    a = script.act(sim)
    return int(a) if a and sim.legal(player, a) else 0
