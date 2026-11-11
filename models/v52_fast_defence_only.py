"""v52: v46 + v51's committed fast defence, without v51's opponent classifier.

= v51 minus the classifier: the approach detector, the defence need, the
PCST gather into the general with an ETA - 2 move budget and the
merge-in of stacks next to the general are v51's; opponent typing is v46's
own (move-likelihood posterior over four scripts). v51 is loaded privately
with its classifier switched off, so v51 itself is unchanged.
"""
from __future__ import annotations

import importlib.util as _ilu
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

import core36  # noqa: E402
from sim36 import Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v51_for_v52", pathlib.Path(core36.__file__).with_name("v51_fast_defence.py"))
_v51 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v51)
_v51.OFF.add("classifier")
_v51.CLS = None

_MEM = {}
STATS = _v51.STATS


class Planner52(_v51.Planner51):
    pass


Planner = Planner52
live_ctx = _v51.live_ctx


def act(state, player):
    ctx = live_ctx(state)
    key = (id(ctx), player)
    mem = _MEM.get(key)
    if mem is None or mem[0] > int(state.tick):
        mem = [int(state.tick), Planner52(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
