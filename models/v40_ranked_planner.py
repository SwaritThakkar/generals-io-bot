"""v40: learned plan pruning + regret-matching plan game + asymmetric tactics.

Built from the papers summarised in RESEARCH_IDEAS.md ("v40"):
  1. Ranker (Barriga, Stanescu & Buro 2017): a small CNN trained on the Mac
     GPU from regret-matching teacher labels scores every candidate plan;
     only the top-k plus protected plans are searched (k by phase).
  2. Plan game (Lisy et al. 2013): kept plans x six opponent replies are
     filled by exact simultaneous rollouts, PGS+ style (Lelis 2017: the plan
     acts for the commit interval, then a default policy), and solved by
     regret matching instead of v37's mean/worst blend.
  3. Anytime budget (Lelis 2017, SSS+ meta-reasoning): cells are filled in
     priority order until the time budget is used; unfinished replies drop.
  4. Asymmetric tactical step (Moraes & Lelis 2018; Human.exe's
     simultaneous-move minimax for short-range fights): on contact, all raw
     moves of up to four critical stacks are scored against plausible
     simultaneous enemy replies (worst case); the result replaces the plan's
     move only if it scores higher under the same evaluation.
  5. v39 swap-safe defence.

Ranker inference is a tiny forward pass on CPU; training is GPU-only
(train_ranker40.py refuses to run without MPS).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

for _root in [pathlib.Path(__file__).resolve().parent, *map(pathlib.Path, sys.path)]:
    for _cand in (_root, _root / "models"):
        if (_cand / "core40.py").exists():
            sys.path.insert(0, str(_cand))
            break
    else:
        continue
    break

import importlib.util as _ilu  # noqa: E402

import numpy as np  # noqa: E402

import core36  # noqa: E402
import core37  # noqa: E402
import core40  # noqa: E402
from core36 import INF, ctx_for, mv  # noqa: E402
from sim36 import Board, Sim  # noqa: E402

_spec = _ilu.spec_from_file_location("_v37_for_v40", pathlib.Path(core36.__file__).with_name("v37_phase_planner.py"))
_v37 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v37)

LOG = os.environ.get("V40_LOG")
FLAGS = set(filter(None, os.environ.get("V40_OFF", "").split(",")))   # ablations
BUDGET_MS = float(os.environ.get("V40_BUDGET_MS", "150"))
K_KEEP = {"opening": 2, "mid": 3, "end": 3}
STATS = {"plans": 0, "cands": 0, "kept": 0, "cells": 0, "tactic_checks": 0,
         "tactic_used": 0, "ms": 0.0}
_MEM = {}


class _Ranker:
    def __init__(self):
        self.model = None
        path = pathlib.Path(core40.__file__).with_name("ranker40.pt")
        if not path.exists() or "ranker" in FLAGS:
            return
        try:
            import torch
            sys.path.insert(0, str(path.parent))
            from train_ranker40 import Ranker as Net  # noqa: F401 (needs MPS only to train)
        except SystemExit:
            # train_ranker40 refuses to import without MPS; build the net here
            Net = None
        except Exception:
            return
        import torch
        import torch.nn as nn
        if Net is None:
            return
        net = Net()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        self.model, self.torch = net, torch

    def scores(self, x, plan_vecs):
        t = self.torch
        with t.no_grad():
            X = t.tensor(x[None].astype(np.float32))
            pv = t.tensor(np.asarray(plan_vecs, np.float32))
            owner = t.zeros(len(plan_vecs), dtype=t.long)
            logit, _v = self.model(X, pv, owner)
        return logit.numpy()


_RANKER = None


def ranker():
    global _RANKER
    if _RANKER is None:
        _RANKER = _Ranker()
    return _RANKER


def tactical_value(ctx, s2, p):
    """Static score after one joint tick: win/loss, then home margin and
    material/land swing."""
    if s2.done:
        return 1e6 if s2.winner == p else -1e6
    thr = core36.threats(ctx, s2, p)
    m0 = core36.home_margin(ctx, s2, p, thr=thr, reinf=False) if thr else 200
    land, tot = s2.totals()
    o = 1 - p
    og = s2.gen[o]
    tg_press = 0
    if og >= 0:
        for v in ctx.b.nbrs[og]:
            if s2.owner[v] == p:
                tg_press = max(tg_press, s2.army[v] - s2.army[og])
    return 3.0 * min(m0, 200) + (tot[p] - tot[o]) + 2.0 * (land[p] - land[o]) + 0.5 * tg_press


class Planner40(_v37.Planner):
    def __init__(self, ctx, p):
        super().__init__(ctx, p)
        self.econ.swap_safe = True
        self.weights = {k: 1.0 for k in core40.replies(ctx, 1 - p)}

    # -- plan selection ------------------------------------------------------
    def replan(self, sim, phase):
        t0 = time.perf_counter()
        p = self.p
        cands = core37.plan_set(self.ctx, sim, p, phase)
        m0 = self.margin
        kept = cands
        rk = ranker()
        if rk.model is not None and len(cands) > 2:
            x = core40.planes(self.ctx, sim, p)
            pv = [core40.plan_vec(self.ctx, sim, p, n, m0) for n, _f in cands]
            sc = rk.scores(x, pv)
            order = list(np.argsort(-sc))
            keep = list(order[:K_KEEP[phase]])
            for i, (n, _f) in enumerate(cands):
                t = core40.plan_type(n)
                if i not in keep and (t in ("expand", "rush")
                                      or (t == "gather_home" and m0 <= 3 * self.econ.pad + 10)):
                    keep.append(i)
            kept = [cands[i] for i in keep]          # ranker order first
        horizon = min(core37.HORIZON[phase], 40, sim.max_steps - sim.step)
        k = core37.COMMIT[phase]
        models = core40.replies(self.ctx, 1 - p)
        cols = sorted(models, key=lambda o: -self.weights.get(o, 1.0))
        # anytime fill: column by column (most likely replies first)
        M = np.full((len(kept), len(cols)), np.nan)
        deadline = t0 + BUDGET_MS / 1000.0
        filled_cols = 0
        out_of_time = False
        for j, oname in enumerate(cols):
            for i, (name, fac) in enumerate(kept):
                ours = (lambda fac=fac: core40.Then(self.ctx, p, fac(), core37.Expand(self.ctx, p), k))
                leaf = core40.rollout(sim, p, ours, models[oname], horizon)
                M[i, j] = _v37._VALUE(leaf, p, phase)
                STATS["cells"] += 1
                if j > 0 and time.perf_counter() > deadline:
                    out_of_time = True
                    break
            if out_of_time:
                break
            filled_cols = j + 1
            if time.perf_counter() > deadline:
                break
        M = M[:, :max(1, filled_cols)]
        if "rm" in FLAGS:
            w = np.array([self.weights.get(o, 1.0) for o in cols[:filled_cols]])
            s = 0.5 * (M @ w) / w.sum() + 0.5 * M.min(1)
            best = int(np.argmax(s))
        else:
            sr, _sc, _v = core40.regret_matching(M)
            best = int(np.argmax(sr))
        name, fac = kept[best]
        self.macro, self.macro_name = fac(), name
        self.until = sim.step + k
        ms = 1000 * (time.perf_counter() - t0)
        STATS["plans"] += 1; STATS["cands"] += len(cands); STATS["kept"] += len(kept)
        STATS["ms"] += ms
        if LOG:
            with open(LOG, "a") as f:
                f.write(json.dumps({"tick": sim.step, "p": p, "phase": phase, "choice": name,
                                    "cands": [n for n, _ in cands], "kept": [n for n, _ in kept],
                                    "cols": filled_cols, "ms": round(ms, 1)}) + "\n")

    # -- asymmetric tactical step ---------------------------------------------
    def tactical(self, sim, a0):
        if "tactics" in FLAGS:
            return a0
        p = self.p
        cells = core40.critical_cells(sim, p, 4)
        if not cells:
            return a0
        STATS["tactic_checks"] += 1
        ours = [a0] + [a for a in core40.raw_moves(sim, cells) if a != a0] + ([0] if a0 else [])
        theirs = core40.enemy_replies(sim, p, limit=8)
        deadline = time.perf_counter() + 0.04
        best_a, best_v, v0 = a0, -INF, None
        for a in ours:
            worst = INF
            for b in theirs:
                s2 = sim.copy()
                s2.step2(*((a, b) if p == 0 else (b, a)))
                worst = min(worst, tactical_value(self.ctx, s2, p))
                if worst <= best_v:
                    break               # cannot beat the incumbent
            if a == a0:
                v0 = worst
            if worst > best_v:
                best_a, best_v = a, worst
            if time.perf_counter() > deadline:
                break
        if best_a != a0 and v0 is not None and best_v > v0 + 1.0:
            STATS["tactic_used"] += 1
            self.why = "tactic"
            return best_a
        return a0

    def act(self, sim):
        a = super().act(sim)
        a = a if a and a > 0 else 0
        if sim.gen[self.p] < 0:
            return a
        b = self.tactical(sim, a)
        return b if (b == 0 or sim.legal(self.p, b)) else a


Planner = Planner40


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
        mem = [int(state.tick), Planner40(ctx, player)]
        _MEM[key] = mem
    mem[0] = int(state.tick)
    sim = Sim.from_state(state, ctx.b)
    a = mem[1].act(sim)
    return int(a) if a and sim.legal(player, a) else 0
