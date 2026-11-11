"""Small, conservative learned branch-pruning helper (never chooses a move).

The classifier is trained to recognize redundant *extra bucket branches*.
Core scripts and tactical branches are protected by the caller. Missing or
uncertain model output means keep the branch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURES = 20


def branch_features(state, player: int, action: int, core_actions=(),
                    shallow_delta: float = 0.0, shallow_gap: float = 0.0):
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    castle = np.asarray(state.castle, dtype=bool)
    general = np.asarray(state.general, dtype=bool)
    mountain = np.asarray(state.mountain, dtype=bool)
    h, w = owner.shape
    opp = 1 - int(player)
    gen_opp = np.argwhere(general & (owner == opp))
    gr, gc = map(int, gen_opp[0]) if len(gen_opp) else (0, 0)
    own_land = int(np.sum(owner == player))
    opp_land = int(np.sum(owner == opp))
    own_prod = int(np.sum((owner == player) & (castle | general)))
    opp_prod = int(np.sum((owner == opp) & (castle | general)))
    tick = int(getattr(state, "tick", getattr(state, "step", 0)))
    core = int(action in core_actions)
    if action <= 0:
        return np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, (own_land-opp_land)/500,
                           own_prod-opp_prod, tick/1000, core,
                           np.clip(shallow_delta/100, -5, 5),
                           np.clip(shallow_gap/100, -5, 5)], np.float32)
    src, k = divmod(int(action), 9)
    sr, sc = divmod(src, w)
    d = (k - 1) % 4
    dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[d]
    rr, cc = sr + dr, sc + dc
    if not (0 <= rr < h and 0 <= cc < w) or mountain[rr, cc]:
        return np.zeros((FEATURES,), np.float32)
    source_army = int(army[sr, sc])
    sent = source_army // 2 if k >= 5 else source_army - 1
    target_owner = int(owner[rr, cc])
    defenders = int(army[rr, cc]) if target_owner != player else 0
    capture_margin = sent - defenders
    source_role = (1 if general[sr, sc] else 0,
                   1 if castle[sr, sc] else 0,
                   1 if source_army >= 12 else 0)
    target_role = (1 if target_owner < 0 else 0,
                   1 if target_owner == opp else 0,
                   1 if castle[rr, cc] else 0,
                   1 if general[rr, cc] else 0)
    return np.asarray([
        0, *source_role, *target_role, int(k >= 5),
        np.log1p(source_army)/6, np.log1p(sent)/6,
        np.clip(capture_margin/80, -2, 2),
        np.clip((abs(rr-gr)+abs(cc-gc))/50, 0, 1),
        int(sr*w+sc in [int(x) for x in np.flatnonzero(general & (owner == player))]),
        (own_land-opp_land)/500, own_prod-opp_prod,
        tick/1000, core,
        np.clip(shallow_delta/100, -5, 5),
        np.clip(shallow_gap/100, -5, 5),
    ], dtype=np.float32)


def cheap_score(state, player: int, max_steps: int = 400):
    """Cheap successor score used as a cutoff feature (no route searches)."""
    owner=np.asarray(state.owner); army=np.asarray(state.army,dtype=np.int32)
    producer=np.asarray(state.castle,bool)|np.asarray(state.general,bool)
    me=int(player); opp=1-me
    alive=np.asarray(state.alive,bool)
    if not alive[me]: return -1e6
    if not alive[opp]: return 1e6
    land=float(np.sum(owner==me)-np.sum(owner==opp))
    armies=float(army[owner==me].sum()-army[owner==opp].sum())
    prod=float(np.sum((owner==me)&producer)-np.sum((owner==opp)&producer))
    remaining=max(0,int(max_steps)-int(state.step))
    return .20*land+.04*armies+.02*remaining*(.02*land+.50*prod)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def predict_prune_probability(model, features):
    if not model or not bool(model.get("enabled", False)):
        return 0.0
    x = (np.asarray(features, np.float32) - np.asarray(model["mean"], np.float32)) \
        / np.maximum(np.asarray(model["scale"], np.float32), 1e-4)
    w1, b1 = np.asarray(model["w1"], np.float32), np.asarray(model["b1"], np.float32)
    w2, b2 = np.asarray(model["w2"], np.float32), np.asarray(model["b2"], np.float32)
    hidden = np.tanh(x @ w1 + b1)
    return float(_sigmoid(hidden @ w2 + b2)[0])


def load_model(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None

