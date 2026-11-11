"""Stress policy: charge for a castle, then repeatedly all-in the enemy general."""
from strategies.wastar import search
from final import _castle_plan, _encode, DIRS
import numpy as np


def _route(state, anchor, target):
    owner = np.asarray(state.owner)
    mountain = np.asarray(state.mountain, bool)
    h, w = owner.shape
    seen = {anchor: None}
    q = [anchor]
    for r, c in q:
        if (r, c) == target:
            break
        for d, (dr, dc) in enumerate(DIRS):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w) or mountain[rr, cc]:
                continue
            if owner[rr, cc] >= 0 and owner[rr, cc] != owner[anchor] and (rr, cc) != target:
                continue
            if (rr, cc) not in seen:
                seen[(rr, cc)] = ((r, c), d)
                q.append((rr, cc))
    if target not in seen or target == anchor:
        return []
    actions, cur = [], target
    while cur != anchor:
        prev, d = seen[cur]
        actions.append(_encode(prev[0], prev[1], d, w))
        cur = prev
    return list(reversed(actions))[:3]


def act(state, player):
    plan = search(state, player, epsilon=1.0)
    if plan is not None and plan.target_kind == "enemy general":
        return int(plan.action)
    siege = _castle_plan(state, player)
    if siege is not None:
        anchor, target, _direction, wait, _steps = siege
        if wait > 0:
            return 0
        route = _route(state, anchor, target)
        if route:
            return route
    return 0
