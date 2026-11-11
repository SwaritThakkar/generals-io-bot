"""Test opponent: bank at a producer, then queue the route to a castle."""
from collections import deque
import numpy as np
from final import _castle_plan, _encode, DIRS


def _route_actions(state, start, target):
    owner = np.asarray(state.owner)
    mountain = np.asarray(state.mountain, bool)
    h, w = owner.shape
    seen = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == target:
            break
        for d, (dr, dc) in enumerate(DIRS):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w) or mountain[rr, cc]:
                continue
            if owner[rr, cc] >= 0 and owner[rr, cc] != owner[start]:
                if (rr, cc) != target:
                    continue
            nxt = (rr, cc)
            if nxt not in seen:
                seen[nxt] = ((r, c), d)
                q.append(nxt)
    if target not in seen or target == start:
        return []
    rev = []
    cur = target
    while cur != start:
        prev, d = seen[cur]
        rev.append(_encode(prev[0], prev[1], d, w))
        cur = prev
    return list(reversed(rev))[:3]


def act(state, player):
    plan = _castle_plan(state, player)
    if plan is not None:
        anchor, target, _direction, wait, _steps = plan
        if wait > 0:
            return 0
        route = _route_actions(state, anchor, target)
        if route:
            return route
    from strategies.wastar import search
    plan = search(state, player, epsilon=1.0)
    if plan is not None:
        return [((r * state.w + c) * 9 + 1 + ([( -1,0),(1,0),(0,-1),(0,1)].index((nr-r,nc-c))))
                for (r,c),(nr,nc) in zip(plan.path[:3],plan.path[1:4])]
    return 0
