"""Search bot for the repo's perfect-information, simultaneous-move Generals.

The strategic core is the repo's weighted A* planner: it searches feasible
routes to high-value objectives (especially castles and the enemy general),
accounting for mountains, distance, garrisons, and stack attrition. When no
objective is affordable yet, this bot uses a multi-source BFS frontier march
to keep converting production into reachable territory.

Entry point: ``act(state, player)`` for the genmod-bot/1 protocol.
"""
from __future__ import annotations

import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _encode(r: int, c: int, d: int, width: int, half: bool = False) -> int:
    return (r * width + c) * 9 + 1 + d + (4 if half else 0)


def _frontier_dist(owner: np.ndarray, mountain: np.ndarray, player: int) -> np.ndarray:
    """Distance through our territory to the nearest non-owned passable cell."""
    h, w = owner.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    q = []
    for r, c in zip(*np.nonzero((owner != player) & ~mountain)):
        dist[r, c] = 0
        q.append((int(r), int(c)))
    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        nd = dist[r, c] + 1
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w and not mountain[rr, cc]
                    and owner[rr, cc] == player and nd < dist[rr, cc]):
                dist[rr, cc] = nd
                q.append((rr, cc))
    return dist


def _castle_plan(state, player: int):
    """Return (anchor, castle, first_direction, ticks_to_ready) for a siege.

    A stack on a general or owned castle can keep collecting production while
    it waits. Search those producer stacks against reachable castles, pricing
    the exact travel attrition and future +1 production ticks. The plan is an
    explicit macro action: hold the stack, then launch it when affordable.
    """
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    mountain = np.asarray(state.mountain, bool)
    castle = np.asarray(state.castle, bool)
    general = np.asarray(state.general, bool)
    h, w = owner.shape
    tick = int(state.tick)
    anchors = list(zip(*np.nonzero(((castle | general) & (owner == player)) & (army >= 1))))
    if not anchors:
        return None
    # Take neutral castles first. An enemy castle is a fallback only after no
    # neutral castle remains, since its garrison makes the race much costlier.
    targets = list(zip(*np.nonzero(castle & (owner < 0) & ~mountain)))
    if not targets:
        targets = list(zip(*np.nonzero(castle & (owner >= 0) & (owner != player) & ~mountain)))
    if not targets:
        return None

    best = None
    for ar, ac in anchors:
        # BFS permits owned and neutral ground, but never routes through an
        # enemy garrison. Track the first direction for one-step execution.
        dist = np.full((h, w), np.inf)
        first = np.full((h, w), -1, np.int8)
        dist[ar, ac] = 0
        q = [(int(ar), int(ac))]
        head = 0
        while head < len(q):
            r, c = q[head]; head += 1
            for d, (dr, dc) in enumerate(DIRS):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w) or mountain[rr, cc]:
                    continue
                if owner[rr, cc] >= 0 and owner[rr, cc] != player:
                    continue
                if dist[rr, cc] != np.inf:
                    continue
                dist[rr, cc] = dist[r, c] + 1
                first[rr, cc] = d if (r, c) == (ar, ac) else first[r, c]
                q.append((rr, cc))

        for tr, tc in targets:
            tr, tc = int(tr), int(tc)
            steps = int(dist[tr, tc]) if np.isfinite(dist[tr, tc]) else 10**6
            if steps <= 0 or steps >= 10**5:
                continue
            required = int(army[tr, tc]) + steps + 2
            # Count future fast production at this general/castle and land
            # production exactly on its configured cadence.
            projected = int(army[ar, ac])
            wait = 0
            while projected < required and wait < 160:
                wait += 1
                next_tick = tick + wait
                if next_tick % 2 == 0:
                    projected += 1
                if next_tick % 50 == 0:
                    projected += 1
            if projected < required:
                continue
            # Prefer the earliest castle-ready arrival. Mildly prefer the
            # general as the anchor when arrival times tie, because it is
            # harder for an opponent to remove its production.
            general_bonus = 0.01 if general[ar, ac] else 0.0
            rank = (wait + steps - general_bonus, -int(army[ar, ac]))
            item = (rank, (int(ar), int(ac)), (tr, tc), int(first[tr, tc]), wait)
            if best is None or item[0] < best[0]:
                best = item
    if best is None:
        return None
    _, anchor, target, direction, wait = best
    return anchor, target, direction, wait


def _march(state, player: int) -> int:
    owner = np.asarray(state.owner)
    army = np.asarray(state.army)
    mountain = np.asarray(state.mountain, bool)
    h, w = owner.shape
    movable = (owner == player) & (army >= 2)
    if not movable.any():
        return 0
    dist = _frontier_dist(owner, mountain, player)

    # Prefer the largest mobile stack that can advance toward unexplored land.
    sources = list(zip(*np.nonzero(movable)))
    sources.sort(key=lambda rc: (int(army[rc]), -float(dist[rc])), reverse=True)
    for r, c in sources:
        choices = []
        for d, (dr, dc) in enumerate(DIRS):
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w and not mountain[rr, cc]
                    and dist[rr, cc] < dist[r, c]):
                # Taking a neutral tile right now is worthwhile if possible.
                if owner[rr, cc] != player and int(army[r, c]) - 1 > int(army[rr, cc]):
                    return _encode(r, c, d, w)
                if owner[rr, cc] == player:
                    choices.append((float(dist[rr, cc]), d))
        if choices:
            _, d = min(choices)
            return _encode(r, c, d, w)
    return 0


def act(state, player: int) -> int:
    """Choose one legal action from the complete public board state."""
    siege = _castle_plan(state, player)
    try:
        from strategies.wastar import search
        plan = search(state, player, epsilon=1.0)
        if siege is not None:
            anchor, target, direction, wait = siege
            if wait > 0:
                # Let an independent stack keep playing if the A* move does not
                # disturb our collector. Otherwise PASS is the gather action.
                if plan is not None:
                    source = int(plan.action) // 9
                    if source != anchor[0] * state.w + anchor[1]:
                        return int(plan.action)
                return 0
            # At the affordability threshold, launch the collector on the
            # shortest castle route. Replan every tick in case combat changes.
            if direction >= 0:
                ar, ac = anchor
                return _encode(ar, ac, direction, state.w)
        if plan is not None:
            return int(plan.action)
    except (ImportError, AttributeError, ValueError):
        # Keep the agent usable if the search module is unavailable in a
        # restricted deployment; the BFS expansion policy remains legal.
        pass
    return _march(state, player)


if __name__ == "__main__":
    from server.bot_protocol import serve
    serve(act, name="Final Generals search bot")
