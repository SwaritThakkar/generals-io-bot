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
    try:
        from strategies.wastar import search
        plan = search(state, player, epsilon=1.0)
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
