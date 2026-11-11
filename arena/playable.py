"""Is a generated map playable? Used by the arena map cache and viewer.py.

A few generated maps wall a general in with mountains and castles (castles
start with a 40-50 garrison, so early on they are walls): in 200 cached maps,
seeds 9, 83 and 107 left one general 1-4 free tiles. A map is playable when

  * the two generals are connected by a path that avoids mountains AND castles,
  * each general can reach at least MIN_ROOM castle-free tiles.

Pure numpy / Python: no pygame, no JAX.
"""
from __future__ import annotations

from collections import deque

import numpy as np

MIN_ROOM = 60


def _reach(start: int, free, h: int, w: int, also=()) -> dict:
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        r, c = divmod(u, w)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                v = rr * w + cc
                if v not in dist and (free[v] or v in also):
                    dist[v] = dist[u] + 1
                    q.append(v)
    return dist


def check(mountain, castle, general_idx, h: int, w: int, min_room: int = MIN_ROOM) -> tuple[bool, str]:
    """(ok, reason). Arrays may be (h, w) or flat."""
    m = np.asarray(mountain, bool).reshape(-1)
    c = np.asarray(castle, bool).reshape(-1)
    g0, g1 = (int(x) for x in np.asarray(general_idx).reshape(-1)[:2])
    if g0 < 0 or g1 < 0:
        return False, "missing general"
    free = ~m & ~c
    d0 = _reach(g0, free, h, w, also=(g1,))
    if g1 not in d0:
        return False, "generals not connected without crossing a castle"
    d1 = _reach(g1, free, h, w, also=(g0,))
    room = min(len(d0), len(d1))
    if room < min_room:
        return False, f"only {room} free tiles reachable"
    return True, "ok"


def playable(mountain, castle, general_idx, h: int, w: int) -> bool:
    return check(mountain, castle, general_idx, h, w)[0]
