"""Search bot for the repo's perfect-information, simultaneous-move Generals.

The strategic core is the repo's weighted A* planner: it searches feasible
routes to high-value objectives (especially castles and the enemy general),
accounting for mountains, distance, garrisons, and stack attrition. When no
objective is affordable yet, this bot uses a multi-source BFS frontier march
to keep converting production into reachable territory.

Entry point: ``act(state, player)`` for the genmod-bot/1 protocol.
"""
from __future__ import annotations

import heapq
import copy
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

import genmod
from genmod import EnvConfig
from functools import partial

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SIEGE_CACHE = {}
_PROBCUT_PATH = Path(__file__).with_name("probcut_charge_fit_v30.json")
_PROBCUT_PAIRS = []
_PROBCUT_WRITES = 0
_PROBCUT_PRESET = (0.9057218679, 1.7322082656, 3.1967313677)
_ML_MODEL_PATH = Path(__file__).with_name("probcut_ml_v30.json")
_ML_MODEL = None
_ML_SAMPLES = []
_MACRO_SAMPLES = []
_MACRO_FORCE_ALL = False
_MACRO_LAST_CHOICE = None
EXPLICIT_CHARGE_MACROS = False
try:
    _ML_MODEL = json.loads(_ML_MODEL_PATH.read_text())
except (OSError, ValueError, TypeError):
    pass
try:
    _saved_fit = json.loads(_PROBCUT_PATH.read_text())
    _PROBCUT_PAIRS = [tuple(map(float, p)) for p in _saved_fit.get("pairs", [])][-500:]
except (OSError, ValueError, TypeError):
    pass


def _probcut_fit():
    """Fit deep = a*shallow+b and residual sigma from observed charge nodes."""
    if len(_PROBCUT_PAIRS) < 12:
        return _PROBCUT_PRESET
    data = np.asarray(_PROBCUT_PAIRS, dtype=np.float64)
    x, y = data[:, 0], data[:, 1]
    design = np.column_stack((x, np.ones_like(x)))
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (a * x + b)
    sigma = max(float(np.sqrt(np.mean(residual ** 2))), 1e-3)
    return float(a), float(b), sigma


def _probcut_observe(shallow, deep):
    global _PROBCUT_WRITES
    _PROBCUT_PAIRS.append((float(shallow), float(deep)))
    del _PROBCUT_PAIRS[:-500]
    _PROBCUT_WRITES += 1
    if _PROBCUT_WRITES % 16 and len(_PROBCUT_PAIRS) != 12:
        return
    try:
        _PROBCUT_PATH.write_text(json.dumps({"pairs": _PROBCUT_PAIRS}, separators=(",", ":")))
    except OSError:
        pass


def _ml_features(state, player: int, shallow: float, action: int):
    """Compact state/plan features for the offline nonlinear value model."""
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.float32)
    castle = np.asarray(state.castle, bool)
    general = np.asarray(state.general, bool)
    me, opp = int(player), 1 - int(player)
    own_prod = (castle | general) & (owner == me)
    opp_prod = (castle | general) & (owner == opp)
    own_stacks = army[(owner == me)]
    opp_stacks = army[(owner == opp)]
    own_plan = _charge_general_plans(state, me, limit=1)
    opp_plan = _charge_general_plans(state, opp, limit=1)
    op = own_plan[0] if own_plan else None
    ep = opp_plan[0] if opp_plan else None
    return np.asarray([
        np.clip(shallow, -100, 100),
        (int(own_prod.sum()) - int(opp_prod.sum())) / 8.0,
        (float(army[owner == me].sum()) - float(army[owner == opp].sum())) / 100.0,
        (float(own_stacks.max()) if own_stacks.size else 0.0) / 50.0,
        (float(opp_stacks.max()) if opp_stacks.size else 0.0) / 50.0,
        (float(op["wait"]) / 224.0) if op else 1.0,
        (float(op["travel"]) / 30.0) if op else 5.0,
        (float(ep["wait"]) / 224.0) if ep else 1.0,
        (float(ep["eta"]) / 250.0) if ep else 5.0,
        1.0 if int(action) == 0 else 0.0,
        float(getattr(state, "tick", getattr(state, "step", 0)))
        / max(1.0, float(getattr(state, "config", {}).get("maxTicks", 400))),
    ], dtype=np.float32)


def _ml_predict(features):
    """Return (deep-value mean, conservative error scale), or None pre-fit."""
    model = _ML_MODEL
    if (not model or int(model.get("version", 0)) != 1
            or not bool(model.get("use_ml", False))):
        return None
    x = (np.asarray(features, np.float32) - np.asarray(model["x_mean"], np.float32)) \
        / np.maximum(np.asarray(model["x_scale"], np.float32), 1e-5)
    w1, b1 = np.asarray(model["w1"], np.float32), np.asarray(model["b1"], np.float32)
    w2, b2 = np.asarray(model["w2"], np.float32), np.asarray(model["b2"], np.float32)
    out = np.tanh(x @ w1 + b1) @ w2 + b2
    mean = float(model.get("linear_a", 0.9057)) * float(features[0]) \
        + float(model.get("linear_b", 1.7322)) + float(out[0])
    # Held-out absolute residual quantile supplies a conservative prediction
    # interval. It is intentionally global until there are enough samples to
    # estimate feature-conditioned uncertainty reliably.
    sigma = float(model.get("residual_q90", 3.0))
    return mean, max(sigma, 1.0)


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


def _castle_plan(state, player: int, limit: int = 1):
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
    tick_value = getattr(state, "tick", None)
    tick = int(state.step) if tick_value is None else int(tick_value)
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

    plans = []
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
            base = int(army[ar, ac])
            def projected_after(dt):
                end = tick + dt
                return base + end // 2 - tick // 2 + end // 50 - tick // 50
            lo, hi = 0, 160
            while lo < hi:
                mid = (lo + hi) // 2
                if projected_after(mid) >= required:
                    hi = mid
                else:
                    lo = mid + 1
            wait = lo
            projected = projected_after(wait)
            if projected < required:
                continue
            # Prefer the earliest castle-ready arrival. Mildly prefer the
            # general as the anchor when arrival times tie, because it is
            # harder for an opponent to remove its production.
            general_bonus = 0.01 if general[ar, ac] else 0.0
            rank = (wait + steps - general_bonus, -int(army[ar, ac]))
            item = (rank, (int(ar), int(ac)), (tr, tc), int(first[tr, tc]), wait, steps)
            plans.append(item)
    if not plans:
        return None
    plans.sort(key=lambda item: item[0])
    selected, seen_targets = [], set()
    for item in plans:
        target = item[2]
        if target not in seen_targets:
            selected.append(item)
            seen_targets.add(target)
        if len(selected) >= max(1, int(limit)):
            break
    result = [(anchor, target, direction, wait, steps)
              for _rank, anchor, target, direction, wait, steps in selected]
    return result[0] if limit == 1 else result


def _castle_route_actions(state, player: int, anchor, target, limit: int = 32):
    """Return a queueable shortest route through owned/neutral passable cells."""
    owner = np.asarray(state.owner)
    mountain = np.asarray(state.mountain, bool)
    h, w = owner.shape
    start, goal = tuple(map(int, anchor)), tuple(map(int, target))
    prev = {start: None}
    q = [start]
    for r, c in q:
        if (r, c) == goal:
            break
        for d, (dr, dc) in enumerate(DIRS):
            rr, cc = r + dr, c + dc
            cell = (rr, cc)
            if (not (0 <= rr < h and 0 <= cc < w) or mountain[cell]
                    or (owner[cell] >= 0 and owner[cell] != player and cell != goal)
                    or cell in prev):
                continue
            prev[cell] = ((r, c), d)
            q.append(cell)
    if goal not in prev or goal == start:
        return []
    steps = []
    cur = goal
    while cur != start:
        prior, d = prev[cur]
        steps.append(_encode(prior[0], prior[1], d, w))
        cur = prior
    return list(reversed(steps))[:limit]


def _charge_route_actions(plan, state, player: int, limit: int = 6):
    """Queue a short validated prefix of a ready general all-in path."""
    path = plan.get("path", [])
    w = state.w
    actions = []
    for (r, c), (nr, nc) in zip(path, path[1:]):
        try:
            d = DIRS.index((int(nr) - int(r), int(nc) - int(c)))
        except ValueError:
            break
        actions.append(_encode(int(r), int(c), d, w))
        if len(actions) >= limit:
            break
    return actions


def _incoming_siege(state, defender: int):
    """Estimate the fastest feasible all-in stack route to a general.

    Reverse Dijkstra prices each step by one unit of movement attrition plus
    the garrison on defender or neutral cells. It then checks every enemy
    stack, including a stack still charging on an enemy general/castle, and
    estimates when that producer could launch the shortest feasible attack.
    Returns ``(eta, source, path, wait, travel)``. ETA includes charging time.
    """
    key = (id(state.owner), id(state.army), int(defender))
    if key in _SIEGE_CACHE:
        return _SIEGE_CACHE[key]
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    mountain = np.asarray(state.mountain, bool)
    castle = np.asarray(state.castle, bool)
    general = np.asarray(state.general, bool)
    h, w = owner.shape
    opp = 1 - int(defender)
    hits = np.argwhere(general & (owner == defender))
    if not len(hits):
        result = (10**6, None, [], 10**6, 0)
        _SIEGE_CACHE[key] = result
        return result
    gr, gc = map(int, hits[0])
    inf = 10**9
    cost = np.full((h, w), inf, np.int32)
    ticks = np.full((h, w), inf, np.int32)
    nxt = np.full((h, w, 2), -1, np.int32)
    cost[gr, gc] = 0
    ticks[gr, gc] = 0
    heap = [(0, 0, gr, gc)]
    while heap:
        cst, tks, r, c = heapq.heappop(heap)
        if cst != cost[r, c] or tks != ticks[r, c]:
            continue
        # Reverse edge: a stack at (rr,cc) enters (r,c) on its way to the goal.
        enter = 1 + (int(army[r, c]) if owner[r, c] == defender or owner[r, c] < 0 else 0)
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w) or mountain[rr, cc]:
                continue
            nc, nt = cst + enter, tks + 1
            if nc < cost[rr, cc] or (nc == cost[rr, cc] and nt < ticks[rr, cc]):
                cost[rr, cc], ticks[rr, cc] = nc, nt
                nxt[rr, cc] = (r, c)
                heapq.heappush(heap, (nc, nt, rr, cc))

    tick_value = getattr(state, "tick", None)
    now = int(state.step) if tick_value is None else int(tick_value)
    enemy_producer = (castle | general) & (owner == opp)
    best = (10**6, None, [], 10**6, 0)
    for sr, sc in zip(*np.nonzero((owner == opp) & (army >= 2))):
        sr, sc = int(sr), int(sc)
        if cost[sr, sc] >= inf:
            continue
        travel = int(ticks[sr, sc])
        # Add the defender general's deterministic growth before arrival.
        required = int(cost[sr, sc]) + 1 + travel // 2
        stack = int(army[sr, sc])
        wait = 0
        producer = bool(enemy_producer[sr, sc])
        while stack < required and producer and wait < 240:
            wait += 1
            ntick = now + wait
            if ntick % 2 == 0:
                stack += 1
            if ntick % 50 == 0:
                stack += 1
        if stack < required:
            continue
        eta = wait + travel
        if eta >= best[0]:
            continue
        path = [(sr, sc)]
        cur = (sr, sc)
        guard = 0
        while cur != (gr, gc) and guard <= h * w:
            rr, cc = map(int, nxt[cur[0], cur[1]])
            if rr < 0:
                path = []
                break
            cur = (rr, cc)
            path.append(cur)
            guard += 1
        if path:
            best = (eta, (sr, sc), path, wait, travel)
    _SIEGE_CACHE[key] = best
    return best


def _likely_siege(state, defender: int) -> bool:
    """Cheap gate before running the weighted siege-distance search."""
    owner = np.asarray(state.owner)
    army = np.asarray(state.army)
    castle = np.asarray(state.castle, bool)
    general = np.asarray(state.general, bool)
    mountain = np.asarray(state.mountain, bool)
    hits = np.argwhere(general & (owner == defender))
    if not len(hits):
        return False
    gr, gc = map(int, hits[0])
    opp = 1 - int(defender)
    rr, cc = np.indices(owner.shape)
    distance = np.abs(rr - gr) + np.abs(cc - gc)
    armed = (owner == opp) & (army >= 3) & ~mountain
    if np.any(armed & (distance <= 24)):
        return True
    charging = (owner == opp) & (army >= 2) & (castle | general) & ~mountain
    return bool(np.any(charging & (distance <= 40)))


def _siege_source_waiting(state, source, defender: int) -> bool:
    if source is None:
        return False
    r, c = source
    return bool((np.asarray(state.owner)[r, c] == 1 - defender)
                and (np.asarray(state.castle, bool)[r, c]
                     or np.asarray(state.general, bool)[r, c]))


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


def _defense_action(state, player: int, assessment=None) -> int:
    """Reinforce or intercept the fastest all-in stack headed for our general."""
    if not _likely_siege(state, player):
        return 0
    eta, source, path, wait, travel = assessment if assessment is not None else _incoming_siege(state, player)
    if source is None or eta > 120:
        return 0
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    mountain = np.asarray(state.mountain, bool)
    general = np.asarray(state.general, bool)
    h, w = owner.shape
    gen_hits = np.argwhere(general & (owner == player))
    if not len(gen_hits):
        return 0
    gr, gc = map(int, gen_hits[0])

    # First try to kill the siege stack or seize a cell it must cross.
    for tr, tc in path[:-1]:
        tr, tc = int(tr), int(tc)
        best_attack = None
        for d, (dr, dc) in enumerate(DIRS):
            sr, sc = tr - dr, tc - dc
            if not (0 <= sr < h and 0 <= sc < w):
                continue
            if owner[sr, sc] != player or army[sr, sc] < 2 or mountain[tr, tc]:
                continue
            sent = int(army[sr, sc]) - 1
            # Captures are ideal; otherwise a heavy trade still weakens the
            # all-in stack before it reaches the general.
            target_force = int(army[tr, tc]) if owner[tr, tc] != player else 0
            score = (1000 if sent > target_force else 0) + min(sent, target_force)
            if best_attack is None or score > best_attack[0]:
                best_attack = (score, _encode(sr, sc, d, w))
        if best_attack is not None and best_attack[0] >= 1000:
            return int(best_attack[1])
        # While the enemy is still gathering, a large equal-force trade on
        # its producer/path can destroy the launch timing even if it cannot
        # capture the tile immediately. Do this only when the stack removes a
        # substantial fraction of the target army.
        if (wait > 0 and eta <= 240 and best_attack is not None
                and best_attack[0] >= max(8, int(army[tr, tc]) * 0.60)):
            return int(best_attack[1])

    # Otherwise move a friendly stack toward a cell adjacent to the general,
    # preferring the fastest available reinforcement.
    dist = np.full((h, w), np.inf, dtype=np.float32)
    dist[gr, gc] = 0
    q = [(gr, gc)]
    head = 0
    while head < len(q):
        r, c = q[head]; head += 1
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w and not mountain[rr, cc]
                    and owner[rr, cc] == player and dist[rr, cc] == np.inf):
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    choices = []
    for sr, sc in zip(*np.nonzero((owner == player) & (army >= 2))):
        sr, sc = int(sr), int(sc)
        if (sr, sc) == (gr, gc) or not np.isfinite(dist[sr, sc]):
            continue
        for d, (dr, dc) in enumerate(DIRS):
            rr, cc = sr + dr, sc + dc
            if (0 <= rr < h and 0 <= cc < w and owner[rr, cc] == player
                    and dist[rr, cc] < dist[sr, sc]):
                # Large armies arriving sooner are the best reinforcements.
                choices.append((float(dist[sr, sc]), -int(army[sr, sc]),
                                _encode(sr, sc, d, w)))
    return int(min(choices)[2]) if choices else 0


def _charge_general_plans(state, player: int, limit: int = 3):
    """Producer-to-enemy-general plans with an explicit wait-then-charge phase."""
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    mountain = np.asarray(state.mountain, bool)
    castle = np.asarray(state.castle, bool)
    general = np.asarray(state.general, bool)
    h, w = owner.shape
    opp = 1 - player
    goals = np.argwhere(general & (owner == opp))
    if not len(goals):
        return []
    gr, gc = map(int, goals[0])
    # Reverse Dijkstra: charge one unit per step and pay every garrison on the
    # route. The enemy general's current garrison is charged at the goal.
    inf = 10**9
    cost = np.full((h, w), inf, np.int32)
    ticks = np.full((h, w), inf, np.int32)
    nxt = np.full((h, w, 2), -1, np.int32)
    cost[gr, gc] = 0
    ticks[gr, gc] = 0
    heap = [(int(cost[gr, gc]), 0, gr, gc)]
    while heap:
        cst, tks, r, c = heapq.heappop(heap)
        if cst != cost[r, c] or tks != ticks[r, c]:
            continue
        enter = 1 + (int(army[r, c]) if owner[r, c] != player else 0)
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w) or mountain[rr, cc]:
                continue
            nc, nt = cst + enter, tks + 1
            if nc < cost[rr, cc] or (nc == cost[rr, cc] and nt < ticks[rr, cc]):
                cost[rr, cc], ticks[rr, cc] = nc, nt
                nxt[rr, cc] = (r, c)
                heapq.heappush(heap, (nc, nt, rr, cc))

    now = int(getattr(state, "tick", getattr(state, "step", 0)))
    producers = (castle | general) & (owner == player) & (army >= 1)
    plans = []
    for sr0, sc0 in zip(*np.nonzero(producers)):
        sr, sc = int(sr0), int(sc0)
        if cost[sr, sc] >= inf:
            continue
        travel = int(ticks[sr, sc])
        required = int(cost[sr, sc]) + travel // 2
        base = int(army[sr, sc])
        lo, hi = 0, 240
        while lo < hi:
            mid = (lo + hi) // 2
            end = now + mid
            projected = base + end // 2 - now // 2 + end // 50 - now // 50
            if projected >= required:
                hi = mid
            else:
                lo = mid + 1
        wait = lo
        end = now + wait
        projected = base + end // 2 - now // 2 + end // 50 - now // 50
        if projected < required:
            continue
        path = [(sr, sc)]
        cur = (sr, sc)
        guard = 0
        while cur != (gr, gc) and guard <= h * w:
            rr, cc = map(int, nxt[cur[0], cur[1]])
            if rr < 0:
                path = []
                break
            cur = (rr, cc)
            path.append(cur)
            guard += 1
        if len(path) >= 2:
            plans.append({"source": (sr, sc), "path": path, "wait": wait,
                          "travel": travel, "eta": wait + travel,
                          "army_at_launch": projected})
    plans.sort(key=lambda p: (p["eta"], -p["army_at_launch"]))
    return plans[:limit]


def _charge_action(state, player: int, plan, elapsed: int = 0) -> int:
    """Next action in a fixed wait-then-follow-the-shortest-path macro."""
    if elapsed < plan["wait"]:
        return 0
    path = plan["path"]
    i = elapsed - plan["wait"]
    if i >= len(path) - 1:
        return 0
    r, c = path[i]
    nr, nc = path[i + 1]
    owner = np.asarray(state.owner)
    army = np.asarray(state.army)
    if owner[r, c] != player or army[r, c] < 2:
        return 0
    try:
        d = DIRS.index((nr - r, nc - c))
    except ValueError:
        return 0
    return _encode(r, c, d, state.w)


def _counter_launch_action(state, defender: int, assessment=None) -> int:
    """Enter the launch source on the tick an all-in leaves one unit there."""
    eta, source, path, wait, _travel = (assessment if assessment is not None
                                        else _incoming_siege(state, defender))
    if source is None or eta > 100 or len(path) < 2:
        return 0
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    mountain = np.asarray(state.mountain, bool)
    h, w = owner.shape
    tr, tc = map(int, source)
    # When the opposing stack is still collecting, seize its launch tile if
    # we can beat the entire garrison now. Waiting until it launches forfeits
    # the only cheap chance to interrupt the macro.
    if wait > 0:
        choices = []
        for d, (dr, dc) in enumerate(DIRS):
            sr, sc = tr - dr, tc - dc
            if (0 <= sr < h and 0 <= sc < w and owner[sr, sc] == defender
                    and army[sr, sc] - 1 > int(army[tr, tc]) and not mountain[tr, tc]):
                choices.append((int(army[sr, sc]), _encode(sr, sc, d, w)))
        return int(max(choices)[1]) if choices else 0
    choices = []
    for d, (dr, dc) in enumerate(DIRS):
        sr, sc = tr - dr, tc - dc
        if (0 <= sr < h and 0 <= sc < w and owner[sr, sc] == defender
                and army[sr, sc] >= 3 and not mountain[tr, tc]):
            # Two attacking units beat the single unit left behind by an
            # all-in move; the matrix also evaluates the opponent staying.
            choices.append((int(army[sr, sc]) - 1, _encode(sr, sc, d, w)))
    return int(max(choices)[1]) if choices else 0


@partial(jax.jit, static_argnames=("cfg",))
def _joint_successors(st, joint_actions, cfg):
    """Vectorized exact transitions for the pruned simultaneous action matrix."""
    return jax.vmap(lambda a: genmod.step(st, a, cfg)[0])(joint_actions)


_WAIT_HORIZON = 224


@partial(jax.jit, static_argnames=("cfg",))
def _rollout_macro_pairs(st, action_schedule, cfg):
    """Batch exact game ticks for scripted simultaneous macro-action pairs.

    action_schedule has shape (ticks, macro_pairs, players). JAX places these
    batched transitions on its default device, including a GPU when available.
    """
    n_pairs = action_schedule.shape[1]
    batched = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_pairs,) + x.shape), st)
    step_batch = jax.vmap(lambda s, a: genmod.step(s, a, cfg)[0])
    def body(carry, actions):
        return step_batch(carry, actions), None
    final, _ = jax.lax.scan(body, batched, action_schedule)
    return final


def _macro_option(state, player: int, name: str, wait: int, route, target_kind=""):
    """Build a fixed hold-then-route schedule for one durative script."""
    schedule = np.zeros((_WAIT_HORIZON,), np.int32)
    route = [int(a) for a in route if int(a) > 0]
    start = max(0, min(_WAIT_HORIZON, int(wait)))
    n = min(len(route), _WAIT_HORIZON - start)
    if n:
        schedule[start:start + n] = route[:n]
    first = route[0] if start == 0 and route else 0
    return {"name": name, "wait": int(wait), "schedule": schedule, "first": int(first),
            "route": route, "target_kind": target_kind}


def _macro_options(state, player: int, baseline):
    """Small script portfolio: pass, immediate tactic, castle, general charge."""
    options = [_macro_option(state, player, "hold", 0, [])]
    action = next((int(a) for a in baseline if int(a) > 0), 0)
    if action:
        options.append(_macro_option(state, player, "tactical", 0, [action]))
    # Give expansion the same durative rollout treatment as waiting. The old
    # one-step fallback made it structurally impossible for a castle route to
    # compete with a 96-tick wait/charge script.
    from strategies.wastar import search
    expansion = search(state, player, epsilon=1.0)
    if expansion is not None and len(expansion.path) > 1:
        route = []
        for (r, c), (nr, nc) in zip(expansion.path, expansion.path[1:]):
            try:
                d = DIRS.index((int(nr) - int(r), int(nc) - int(c)))
            except ValueError:
                break
            route.append(_encode(int(r), int(c), d, state.w))
        if route:
            options.append(_macro_option(state, player, "expand", 0, route,
                                         getattr(expansion, "target_kind", "")))
    castle_plans = _castle_plan(state, player, limit=2)
    for castle_index, castle_plan in enumerate(castle_plans or []):
        anchor, target, _d, wait, _travel = castle_plan
        route = _castle_route_actions(state, player, anchor, target, limit=_WAIT_HORIZON)
        if route:
            options.append(_macro_option(state, player, f"wait-castle-{castle_index}", wait, route))
    general_plans = _charge_general_plans(state, player, limit=1)
    if general_plans:
        plan = general_plans[0]
        route = _charge_route_actions(plan, state, player, limit=_WAIT_HORIZON)
        if route:
            options.append(_macro_option(state, player, "wait-general", plan["wait"], route))
    # Identical schedules add no strategic option and only inflate the matrix.
    unique, seen = [], set()
    for item in options:
        key = item["schedule"].tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:6]


def _macro_score(st, player: int, cfg) -> float:
    """Value producers for their future income across the remaining match."""
    score = _score_successor(st, player, cfg)
    owner = np.asarray(st.owner)
    prod = np.asarray(st.castle, bool) | np.asarray(st.general, bool)
    me_prod = int(np.sum((owner == player) & prod))
    opp_prod = int(np.sum((owner == 1 - player) & prod))
    remaining = max(0, cfg.max_steps - int(st.step))
    # A castle produces about half a unit each tick. The base evaluator's
    # discounted producer term was too small to make early castle captures
    # compete with a general all-in over this tactical horizon.
    return float(score + 0.04 * remaining * 0.5 * (me_prod - opp_prod))


def _macro_wait_choice(state, player: int, mine, theirs) -> int:
    """ProbCut over simultaneous wait/launch scripts (up to 96 ticks)."""
    cfg = state.env_config()
    st = state.to_genmod_state(cfg)
    options = [_macro_options(state, side, mine if side == player else theirs)
               for side in (player, 1 - player)]
    first_pairs, schedules = [], []
    for mine_opt in options[0]:
        for their_opt in options[1]:
            joint_first = np.zeros((cfg.num_players,), np.int32)
            joint_first[player] = mine_opt["first"]
            joint_first[1 - player] = their_opt["first"]
            first_pairs.append(joint_first)
            by_player = [None] * cfg.num_players
            by_player[player] = mine_opt["schedule"]
            by_player[1 - player] = their_opt["schedule"]
            schedules.append(np.stack(by_player, axis=-1))
    shape = (len(options[0]), len(options[1]))
    shallow_batch = _joint_successors(st, jnp.asarray(first_pairs, jnp.int32), cfg)
    shallow_states = [jax.tree.map(lambda x, i=i: x[i], shallow_batch)
                      for i in range(len(first_pairs))]
    shallow = np.asarray([_macro_score(s, player, cfg) for s in shallow_states])
    shallow_rows = shallow.reshape(shape).min(axis=1)
    features = [_ml_features(state, player, shallow_rows[i], options[0][i]["first"])
                for i in range(len(options[0]))]
    predictions = [_ml_predict(f) for f in features]
    order = sorted(range(len(options[0])),
                   key=lambda i: predictions[i][0] if predictions[i] else shallow_rows[i],
                   reverse=True)
    estimates = shallow_rows.astype(np.float64).copy()
    alpha = -np.inf
    deepened = 0
    for i in order:
        pred = predictions[i]
        estimate = pred[0] if pred else float(shallow_rows[i])
        sigma = pred[1] if pred else 0.0
        contact = _move_contact(options[0][i]["first"], state)
        captures_general = False
        if contact is not None:
            _src, dst = contact
            rr, cc = divmod(dst, state.w)
            captures_general = bool(np.asarray(state.general, bool)[rr, cc]
                                    and np.asarray(state.owner)[rr, cc] != player)
        critical_macro = (options[0][i]["name"].startswith("wait-castle")
                          or options[0][i]["name"] == "wait-general"
                          or (options[0][i]["name"] == "expand"
                              and options[0][i]["target_kind"] in
                              ("enemy general", "enemy city", "neutral city"))
                          or captures_general or abs(float(shallow_rows[i])) >= 1e8)
        if (not _MACRO_FORCE_ALL and not critical_macro
                and np.isfinite(alpha) and pred is not None):
            if estimate + 1.5 * sigma < alpha:
                estimates[i] = estimate
                continue
        if not _MACRO_FORCE_ALL and deepened >= 3 and not critical_macro:
            estimates[i] = estimate
            continue
        row_indices = [i * shape[1] + j for j in range(shape[1])]
        action_schedule = np.stack([schedules[j] for j in row_indices], axis=1)
        final_batch = _rollout_macro_pairs(st, jnp.asarray(action_schedule), cfg)
        final_states = [jax.tree.map(lambda x, k=k: x[k], final_batch)
                        for k in range(shape[1])]
        deep_values = np.asarray([_macro_score(s, player, cfg) for s in final_states], np.float64)
        deep_row = float(np.min(deep_values))
        estimates[i] = deep_row
        # General captures are exact terminal facts, not regression targets.
        # ProbCut must keep those tactical rows in the searched set.
        if abs(deep_row) < 1e7:
            _MACRO_SAMPLES.append((features[i].tolist(), deep_row))
            del _MACRO_SAMPLES[:-5000]
        alpha = max(alpha, deep_row)
        deepened += 1
    best = int(np.argmax(estimates))
    selected = options[0][best]
    global _MACRO_LAST_CHOICE
    _MACRO_LAST_CHOICE = {
        "player": int(player), "selected": selected["name"],
        "rows": [(opt["name"], float(estimates[i]), float(shallow_rows[i]))
                 for i, opt in enumerate(options[0])],
    }
    # A launch queue is short enough to get a fresh search before contact; a
    # wait macro returns PASS and is reconsidered after the next production tick.
    if selected["wait"] > 0:
        return 0
    if selected["route"]:
        return selected["route"][:6]
    return int(selected["first"])


def _deep_charge_minimax(state, player: int, cfg, depth: int = 2) -> float:
    """Reduced simultaneous search with a deeper principal variation."""
    from server.local_server import to_bot_state
    state = to_bot_state(state, cfg, player)
    actions = []
    for side in (player, 1 - player):
        plans = _charge_general_plans(state, side, limit=1)
        charge = _charge_action(state, side, plans[0]) if plans else 0
        threat = _incoming_siege(state, side) if _likely_siege(state, side) else None
        defense = _defense_action(state, side, threat) if threat is not None else 0
        # Keep an expansion/attack reply alongside the macro continuation and
        # pass; otherwise a charge node can make both players freeze in place.
        response = defense or _march(state, side)
        actions.append(list(dict.fromkeys((charge, response, 0)))[:3])
    st = state.to_genmod_state(cfg)
    pairs = []
    for mine in actions[0]:
        for theirs in actions[1]:
            joint = [0] * cfg.num_players
            joint[player], joint[1 - player] = int(mine), int(theirs)
            pairs.append(joint)
    successors = _joint_successors(st, jnp.asarray(pairs, jnp.int32), cfg)
    states = [jax.tree.map(lambda x, i=i: x[i], successors) for i in range(len(pairs))]
    values = np.asarray([_score_successor(s, player, cfg) for s in states])
    matrix = values.reshape(len(actions[0]), len(actions[1]))
    row_values = matrix.min(axis=1)
    best_row = int(np.argmax(row_values))
    value = float(row_values[best_row])
    if depth <= 1 or value >= 1e8:
        return value
    # Spend the extra ply on the current minimax principal variation: our best
    # continuation against the opponent's strongest reply. This adds another
    # B without multiplying every response branch by the full width.
    worst_col = int(np.argmin(matrix[best_row]))
    pv_state = states[best_row * matrix.shape[1] + worst_col]
    return _deep_charge_minimax(pv_state, player, cfg, depth - 1)


def _plan_candidates(state, player: int):
    """Pruned macro-action set with a separate response to a general siege."""
    from strategies.wastar import search

    plan = search(state, player, epsilon=1.0)
    siege = _castle_plan(state, player)
    if siege is None:
        castle_action = 0
    else:
        anchor, _target, direction, wait, _steps = siege
        castle_action = (0 if wait > 0 else
                         _encode(anchor[0], anchor[1], direction, state.w))
    astar_action = int(plan.action) if plan is not None else 0
    march_action = _march(state, player)
    defense_action = _defense_action(state, player)
    incoming = _incoming_siege(state, player) if _likely_siege(state, player) else None
    counter_action = (_counter_launch_action(state, player, incoming)
                      if incoming is not None else 0)
    charge_plans = _charge_general_plans(state, player, limit=1)
    charge_action = (_charge_action(state, player, charge_plans[0])
                     if charge_plans else 0)
    # PASS is a first-class collect-and-wait option in the response matrix.
    return [astar_action, castle_action, march_action, defense_action,
            counter_action, charge_action, 0]


def _move_contact(action: int, state):
    """Return (source, destination) for a legal encoded move, else None."""
    if action <= 0 or action % 9 == 0:
        return None
    h, w = state.owner.shape
    src = action // 9
    r, c = divmod(src, w)
    d = (action - 1) % 4
    dr, dc = DIRS[d]
    rr, cc = r + dr, c + dc
    if not (0 <= rr < h and 0 <= cc < w):
        return None
    return src, rr * w + cc


def _has_conflict(state, mine, theirs) -> bool:
    """Only pay for the simultaneous response matrix near an actual collision."""
    _h, width = state.owner.shape
    contacts0 = [_move_contact(a, state) for a in mine]
    contacts1 = [_move_contact(a, state) for a in theirs]
    gen = np.asarray(state.general, bool)
    for x in contacts0:
        if x is None:
            continue
        s0, d0 = x
        r0, c0 = divmod(d0, width)
        if gen[r0, c0]:
            return True
        for y in contacts1:
            if y is None:
                continue
            s1, d1 = y
            if d0 == d1 or (d0 == s1 and d1 == s0):
                return True
            r1, c1 = divmod(d1, width)
            if gen[r1, c1]:
                return True
    for defender in range(state.n):
        if _likely_siege(state, defender):
            eta, source, _path, wait, _travel = _incoming_siege(state, defender)
            # Charging on a producer is a strategic wait branch. Do not pay
            # for a full response matrix until that stack actually launches.
            if eta <= 40 or (source is not None and wait <= 40 and _siege_source_waiting(state, source, defender)):
                return True
    return False


def _safe_prefix(plan, state, player: int):
    """Queue a short, revalidated path prefix when the route is stable.

    Local server checks every queued source for ownership and army before it
    executes. Keep prefixes short near enemy territory so the bot can replan
    quickly after a simultaneous move changes the board.
    """
    path = getattr(plan, "path", None)
    if not path or len(path) < 2:
        return int(plan.action)
    owner = np.asarray(state.owner)
    _h, w = owner.shape
    # Longer queues buy computation time while the stack follows an owned,
    # low-contact corridor. Stop early for a direct general/castle assault.
    fragile = getattr(plan, "target_kind", "") in ("enemy general", "enemy city")
    prefix = 1 if fragile else 6
    opponent_cells = np.argwhere((owner >= 0) & (owner != player))
    safe_prefix = 0
    for _idx, ((r, c), (nr, nc)) in enumerate(zip(path, path[1:])):
        if _idx >= prefix or owner[int(nr), int(nc)] != player:
            break
        if len(opponent_cells) and np.min(
                np.abs(opponent_cells[:, 0] - int(nr))
                + np.abs(opponent_cells[:, 1] - int(nc))) <= 4:
            break
        safe_prefix += 1
    prefix = min(prefix, max(1, safe_prefix))
    actions = []
    for (r, c), (nr, nc) in zip(path[:prefix], path[1:prefix + 1]):
        delta = (int(nr) - int(r), int(nc) - int(c))
        try:
            d = DIRS.index(delta)
        except ValueError:
            break
        actions.append(_encode(int(r), int(c), d, w))
    return actions or int(plan.action)


def _castle_eta(state, player: int) -> float:
    plan = _castle_plan(state, player)
    if plan is None:
        return 300.0
    return float(plan[3] + plan[4])


def _score_successor(st, player: int, cfg) -> float:
    """Fast state evaluation with production, safety, and castle readiness."""
    owner = np.asarray(st.owner)
    army = np.asarray(st.army)
    castle = np.asarray(st.castle)
    general = np.asarray(st.general)
    me = int(player)
    opps = [p for p in range(cfg.num_players) if p != me and bool(np.asarray(st.alive)[p])]
    if not opps:
        return 1e9
    opp = opps[0]
    alive = np.asarray(st.alive)
    if not alive[me]:
        return -1e9
    if not alive[opp]:
        return 1e9
    land_me, land_opp = int((owner == me).sum()), int((owner == opp).sum())
    army_me, army_opp = int(army[owner == me].sum()), int(army[owner == opp].sum())
    prod = castle | general
    prod_me = int(((owner == me) & prod).sum())
    prod_opp = int(((owner == opp) & prod).sum())
    remaining = max(0, cfg.max_steps - int(st.step))
    score = (0.20 * (land_me - land_opp)
             + 0.04 * (army_me - army_opp)
             + 0.02 * remaining * (0.02 * (land_me - land_opp)
                                    + 0.50 * (prod_me - prod_opp)))
    # A producer stack close to an affordable castle is useful even before the
    # capture. This makes PASS/collect compete with expansion in the tree.
    # Keep siege readiness as a real feature, but small enough that it cannot
    # lock the policy into waiting when expansion or a tactical reply is better.
    score += 0.01 * (_castle_eta(st, opp) - _castle_eta(st, me))
    # Race value for an all-in general attack. A short enemy ETA is a serious
    # liability even before its stack reaches an adjacent tile. Conversely,
    # if our own stack can reach their general sooner, accelerate the attack.
    if _likely_siege(st, me):
        eta_to_me, source, _, wait_to_me, _ = _incoming_siege(st, me)
        if eta_to_me <= 100 and source is not None:
            er, ec = source
            enemy_waiting = bool((castle[er, ec] | general[er, ec]) and owner[er, ec] == opp)
            my_gen = np.argwhere(general & (owner == me))
            if enemy_waiting and len(my_gen):
                gr, gc = map(int, my_gen[0])
                score += 2.0 * min(80, int(army[gr, gc]))
            elif eta_to_me <= 100:
                score -= 2.0 * (101 - eta_to_me)
            if eta_to_me <= 4:
                score -= 100_000
    if _likely_siege(st, opp):
        eta_to_opp, _, _, wait_to_opp, _ = _incoming_siege(st, opp)
        if eta_to_opp <= 100:
            # Include producer charging time in the value of an attack. The
            # older gate rewarded only already-launched stacks and therefore
            # made a safe wait-then-charge plan invisible to minimax.
            readiness = 1.0 if wait_to_opp == 0 else 0.75
            score += readiness * (101 - eta_to_opp)
    # Reward a mobile stack for making progress toward the frontier. Without
    # this feature, a one-ply evaluator sees no value in moving through our own
    # territory and can PASS forever while an opponent expands.
    def stack_frontier_distance(side):
        movable = (owner == side) & (army >= 2)
        if not movable.any():
            return 100.0
        dist = _frontier_dist(owner, np.asarray(st.mountain, bool), side)
        rows, cols = np.nonzero(movable)
        order = np.argsort(army[rows, cols])[::-1]
        for ix in order:
            d = float(dist[rows[ix], cols[ix]])
            if np.isfinite(d):
                return d
        return 100.0
    score += 0.25 * (stack_frontier_distance(opp) - stack_frontier_distance(me))
    # Never accept a position where an adjacent enemy can take our general on
    # its next move. Exact one-tick threats matter more than material.
    my_gen = np.argwhere(general & (owner == me))
    if len(my_gen):
        gr, gc = map(int, my_gen[0])
        garrison = int(army[gr, gc])
        h, w = owner.shape
        for enemy in opps:
            for dr, dc in DIRS:
                rr, cc = gr + dr, gc + dc
                if (0 <= rr < h and 0 <= cc < w and owner[rr, cc] == enemy
                        and int(army[rr, cc]) - 1 > garrison):
                    score -= 100_000
    return float(score)


def _matrix_choice(state, player: int, mine, theirs) -> int:
    """Shallow matrix plus ProbCut on the wait/launch sequence nodes."""
    cfg = state.env_config()
    st = state.to_genmod_state(cfg)
    pairs = []
    for a in mine:
        for b in theirs:
            joint = [0] * cfg.num_players
            joint[player] = int(a)
            joint[1 - player] = int(b)
            pairs.append(joint)
    successors = _joint_successors(st, jnp.asarray(pairs, jnp.int32), cfg)
    states = [jax.tree.map(lambda x, i=i: x[i], successors) for i in range(len(pairs))]
    values = np.asarray([_score_successor(s, player, cfg) for s in states])
    matrix = values.reshape(len(mine), len(theirs))
    # ProbCut is restricted to nodes where either side has a real producer
    # charge plan. At those nodes, cheaply score the first simultaneous ply,
    # then deepen promising rows by one reply ply around charge/defense/pass.
    # is fitted online from these exact shallow/deep row pairs.
    waiting_attack = False
    incoming_wait = 0
    for defender in range(state.n):
        if not _likely_siege(state, defender):
            continue
        incoming = _incoming_siege(state, defender)
        if (incoming[1] is not None and incoming[3] > 0
                and _siege_source_waiting(state, incoming[1], defender)):
            waiting_attack = True
            incoming_wait = max(incoming_wait, int(incoming[3]))
            break
    charge_plans = (_charge_general_plans(state, player, limit=1)
                    + _charge_general_plans(state, 1 - player, limit=1))
    charge_wait_node = any(p["wait"] > 0 and p["eta"] <= 240 for p in charge_plans)
    if waiting_attack or charge_wait_node:
        longest_wait = max(incoming_wait, max((p["wait"] for p in charge_plans), default=0))
        # A two-ply continuation sees only two PASS actions for a long charge.
        # Switch to the temporal macro tree once the wait is more than a few
        # turns, so the search reaches readiness and the ensuing route.
        if longest_wait >= 4:
            return _macro_wait_choice(state, player, mine, theirs)
        shallow_rows = matrix.min(axis=1)
        order = np.argsort(-shallow_rows)
        fit = _probcut_fit()
        alpha = -np.inf
        best_row = int(order[0])
        estimates = shallow_rows.astype(np.float64).copy()
        deepened = 0
        for rank, i in enumerate(order):
            i = int(i)
            predicted = float(shallow_rows[i])
            if fit is not None and np.isfinite(alpha):
                ml = _ml_predict(_ml_features(state, player, shallow_rows[i], mine[i]))
                if ml is not None:
                    predicted, sigma = ml
                else:
                    a, b, sigma = fit
                    predicted = a * float(shallow_rows[i]) + b
                # Fail-low: skip this deep row when its fitted 1.5-sigma
                # upper confidence bound remains below the best deep row.
                if predicted + 1.5 * sigma < alpha:
                    estimates[i] = predicted
                    continue
            # Spend more search on ambiguous wait/launch nodes. Deepen at most
            # four promising rows; regression estimates rank the rest.
            if deepened >= 4:
                estimates[i] = predicted
                continue
            child_states = []
            for j in range(len(theirs)):
                first = states[i * len(theirs) + j]
                child_states.append(first)
            deep_values = np.asarray([_deep_charge_minimax(s, player, cfg, depth=2)
                                      for s in child_states])
            deep_row = float(deep_values.min())
            _ML_SAMPLES.append((_ml_features(state, player, shallow_rows[i], mine[i]).tolist(), deep_row))
            _probcut_observe(shallow_rows[i], deep_row)
            estimates[i] = deep_row
            alpha = max(alpha, deep_row)
            deepened += 1
            # A proven terminal result cannot be improved meaningfully.
            if alpha >= 1e8:
                break
        best_row = int(np.argmax(estimates))
        return int(mine[best_row])
    # Fictitious play gives a fast approximation to the mixed equilibrium of
    # this small zero-sum matrix. Pure maximin overweights weird edge replies
    # and can choose PASS forever, so use the opponent's empirical response
    # mix while retaining a conservative tactical floor.
    nr, nc = matrix.shape
    row_count = np.ones(nr, dtype=np.float64)
    col_count = np.ones(nc, dtype=np.float64)
    for _ in range(32):
        row = int(np.argmax(matrix @ col_count))
        col = int(np.argmin(row_count @ matrix))
        row_count[row] += 1
        col_count[col] += 1
    expected = matrix @ (col_count / col_count.sum())
    worst = matrix.min(axis=1)
    utility = expected + 0.15 * (worst - expected)
    return int(mine[int(np.argmax(utility))])


def act(state, player: int) -> int:
    """Choose one legal action from the complete public board state."""
    _SIEGE_CACHE.clear()
    try:
        candidates = [_plan_candidates(state, p) for p in range(state.n)]
        # Deduplicate choices while retaining six candidate branches per side.
        candidates = [list(dict.fromkeys(c)) for c in candidates]
        # Protect the home general only when the attack is close enough that
        # collecting there is a tactical choice, not a reason to stop expanding.
        for p in range(state.n):
            if not _likely_siege(state, p):
                continue
            incoming = _incoming_siege(state, p)
            if (incoming[0] > 100 or incoming[3] <= 0 or incoming[3] > 50
                    or not _siege_source_waiting(state, incoming[1], p)):
                continue
            gens = np.argwhere(np.asarray(state.general, bool)
                               & (np.asarray(state.owner) == p))
            if len(gens):
                home = int(gens[0, 0]) * state.w + int(gens[0, 1])
                candidates[p] = [0 if (a > 0 and a // 9 == home) else a
                                 for a in candidates[p]]
        for c in candidates:
            c.extend([0] * (6 - len(c)))
        threat = _incoming_siege(state, player) if _likely_siege(state, player) else None
        hold_home = bool(threat is not None and threat[0] <= 100
                         and 0 < threat[3] <= 50
                         and _siege_source_waiting(state, threat[1], player))
        home = np.argwhere(np.asarray(state.general, bool)
                           & (np.asarray(state.owner) == player))
        home_idx = int(home[0, 0]) * state.w + int(home[0, 1]) if len(home) else -1
        # A waiting enemy producer is a tactical node, not a reason to pass.
        # Try to capture the launch tile or reinforce before it can leave.
        if (threat is not None and threat[3] <= 40
                and _siege_source_waiting(state, threat[1], player)):
            # Keep the strategic wait/launch response in the matrix so the
            # ProbCut extension can compare interrupting, reinforcing, and
            # letting the stack charge through a short queued sequence.
            mine = list(dict.fromkeys([
                _counter_launch_action(state, player, threat),
                _defense_action(state, player, threat),
                candidates[player][0], candidates[player][5], 0,
            ]))
            theirs = list(dict.fromkeys([
                candidates[1 - player][0], candidates[1 - player][5],
                _counter_launch_action(state, 1 - player), 0,
            ]))
            return _matrix_choice(state, player, mine, theirs)
        castle_eta = 10**6
        if EXPLICIT_CHARGE_MACROS:
            # First finish a competitive castle wait/charge plan. This gives
            # the general stack a concrete production objective before all-in.
            castle_macro = _castle_plan(state, player)
            if castle_macro is not None:
                anchor, target, direction, wait, travel = castle_macro
                castle_eta = int(wait + travel)
                if castle_eta <= 90:
                    if wait > 0:
                        return 0  # collect on the general/castle producer
                    if direction >= 0:
                        route = _castle_route_actions(state, player, anchor, target, limit=6)
                        if route and not (hold_home and route[0] // 9 == home_idx):
                            return route
            # Execute the producer-wait -> shortest-path general macro that
            # the charge-window tree evaluates after the next castle.
            own_charge = _charge_general_plans(state, player, limit=1)
            if own_charge:
                charge = own_charge[0]
                enemy_eta = threat[0] if threat is not None else 10**6
                if (charge["eta"] <= 100 and charge["eta"] + 2 < enemy_eta
                        and castle_eta > charge["eta"]):
                    if charge["wait"] > 0:
                        return 0
                    route = _charge_route_actions(charge, state, player, limit=6)
                    if route and not (hold_home and route[0] // 9 == home_idx):
                        return route
        # Safety override: when a computed all-in path can reach the general
        # within twenty ticks, do not let the material evaluator bury the
        # defensive branch under immediate expansion gains.
        if threat is not None and threat[3] == 0 and threat[0] <= 20:
            defense = _defense_action(state, player, threat)
            if defense:
                return int(defense)
            return 0
        if _has_conflict(state, candidates[player], candidates[1 - player]):
            return _matrix_choice(state, player, candidates[player], candidates[1 - player])

        # Start defending before the all-in is one move away. The route search
        # accounts for a stack charging on an enemy general or castle, so we
        # can reinforce or intercept while there is still time.
        if threat is not None and threat[3] == 0 and threat[0] <= 55:
            defense = _defense_action(state, player, threat)
            if defense:
                return int(defense)

        # Search long producer waits even without an immediate contact move.
        # A 170-tick charge is a single durative macro here, not 170 blind PASS
        # plies or an unexamined one-action A* decision.
        own_charge = _charge_general_plans(state, player, limit=1)
        own_castle = _castle_plan(state, player)
        long_general = bool(own_charge and own_charge[0]["wait"] >= 4
                            and own_charge[0]["eta"] <= _WAIT_HORIZON)
        long_castle = bool(own_castle and own_castle[3] >= 4
                           and own_castle[3] + own_castle[4] <= _WAIT_HORIZON)
        if long_general or long_castle:
            return _macro_wait_choice(state, player, candidates[player], candidates[1 - player])

        # Outside a conflict zone, A* gives the fast strategic move. Keep the
        # explicit gather option only when its castle is close to affordable
        # and we can reach it before the opponent's collector.
        from strategies.wastar import search
        plan = search(state, player, epsilon=1.0)
        siege = _castle_plan(state, player)
        if siege is not None:
            anchor, target, direction, wait, steps = siege
            ours_eta = wait + steps
            rival = next((p for p in range(state.n) if p != player and state.alive[p]), None)
            enemy_siege = _castle_plan(state, rival) if rival is not None else None
            enemy_eta = 300.0
            if enemy_siege is not None and enemy_siege[1] == target:
                enemy_eta = enemy_siege[3] + enemy_siege[4]
            if wait == 0 and direction >= 0:
                # Let A* select an enemy general rush if it has one; otherwise
                # launch the ready siege immediately.
                if plan is None or plan.target_kind != "enemy general":
                    action = _encode(anchor[0], anchor[1], direction, state.w)
                    if not (hold_home and action // 9 == home_idx):
                        return action
            elif wait <= 20 and ours_eta + 2 < enemy_eta:
                source = int(plan.action) // 9 if plan is not None else -1
                if source != anchor[0] * state.w + anchor[1]:
                    return _safe_prefix(plan, state, player)
                return 0
        if plan is not None:
            action = _safe_prefix(plan, state, player)
            first = action[0] if isinstance(action, list) and action else action
            if not (hold_home and first > 0 and first // 9 == home_idx):
                return action
        action = _march(state, player)
        if hold_home and action > 0 and action // 9 == home_idx:
            return 0
        return action
    except (ImportError, AttributeError, ValueError, TypeError):
        # Keep the agent usable if the search module is unavailable in a
        # restricted deployment; the BFS expansion policy remains legal.
        pass
    return _march(state, player)


def _live_reinforce_action(state, defender: int, threat) -> int:
    """Move a nearby mobile stack toward home when its general is threatened."""
    eta, source, _path, _wait, _travel = threat
    if source is None or eta > 120:
        return 0
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    mountain = np.asarray(state.mountain, bool)
    general = np.asarray(state.general, bool)
    hits = np.argwhere(general & (owner == defender))
    if not len(hits):
        return 0
    gr, gc = map(int, hits[0])
    h, w = owner.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    dist[gr, gc] = 0
    q = [(gr, gc)]
    for r, c in q:
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w and not mountain[rr, cc]
                    and owner[rr, cc] == defender and not np.isfinite(dist[rr, cc])):
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    options = []
    for sr, sc in zip(*np.nonzero((owner == defender) & (army >= 3))):
        sr, sc = int(sr), int(sc)
        if (sr, sc) == (gr, gc) or not np.isfinite(dist[sr, sc]):
            continue
        for d, (dr, dc) in enumerate(DIRS):
            rr, cc = sr + dr, sc + dc
            if (0 <= rr < h and 0 <= cc < w and owner[rr, cc] == defender
                    and dist[rr, cc] < dist[sr, sc]):
                # Prefer a stack that can arrive soon and still bring force.
                options.append((float(dist[sr, sc]), -int(army[sr, sc]),
                                _encode(sr, sc, d, w)))
    return int(min(options)[2]) if options else 0


def _live_intercepts(state, defender: int, threat, limit: int = 2):
    """Build move candidates that contest the current live rush route.

    Friendly route cells are explicitly excluded: sending into them is a
    reinforcement, not an interception. The exact joint transition later
    resolves the case where the rushing stack leaves the target this tick.
    """
    _eta, source, path, _wait, _travel = threat
    if source is None or len(path) < 2:
        return []
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    mountain = np.asarray(state.mountain, bool)
    h, w = owner.shape
    attacker = 1 - defender
    ranked = {}
    # Contest the near route and chokepoints first. Recompute this route on
    # the next observed tick; no path prefix is queued.
    for step, (tr0, tc0) in enumerate(path[1:min(len(path), 9)], start=1):
        tr, tc = int(tr0), int(tc0)
        target_owner = int(owner[tr, tc])
        if target_owner == defender or mountain[tr, tc]:
            continue
        target_force = int(army[tr, tc]) if target_owner >= 0 else int(army[tr, tc])
        for d, (dr, dc) in enumerate(DIRS):
            sr, sc = tr - dr, tc - dc
            if not (0 <= sr < h and 0 <= sc < w):
                continue
            if owner[sr, sc] != defender or army[sr, sc] < 3:
                continue
            if target_owner == attacker and (sr, sc) == (tr, tc):
                continue
            full = int(army[sr, sc]) - 1
            if full <= 0:
                continue
            # Exact capture/damage value is deliberately simple; joint-state
            # scoring decides whether a candidate is safe against a reply.
            wins_now = full > target_force
            meaningful_trade = full >= max(6, target_force // 2)
            if not (wins_now or meaningful_trade):
                continue
            action = _encode(sr, sc, d, w)
            value = (10000 if wins_now else 0) + min(full, target_force) * 4
            # Slight preference for stopping a stack earlier on its route.
            value -= step * 0.1
            ranked[action] = max(ranked.get(action, -1e9), value)
    return [a for a, _v in sorted(ranked.items(), key=lambda kv: -kv[1])[:limit]]


def _live_action_set(state, side: int):
    """Small script portfolio for a one-tick simultaneous response search."""
    from strategies.wastar import search

    incoming = _incoming_siege(state, side)  # unconditional latent rush scan
    general_plans = _charge_general_plans(state, side, limit=1)
    charge = _charge_action(state, side, general_plans[0]) if general_plans else 0
    plan = search(state, side, epsilon=1.0)
    astar = int(plan.action) if plan is not None else 0
    march = _march(state, side)
    castle = _castle_plan(state, side)
    castle_action = 0
    if castle is not None:
        anchor, _target, direction, wait, _steps = castle
        if wait == 0 and direction >= 0:
            castle_action = _encode(anchor[0], anchor[1], direction, state.w)
    defend = _live_reinforce_action(state, side, incoming)
    intercepts = _live_intercepts(state, side, incoming, limit=2)

    # One-step plans only. The next call receives the actual post-combat board
    # and recomputes launch readiness, routes, defenders, and interceptions.
    # A* returns an all-but-one action. Also offer its half-send form so the
    # search can build territory while keeping a real reserve at home.
    moves = [charge, castle_action, astar, march]
    actions = [0]
    for action in moves:
        if action <= 0:
            continue
        if _safe_to_depart_home(state, side, int(action)):
            actions.append(int(action))
        half = (int(action) // 9) * 9 + 5 + ((int(action) - 1) % 4)
        if _safe_to_depart_home(state, side, half):
            actions.append(half)
    # Defend and intercept branches remain present even when strategic moves
    # fill the compact action set.
    actions.extend([defend, *intercepts])
    unique = list(dict.fromkeys(int(a) for a in actions))
    return unique[:9]


def _safe_to_depart_home(state, side: int, action: int) -> bool:
    """Reject a move that exposes home to a rush launched next tick.

    Simultaneous moves make the root state's threat estimate deceptive: an
    opponent may be unable to beat today's garrison but able to attack as soon
    as our move leaves one unit behind. Re-run the exact route estimator with
    the candidate's resulting home reserve before permitting that departure.
    """
    if action <= 0:
        return True
    general = np.asarray(state.general, bool)
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    homes = np.argwhere(general & (owner == side))
    if not len(homes):
        return True
    hr, hc = map(int, homes[0])
    home_idx = hr * state.w + hc
    src = action // 9
    if src != home_idx:
        return True
    k = action % 9
    if k == 0:
        return True
    send = int(army[hr, hc]) // 2 if k >= 5 else int(army[hr, hc]) - 1
    remaining = int(army[hr, hc]) - send
    # An immediate capture ends the game before a return attack can matter.
    d = (k - 1) % 4
    dr, dc = DIRS[d]
    rr, cc = hr + dr, hc + dc
    if 0 <= rr < state.h and 0 <= cc < state.w:
        if (general[rr, cc] and owner[rr, cc] == 1 - side
                and send > int(army[rr, cc])):
            return True
    hypothetical = copy.copy(state)
    hypothetical.army = army.copy()
    hypothetical.army[hr, hc] = remaining
    threatened = _incoming_siege(hypothetical, side)
    eta, source, path, wait, travel = threatened
    if source is None or eta > 90 or len(path) < 2:
        return True
    # If the opponent still needs to collect, the next real tick is the
    # replanning point. Do not freeze growth forever by reserving against a
    # hypothetical force it does not own yet.
    if wait > 0:
        return True

    # Estimate the enemy force that would reach home from its current launch
    # source, assuming the route is clear. This deliberately underestimates
    # defence requirements by ignoring future enemy production after launch.
    sr, sc = source
    launch_army = int(army[sr, sc])
    route_cost = max(0, len(path) - 1)
    for tr0, tc0 in path[1:-1]:
        tr, tc = int(tr0), int(tc0)
        if owner[tr, tc] == side or owner[tr, tc] < 0:
            route_cost += int(army[tr, tc])
    arrival_force = max(0, launch_army - route_cost)
    # A tie cannot capture the general, so require a strict reserve margin.
    return remaining >= arrival_force


def _live_race_score(st, player: int, cfg) -> float:
    """Rule-derived value: victory, attack-time advantage, and production."""
    owner = np.asarray(st.owner)
    army = np.asarray(st.army, dtype=np.int32)
    castle = np.asarray(st.castle, bool)
    general = np.asarray(st.general, bool)
    alive = np.asarray(st.alive, bool)
    opponent = 1 - int(player)
    if not alive[player]:
        return -1e9
    if not alive[opponent]:
        return 1e9

    # The race calculation is recomputed on every exact joint successor. It
    # includes all stacks, producer growth while waiting, route garrisons,
    # and general growth at arrival; it does not trust a stored route.
    their_eta = _incoming_siege(st, player)[0]
    our_eta = _incoming_siege(st, opponent)[0]
    horizon = 240.0
    their_eta = min(horizon, float(their_eta))
    our_eta = min(horizon, float(our_eta))
    # Keep distant speculative races from freezing all expansion. The race
    # gets full tactical weight only as either feasible attack approaches;
    # every newly observed tick moves this urgency window forward.
    urgency = float(np.clip((120.0 - min(their_eta, our_eta)) / 100.0, 0.0, 1.0))
    race_weight = 0.5 + 3.5 * urgency

    producers = castle | general
    land_delta = int(np.sum(owner == player) - np.sum(owner == opponent))
    army_delta = int(army[owner == player].sum() - army[owner == opponent].sum())
    producer_delta = int(np.sum((owner == player) & producers)
                         - np.sum((owner == opponent) & producers))
    gen_me = np.argwhere(general & (owner == player))
    gen_opp = np.argwhere(general & (owner == opponent))
    home_army = int(army[tuple(gen_me[0])]) if len(gen_me) else 0
    enemy_home_army = int(army[tuple(gen_opp[0])]) if len(gen_opp) else 0
    remaining = max(0, cfg.max_steps - int(st.step))
    return (race_weight * (their_eta - our_eta)
            + 10.0 * producer_delta
            + 0.80 * land_delta
            + 0.030 * army_delta
            + 0.20 * (home_army - enemy_home_army)
            + 0.02 * remaining * producer_delta)


def _live_plan_bonus(state, side: int, actions):
    """Shape the one-tick tree toward a concrete A* objective route.

    A one-ply evaluator sees the first step of a castle route as a pure army
    loss. The bonus lets the matrix compare that step as the start of the A*
    plan, while still simulating the opponent's simultaneous response exactly.
    Only the current first step receives credit; the route is replanned next
    tick from the new board.
    """
    from strategies.wastar import search

    bonuses = np.zeros(len(actions), dtype=np.float64)
    try:
        plan = search(state, side, epsilon=1.0)
        castle = _castle_plan(state, side)
        targets = []
        if plan is not None:
            base = {"enemy general": 24.0, "neutral city": 35.0,
                    "enemy city": 20.0, "enemy tile": 8.0}.get(plan.target_kind, 0.0)
            if base:
                targets.append((int(plan.action), base))
        if castle is not None:
            anchor, _target, direction, wait, _steps = castle
            if wait == 0 and direction >= 0:
                targets.append((_encode(anchor[0], anchor[1], direction, state.w), 36.0))
        for idx, action in enumerate(actions):
            if action <= 0:
                continue
            src, k = divmod(int(action), 9)
            direction = (k - 1) % 4
            half = k >= 5
            for target, value in targets:
                if src == target // 9 and direction == (target % 9 - 1) % 4:
                    bonuses[idx] = max(bonuses[idx], value * (0.65 if half else 1.0))
    except (ImportError, AttributeError, ValueError, TypeError):
        pass
    return bonuses


def _live_matrix_choice(state, player: int, mine, theirs) -> int:
    """Exact one-tick simultaneous matrix with a shallow equilibrium policy."""
    cfg = state.env_config()
    st = state.to_genmod_state(cfg)
    joint_actions = []
    for a in mine:
        for b in theirs:
            joint = [0] * cfg.num_players
            joint[player] = int(a)
            joint[1 - player] = int(b)
            joint_actions.append(joint)
    successors = _joint_successors(st, jnp.asarray(joint_actions, jnp.int32), cfg)
    states = [jax.tree.map(lambda x, i=i: x[i], successors)
              for i in range(len(joint_actions))]
    values = np.asarray([_live_race_score(child, player, cfg) for child in states],
                        dtype=np.float64).reshape(len(mine), len(theirs))
    mine_bonus = _live_plan_bonus(state, player, mine)
    their_bonus = _live_plan_bonus(state, 1 - player, theirs)
    values += mine_bonus[:, None] - their_bonus[None, :]

    # Fictitious play approximates the mixed equilibrium for simultaneous
    # actions. Deterministic state-derived sampling realizes our mixed policy
    # without storing a stale plan between ticks.
    row_counts = np.ones(len(mine), dtype=np.float64)
    col_counts = np.ones(len(theirs), dtype=np.float64)
    for _ in range(48):
        row_counts[int(np.argmax(values @ col_counts))] += 1.0
        col_counts[int(np.argmin(row_counts @ values))] += 1.0
    probabilities = row_counts / row_counts.sum()
    flat_owner = np.asarray(state.owner).reshape(-1)
    flat_army = np.asarray(state.army).reshape(-1)
    tick = int(getattr(state, "tick", getattr(state, "step", 0)))
    seed = (tick * 1_000_003
            + int(np.dot(flat_owner.astype(np.int64) + 2,
                         np.arange(1, flat_owner.size + 1, dtype=np.int64)))
            + int(np.dot(flat_army.astype(np.int64) % 97,
                         np.arange(7, 7 + flat_army.size, dtype=np.int64))))
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    chosen = int(rng.choice(len(mine), p=probabilities))
    return int(mine[chosen])


def _replan_actions(state, side: int, limit: int = 3):
    """Small action set regenerated from an exact intermediate board."""
    from strategies.wastar import search

    actions = [0]
    plan = search(state, side, epsilon=1.0)
    if plan is not None and len(actions) < limit:
        full = int(plan.action)
        if _safe_to_depart_home(state, side, full):
            actions.append(full)
        k = full % 9
        half = (full // 9) * 9 + 5 + ((k - 1) % 4)
        if len(actions) < limit and half != full and _safe_to_depart_home(state, side, half):
            actions.append(half)

    incoming = _incoming_siege(state, side)
    if incoming[0] <= 100 and len(actions) < limit:
        for move in _live_intercepts(state, side, incoming, limit=1):
            if move not in actions:
                actions.append(int(move))
        reinforce = _live_reinforce_action(state, side, incoming)
        if reinforce and reinforce not in actions and len(actions) < limit:
            actions.append(int(reinforce))
    elif len(actions) < limit:
        castle = _castle_plan(state, side)
        if castle is not None:
            anchor, _target, direction, wait, _steps = castle
            if wait == 0 and direction >= 0:
                move = _encode(anchor[0], anchor[1], direction, state.w)
                if move not in actions and _safe_to_depart_home(state, side, move):
                    actions.append(move)
        if len(actions) < limit:
            move = int(_march(state, side))
            if move and move not in actions and _safe_to_depart_home(state, side, move):
                actions.append(move)
    return actions[:limit]


def _equilibrium_value(matrix: np.ndarray, iterations: int = 24):
    """Fictitious-play value and mixed frequencies for a zero-sum matrix."""
    rows, cols = matrix.shape
    row_count = np.ones(rows, dtype=np.float64)
    col_count = np.ones(cols, dtype=np.float64)
    for _ in range(iterations):
        row_count[int(np.argmax(matrix @ col_count))] += 1.0
        col_count[int(np.argmin(row_count @ matrix))] += 1.0
    value = float((row_count @ matrix @ col_count)
                  / (row_count.sum() * col_count.sum()))
    return value, row_count


def _live_two_ply_values(state, player: int, mine, theirs):
    """Score root joint actions through one freshly replanned joint reply."""
    from server.local_server import to_bot_state

    cfg = state.env_config()
    st = state.to_genmod_state(cfg)
    root_pairs = []
    for a in mine:
        for b in theirs:
            joint = [0] * cfg.num_players
            joint[player], joint[1 - player] = int(a), int(b)
            root_pairs.append(joint)
    first_batch = _joint_successors(st, jnp.asarray(root_pairs, jnp.int32), cfg)
    first_states = [jax.tree.map(lambda x, i=i: x[i], first_batch)
                    for i in range(len(root_pairs))]
    root_values = np.zeros((len(mine), len(theirs)), dtype=np.float64)
    for k, first in enumerate(first_states):
        if bool(np.asarray(first.done)):
            root_values.flat[k] = _live_race_score(first, player, cfg)
            continue
        observed = to_bot_state(first, cfg, player)
        # Both policies see the resolved board and rebuild A* routes, rush
        # readiness, defense, and interceptions before choosing the next tick.
        my_next = _replan_actions(observed, player, limit=3)
        opp_next = _replan_actions(observed, 1 - player, limit=3)
        reply_pairs = []
        for a in my_next:
            for b in opp_next:
                joint = [0] * cfg.num_players
                joint[player], joint[1 - player] = int(a), int(b)
                reply_pairs.append(joint)
        second_batch = _joint_successors(first, jnp.asarray(reply_pairs, jnp.int32), cfg)
        leaves = [jax.tree.map(lambda x, i=i: x[i], second_batch)
                  for i in range(len(reply_pairs))]
        reply_matrix = np.asarray([_live_race_score(x, player, cfg) for x in leaves],
                                  dtype=np.float64).reshape(len(my_next), len(opp_next))
        root_values.flat[k] = _equilibrium_value(reply_matrix)[0]
    return root_values


def _race_context(state) -> bool:
    """Spend the extra reply ply only around feasible producer/general races."""
    from strategies.wastar import search

    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    general = np.asarray(state.general, bool)
    castle = np.asarray(state.castle, bool)
    for side in range(state.n):
        homes = np.argwhere(general & (owner == side))
        home_army = int(army[tuple(homes[0])]) if len(homes) else 0
        plan = search(state, side, epsilon=1.0)
        if plan is not None and home_army >= 10 and plan.target_kind == "enemy general":
            return True
        latent = _charge_general_plans(state, side, limit=1)
        if (home_army >= 10 and latent and latent[0]["eta"] <= 180):
            return True
        siege = _castle_plan(state, side)
        if siege is not None and siege[3] <= 4 and siege[3] + siege[4] <= 24:
            return True
        incoming = _incoming_siege(state, side)
        if incoming[0] <= 55:
            return True
    return False


def _root_race_actions(state, side: int, limit: int = 5):
    """Compact root matrix while preserving the race/defense choice points."""
    from strategies.wastar import search

    incoming = _incoming_siege(state, side)
    intercepts = _live_intercepts(state, side, incoming, limit=2)
    reinforce = _live_reinforce_action(state, side, incoming)
    plan = search(state, side, epsilon=1.0)
    castle = _castle_plan(state, side)
    plan_moves = []
    if plan is not None:
        full = int(plan.action)
        if _safe_to_depart_home(state, side, full):
            plan_moves.append(full)
        half = (full // 9) * 9 + 5 + ((full % 9 - 1) % 4)
        if half != full and _safe_to_depart_home(state, side, half):
            plan_moves.append(half)
    castle_move = 0
    if castle is not None and castle[3] == 0 and castle[2] >= 0:
        castle_move = _encode(castle[0][0], castle[0][1], castle[2], state.w)
        if not _safe_to_depart_home(state, side, castle_move):
            castle_move = 0
    if incoming[0] <= 120:
        actions = [0, *intercepts]
        if reinforce:
            actions.append(reinforce)
        actions.extend(plan_moves)
        if castle_move:
            actions.append(castle_move)
    else:
        actions = [0, *plan_moves]
        if castle_move:
            actions.append(castle_move)
        actions.append(int(_march(state, side)))
        actions.extend(intercepts[:1])
    return list(dict.fromkeys(int(a) for a in actions))[:limit]


def _choose_matrix_action(state, player, mine, theirs, values):
    _value, row_count = _equilibrium_value(values, iterations=48)
    probabilities = row_count / row_count.sum()
    flat_owner = np.asarray(state.owner).reshape(-1)
    flat_army = np.asarray(state.army).reshape(-1)
    tick = int(getattr(state, "tick", getattr(state, "step", 0)))
    seed = (tick * 1_000_003
            + int(np.dot(flat_owner.astype(np.int64) + 2,
                         np.arange(1, flat_owner.size + 1, dtype=np.int64)))
            + int(np.dot(flat_army.astype(np.int64) % 97,
                         np.arange(7, 7 + flat_army.size, dtype=np.int64))))
    return int(mine[int(np.random.default_rng(seed & 0xFFFFFFFF)
                         .choice(len(mine), p=probabilities))])



# Counter-growth race model: calculate the actual stack needed at arrival,
# including every hostile garrison's production during launch wait and travel.
_v29_incoming_siege = _incoming_siege
_v29_charge_general_plans = _charge_general_plans

def _income_between(state, cell, side, start, end):
    r, c = cell
    owner = np.asarray(state.owner)
    if int(owner[r, c]) != int(side):
        return 0
    castle = bool(state.castle[r, c])
    general = bool(state.general[r, c])
    return ((end // 2 - start // 2) if castle or general else 0) + (end // 50 - start // 50)

def _route_launch_requirement(state, path, attacker, now, wait):
    """Minimum source stack to clear path and beat target at its arrival state."""
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    castle = np.asarray(state.castle, dtype=bool)
    general = np.asarray(state.general, dtype=bool)
    launch = now + wait
    required = 1
    for i, cell in enumerate(path[1:], 1):
        r, c = cell
        side = int(owner[r, c])
        if side != attacker:
            at_arrival = int(army[r, c])
            if side >= 0:
                end = launch + i
                producer = bool(castle[r, c] or general[r, c])
                at_arrival += ((end // 2 - now // 2) if producer else 0) + (end // 50 - now // 50)
            required += 1 + at_arrival
        else:
            required += 1
    return required

def _incoming_siege(state, defender: int):
    raw = _v29_incoming_siege(state, defender)
    eta, source, path, _wait0, travel = raw
    if source is None or not path:
        return raw
    now = int(getattr(state, "tick", getattr(state, "step", 0)))
    attacker = 1 - int(defender)
    sr, sc = source
    producer = bool((np.asarray(state.castle, bool)[sr, sc] or
                     np.asarray(state.general, bool)[sr, sc]) and
                    np.asarray(state.owner)[sr, sc] == attacker)
    base = int(np.asarray(state.army)[sr, sc])
    for wait in range(241 if producer else 1):
        stack = base + _income_between(state, source, attacker, now, now + wait)
        if stack >= _route_launch_requirement(state, path, attacker, now, wait):
            return (wait + int(travel), source, path, wait, int(travel))
    return (10**6, None, [], 10**6, 0)

def _charge_general_plans(state, player: int, limit: int = 3):
    plans = _v29_charge_general_plans(state, player, limit=max(limit, 8))
    now = int(getattr(state, "tick", getattr(state, "step", 0)))
    owner = np.asarray(state.owner)
    army = np.asarray(state.army, dtype=np.int32)
    out = []
    for p in plans:
        source = tuple(p["source"])
        base = int(army[source])
        is_producer = bool((np.asarray(state.castle, bool)[source] or
                            np.asarray(state.general, bool)[source]) and owner[source] == player)
        for wait in range(241 if is_producer else 1):
            stack = base + _income_between(state, source, player, now, now + wait)
            if stack >= _route_launch_requirement(state, p["path"], player, now, wait):
                q = dict(p)
                q["wait"] = wait
                q["eta"] = wait + int(p["travel"])
                q["army_at_launch"] = stack
                out.append(q)
                break
    out.sort(key=lambda p: (p["eta"], -p["army_at_launch"]))
    return out[:limit]

def act(state, player: int) -> int:
    """Live receding-horizon matrix, with a second replan at charge nodes."""
    _SIEGE_CACHE.clear()
    try:
        if _race_context(state):
            candidates = [_root_race_actions(state, side) for side in range(state.n)]
        else:
            candidates = [_live_action_set(state, side) for side in range(state.n)]
        mine, theirs = candidates[player], candidates[1 - player]
        if _race_context(state):
            values = _live_two_ply_values(state, player, mine, theirs)
            mb = _live_plan_bonus(state, player, mine)
            tb = _live_plan_bonus(state, 1 - player, theirs)
            values += mb[:, None] - tb[None, :]
            return _choose_matrix_action(state, player, mine, theirs, values)
        return _live_matrix_choice(state, player, mine, theirs)
    except (ImportError, AttributeError, ValueError, TypeError, IndexError):
        from strategies.wastar import search
        plan = search(state, player, epsilon=1.0)
        return int(plan.action) if plan is not None else int(_march(state, player))


if __name__ == "__main__":
    from server.bot_protocol import serve
    serve(act, name="Final Generals search bot")
