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
import numpy as np
import jax
import jax.numpy as jnp

import genmod
from genmod import EnvConfig
from functools import partial

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SIEGE_CACHE = {}


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
            item = (rank, (int(ar), int(ac)), (tr, tc), int(first[tr, tc]), wait, steps)
            if best is None or item[0] < best[0]:
                best = item
    if best is None:
        return None
    _, anchor, target, direction, wait, steps = best
    return anchor, target, direction, wait, steps


def _incoming_siege(state, defender: int):
    """Estimate the fastest feasible all-in stack route to a general.

    Reverse Dijkstra prices each step by one unit of movement attrition plus
    the garrison on defender or neutral cells. It then checks every enemy
    stack, including a stack still charging on an enemy general/castle, and
    estimates when that producer could launch the shortest feasible attack.
    Returns ``(eta, source, path)``. ETA includes waiting to build enough army.
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
        result = (10**6, None, [])
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
    best = (10**6, None, [])
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
            best = (eta, (sr, sc), path)
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
    armed = (owner == opp) & (army >= 12) & ~mountain
    if np.any(armed & (distance <= 24)):
        return True
    charging = (owner == opp) & (army >= 6) & (castle | general) & ~mountain
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
    eta, source, path = assessment if assessment is not None else _incoming_siege(state, player)
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


@partial(jax.jit, static_argnames=("cfg",))
def _joint_successors(st, joint_actions, cfg):
    """Vectorized exact transitions for the pruned simultaneous action matrix."""
    return jax.vmap(lambda a: genmod.step(st, a, cfg)[0])(joint_actions)


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
    return [astar_action, castle_action, march_action, defense_action]


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
            eta, source, _path = _incoming_siege(state, defender)
            # Charging on a producer is a strategic wait branch. Do not pay
            # for a full response matrix until that stack actually launches.
            if eta <= 20 or (eta <= 100 and not _siege_source_waiting(state, source, defender)):
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
    # A route toward a general or enemy castle is tactically fragile.
    prefix = 1 if getattr(plan, "target_kind", "") in ("enemy general", "enemy city") else 3
    opponent_cells = np.argwhere((owner >= 0) & (owner != player))
    if len(opponent_cells):
        for r, c in path[:min(len(path), 4)]:
            if np.min(np.abs(opponent_cells[:, 0] - int(r)) + np.abs(opponent_cells[:, 1] - int(c))) <= 2:
                prefix = 1
                break
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
        eta_to_me, source, _ = _incoming_siege(st, me)
        if eta_to_me <= 100 and source is not None:
            er, ec = source
            enemy_waiting = bool((castle[er, ec] | general[er, ec]) and owner[er, ec] == opp)
            my_gen = np.argwhere(general & (owner == me))
            if enemy_waiting and len(my_gen):
                gr, gc = map(int, my_gen[0])
                score += 2.0 * min(80, int(army[gr, gc]))
            elif eta_to_me <= 20:
                score -= 5.0 * (21 - eta_to_me)
            if eta_to_me <= 4:
                score -= 100_000
    if _likely_siege(st, opp):
        eta_to_opp, _, _ = _incoming_siege(st, opp)
        if eta_to_opp <= 20:
            score += 5.0 * (21 - eta_to_opp)
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
    """One ply of maximin over pruned simultaneous action candidates."""
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
        # Deduplicate choices while retaining four candidate branches per side.
        candidates = [list(dict.fromkeys(c)) for c in candidates]
        for c in candidates:
            c.extend([c[-1] if c else 0] * (4 - len(c)))
        # The wait/charge branch: while their principal stack is still on a
        # producer and can reach us within the planning horizon, bank at our
        # own general rather than spending that stack outward.
        if _likely_siege(state, player):
            threat = _incoming_siege(state, player)
            if threat[0] <= 120 and _siege_source_waiting(state, threat[1], player):
                return 0
        # Safety override: when a computed all-in path can reach the general
        # within twenty ticks, do not let the material evaluator bury the
        # defensive branch under immediate expansion gains.
        if _likely_siege(state, player):
            threat = _incoming_siege(state, player)
            if threat[0] <= 20:
                defense = _defense_action(state, player, threat)
                if defense:
                    return int(defense)
                return 0
        if _has_conflict(state, candidates[player], candidates[1 - player]):
            return _matrix_choice(state, player, candidates[player], candidates[1 - player])

        # Start defending before the all-in is one move away. The route search
        # accounts for a stack charging on an enemy general or castle, so we
        # can reinforce or intercept while there is still time.
        if _likely_siege(state, player):
            threat = _incoming_siege(state, player)
            if threat[0] <= 120:
                # While the attacker is still charging at its general/castle,
                # keep our own general stack in place and grow the garrison.
                # Once it actually launches, use intercept/reinforcement moves.
                if threat[1] is not None:
                    er, ec = threat[1]
                    enemy_producer = (np.asarray(state.general, bool)[er, ec]
                                      or np.asarray(state.castle, bool)[er, ec])
                    if enemy_producer:
                        return 0
                if threat[0] <= 55:
                    defense = _defense_action(state, player, threat)
                    return int(defense) if defense else 0

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
                    return _encode(anchor[0], anchor[1], direction, state.w)
            elif wait <= 20 and ours_eta + 2 < enemy_eta:
                source = int(plan.action) // 9 if plan is not None else -1
                if source != anchor[0] * state.w + anchor[1]:
                    return _safe_prefix(plan, state, player)
                return 0
        if plan is not None:
            return _safe_prefix(plan, state, player)
        return _march(state, player)
    except (ImportError, AttributeError, ValueError, TypeError):
        # Keep the agent usable if the search module is unavailable in a
        # restricted deployment; the BFS expansion policy remains legal.
        pass
    return _march(state, player)


if __name__ == "__main__":
    from server.bot_protocol import serve
    serve(act, name="Final Generals search bot")
