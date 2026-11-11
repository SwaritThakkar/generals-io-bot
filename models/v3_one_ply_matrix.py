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
import jax
import jax.numpy as jnp

import genmod
from genmod import EnvConfig
from functools import partial

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
            item = (rank, (int(ar), int(ac)), (tr, tc), int(first[tr, tc]), wait)
            if best is None or item[0] < best[0]:
                best = item
    if best is None:
        return None
    _, anchor, target, direction, wait = best
    return anchor, target, direction, wait, steps


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


@partial(jax.jit, static_argnames=("cfg",))
def _joint_successors(st, joint_actions, cfg):
    """Vectorized exact transitions for the 3x3 simultaneous action matrix."""
    return jax.vmap(lambda a: genmod.step(st, a, cfg)[0])(joint_actions)


def _plan_candidates(state, player: int):
    """Pruned macro-action set: A*, hold/launch castle, and frontier march."""
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
    return [astar_action, castle_action, march_action]


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
    try:
        candidates = [_plan_candidates(state, p) for p in range(state.n)]
        # Deduplicate choices while retaining a fixed 3x3 branching factor.
        candidates = [list(dict.fromkeys(c)) for c in candidates]
        for c in candidates:
            c.extend([c[-1] if c else 0] * (3 - len(c)))
        return _matrix_choice(state, player, candidates[player], candidates[1 - player])
    except (ImportError, AttributeError, ValueError, TypeError):
        # Keep the agent usable if the search module is unavailable in a
        # restricted deployment; the BFS expansion policy remains legal.
        pass
    return _march(state, player)


if __name__ == "__main__":
    from server.bot_protocol import serve
    serve(act, name="Final Generals search bot")
