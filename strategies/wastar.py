"""Weighted A* over the real board (AIMA 3.5.2, and Likhachev et al. NIPS 2003).

One search, one parameter. `epsilon` inflates the heuristic:

    f(n) = g(n) + epsilon * h(n)

epsilon = 1 is plain A* and returns the optimal plan. epsilon > 1 makes the
search greedy and directional: it commits sooner, expands far fewer nodes, and
the plan it returns costs at most `epsilon` times the optimal one.

The search problem, stated in the game's own terms:

    state      a passable cell of the board
    start      the largest stack we own that is able to move
    action     step to a 4-neighbour, costing one tick
    goal       any cell we do not own that this stack can still afford to take
               when it arrives (it sheds one unit per tile it leaves behind)
    goal cost  reaching target t also costs `penalty(t)`, the value we give up
               by settling for t instead of the most valuable target available

`h` is the Manhattan distance to the cheapest remaining target plus that
target's penalty, which is admissible: Manhattan never overestimates the walk on
a 4-connected unit-cost grid, and the penalty is exact.

The move we play is the first step of the plan.
"""

from __future__ import annotations

import heapq
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from genmod.config import DIRS, NUM_CELL_ACTIONS  # noqa: E402

# Target values in land-tile equivalents. A producer is worth 25 tiles because
# it makes 0.50 army/tick against a tile's 0.02 (see greedy_eval.py); taking a
# tile from the opponent swings two tiles, not one; the general ends the game.
V_ENEMY_GENERAL = 200.0
V_ENEMY_CASTLE = 40.0
V_NEUTRAL_CASTLE = 25.0
V_ENEMY_TILE = 3.0
V_NEUTRAL_TILE = 1.0

#: How many ticks of walking we are willing to trade for one tile of value.
VALUE_TO_TICKS = 1.0


@dataclass
class Plan:
    """What one weighted A* run returned."""

    epsilon: float
    action: int                    # flat action index to play now
    mover: tuple[int, int]         # (row, col) of the stack that moves
    dest: tuple[int, int]          # (row, col) it steps onto
    direction: int                 # 0=up 1=down 2=left 3=right
    target: tuple[int, int] | None  # (row, col) the plan is heading for
    target_kind: str               # "enemy general" | "neutral city" | ...
    path: list[tuple[int, int]] = field(default_factory=list)
    expanded: int = 0              # nodes popped and expanded
    generated: int = 0             # nodes pushed
    cost: float = 0.0              # g + penalty of the goal actually reached
    ticks: int = 0                 # path length in ticks


def _boards(state, player):
    """Pull plain numpy boards out of either a genmod.State or a BotState."""
    owner = np.asarray(state.owner, np.int8)
    army = np.asarray(state.army, np.int32)
    mountain = np.asarray(state.mountain, bool)
    castle = np.asarray(state.castle, bool)
    general = np.asarray(state.general, bool)
    return owner, army, mountain, castle, general


def _classify(r, c, owner, castle, general, me, opp):
    if general[r, c] and owner[r, c] == opp:
        return "enemy general", V_ENEMY_GENERAL
    if castle[r, c] and owner[r, c] == opp:
        return "enemy city", V_ENEMY_CASTLE
    if castle[r, c] and owner[r, c] < 0:
        return "neutral city", V_NEUTRAL_CASTLE
    if owner[r, c] == opp:
        return "enemy tile", V_ENEMY_TILE
    return "neutral tile", V_NEUTRAL_TILE


def search(state, player: int, epsilon: float = 1.0, opp: int | None = None,
           single_goal: bool = True) -> Plan | None:
    """Run weighted A* once and return the plan, or None if we cannot move."""
    owner, army, mountain, castle, general = _boards(state, player)
    h, w = owner.shape
    me = player
    opp = 1 - player if opp is None else opp

    # --- starts: every stack we own that is allowed to move -----------------
    movable = (owner == me) & (army >= 2)
    if not movable.any():
        return None
    starts = [(int(a), int(b)) for a, b in zip(*np.nonzero(movable))]
    flat = np.where(movable.reshape(-1), army.reshape(-1), -1)
    src = int(np.argmax(flat))
    sr, sc = divmod(src, w)
    src_army = int(army[sr, sc])

    # --- targets: cells we do not own and could still afford on arrival -----
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    manhattan_from_src = np.abs(rr - sr) + np.abs(cc - sc)
    garrison = np.where(owner == me, 0, army)
    # A target is affordable if SOME stack can still beat its garrison on arrival:
    # walking d tiles sheds d units, and one more stays behind on the last step.
    affordable = np.zeros((h, w), bool)
    for (ar, ac) in starts:
        reach = int(army[ar, ac]) - (np.abs(rr - ar) + np.abs(cc - ac)) - 1
        affordable |= reach > garrison
    # Only things worth walking to. Plain neutral tiles are filler: if every
    # adjacent empty cell is a goal the search stops before it has started.
    worth_it = castle | (owner == opp)
    is_target = (~mountain) & (owner != me) & affordable & worth_it
    is_target[sr, sc] = False
    if not is_target.any():
        return None

    tgt_cells = [(int(a), int(b)) for a, b in zip(*np.nonzero(is_target))]
    kinds, values = {}, {}
    for t in tgt_cells:
        k, v = _classify(t[0], t[1], owner, castle, general, me, opp)
        kinds[t], values[t] = k, v

    if single_goal:
        # Textbook weighted A*: one start, one goal, h = Manhattan. The most
        # valuable affordable target is the objective; ties go to the nearer one.
        goal_cell = max(tgt_cells, key=lambda t: (values[t], -manhattan_from_src[t]))
        is_target = np.zeros_like(is_target)
        is_target[goal_cell] = True
        tgt_cells = [goal_cell]
        penalty = {goal_cell: 0.0}
        hmap = (np.abs(rr - goal_cell[0]) + np.abs(cc - goal_cell[1])).astype(float)
    else:
        vmax = max(values.values())
        penalty = {t: VALUE_TO_TICKS * (vmax - values[t]) for t in tgt_cells}
        hmap = np.full((h, w), np.inf)
        for t in tgt_cells:
            d = np.abs(rr - t[0]) + np.abs(cc - t[1]) + penalty[t]
            np.minimum(hmap, d, out=hmap)

    # --- weighted A* --------------------------------------------------------
    INF = float("inf")
    g = np.full((h, w), INF)
    best_fuel = np.full((h, w), -1.0)
    parent = np.full((h, w, 2), -1, np.int32)
    root = np.full((h, w, 2), -1, np.int32)
    g[sr, sc] = 0.0

    tie = 0
    open_heap = []
    expanded = generated = 0
    for (ar, ac) in starts:
        g[ar, ac] = 0.0
        best_fuel[ar, ac] = float(army[ar, ac])
        root[ar, ac] = (ar, ac)
        tie += 1
        heapq.heappush(open_heap, (epsilon * float(hmap[ar, ac]), 0.0,
                                   -float(army[ar, ac]), tie, ar, ac))
        generated += 1
    goal = None
    goal_cost = INF

    while open_heap:
        f, gv, negfuel, _, r, c = heapq.heappop(open_heap)
        fuel = -negfuel
        if gv > g[r, c] and fuel < best_fuel[r, c]:
            continue
        expanded += 1

        # goal test: a target this stack can still afford having walked `gv` steps
        if is_target[r, c] and (fuel - 1) > garrison[r, c]:
            goal, goal_cost = (r, c), gv + penalty[(r, c)]
            break

        # do not walk *through* a target: taking it consumes the move
        if is_target[r, c]:
            continue

        for d, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or mountain[nr, nc]:
                continue
            ng, nfuel = gv + 1.0, fuel - 1.0
            if nfuel < 2.0:             # the stack is spent
                continue
            if ng < g[nr, nc] or nfuel > best_fuel[nr, nc]:
                g[nr, nc] = min(g[nr, nc], ng)
                best_fuel[nr, nc] = max(best_fuel[nr, nc], nfuel)
                parent[nr, nc] = (r, c)
                root[nr, nc] = root[r, c]
                tie += 1
                heapq.heappush(open_heap, (ng + epsilon * float(hmap[nr, nc]), ng,
                                           -nfuel, tie, nr, nc))
                generated += 1

    if goal is None:
        return None

    # --- unwind to the first step ------------------------------------------
    origin = (int(root[goal[0], goal[1]][0]), int(root[goal[0], goal[1]][1]))
    path = [goal]
    guard = 0
    while path[-1] != origin:
        pr, pc = parent[path[-1][0], path[-1][1]]
        if pr < 0:
            return None
        path.append((int(pr), int(pc)))
        guard += 1
        if guard > h * w:
            return None
    path.reverse()
    if len(path) < 2:
        return None

    (r0, c0), (r1, c1) = path[0], path[1]
    direction = DIRS.index((r1 - r0, c1 - c0))
    action = (r0 * w + c0) * NUM_CELL_ACTIONS + 1 + direction

    return Plan(
        epsilon=epsilon, action=int(action), mover=(r0, c0), dest=(r1, c1),
        direction=direction, target=goal, target_kind=kinds[goal], path=path,
        expanded=expanded, generated=generated, cost=float(goal_cost),
        ticks=len(path) - 1,
    )


def make_agent(epsilon: float = 1.0):
    """A `genmod-bot/1` agent that plays weighted A* at this epsilon."""

    def agent(state, player: int) -> int:
        plan = search(state, player, epsilon)
        return plan.action if plan is not None else 0

    agent.__name__ = f"wastar_eps{epsilon:g}"
    return agent


act = make_agent(1.0)
