# RTAA\* — Real-Time Adaptive A\*
Koenig & Likhachev · **AAMAS 2006** · 8 pages · AIMA §4.5 (online search agents)
`papers/koenig2006_realtime_adaptive_astar_RTAA_AAMAS.pdf`

## Overview

The setting: an agent must **act now**, cannot search to the goal, and will face the
same map again next step. This is your book's §4.5, and LRTA\* is the algorithm it
teaches.

RTAA\*'s contribution is one line. LRTA\* updates the heuristic of essentially one
state per move. RTAA\* runs a **bounded-depth A\* search**, then updates the heuristic
of **every state it expanded**, all in a single sweep, using one subtraction per state.
Same guarantees, far fewer moves to the goal, and it is trivial to implement.

## The algorithm, in depth

```
procedure realtime_adaptive_astar():
  while s_curr ∉ GOAL:
      lookahead  ← any integer > 0           # how much to think this step
      astar()                                # expand at most `lookahead` states
      if OPEN = ∅: return failure
      s̄ ← the state in OPEN with smallest f  # the frontier state A* was about to expand

      for each s ∈ CLOSED:                   # ← the whole contribution is this loop
          h[s] ← g[s̄] + h[s̄] − g[s]

      movements ← any integer > 0            # how many steps to commit to
      while s_curr ≠ s̄ and movements > 0:
          a ← the action on the cost-minimal trajectory from s_curr to s̄
          s_curr ← succ(s_curr, a)
          movements ← movements − 1
```

### Why that update is legal

Let `s̄` be the frontier state with the smallest `f`. Because A\* expands in
`f`-order, `f[s̄] = g[s̄] + h[s̄]` is a **lower bound on the distance from `s_curr` to
the goal**. Any path from an expanded state `s` to the goal that leaves the searched
region must pass through the frontier, so

```
goal_distance(s)  ≥  f[s̄] − g[s]  =  g[s̄] + h[s̄] − g[s]
```

which is exactly the assignment. So the new `h` is still **admissible** (never
overestimates), and the paper proves it is **monotonically non-decreasing** over time —
`h` only ever gets more informed, never worse.

### RTAA\* vs LRTA\*

Both are in the same family; they differ *only* in the update.

|  | LRTA\* | RTAA\* |
|---|---|---|
| update | Dijkstra-like backward pass over the local search space | one subtraction per expanded state |
| resulting `h` | more informed | slightly less informed |
| cost per move | much higher | one sweep |
| net effect | fewer moves, more time per move | **more moves, far less time — and usually wins on total time** |

The paper is explicit: LRTA\*'s heuristics end up *more* informed for the same
lookahead, but RTAA\* is so much cheaper per move that it reaches the goal sooner.
That tradeoff is exactly ours — we are time-bound, not move-bound.

### The two free parameters

`lookahead` and `movements` are deliberately left open — *"any desired integer greater
than zero"*. The paper does not say how to choose them. **Choosing them well is
metareasoning**, and it is the subject of `06-metareasoning.md`.

## Ideas for our game

**1. This is the shape of our tick loop.** Search as far as 500 ms allows, commit,
keep what we learned. That is literally RTAA\*.

**2. What `h` should mean for us.** Define `h(tile)` = estimated ticks to own that
tile. Initialise it as BFS distance from our territory. Every tick, RTAA\* raises the
values of tiles that turned out to be harder than they looked — a tile defended by a
big enemy stack gets its `h` pushed up and stops attracting our army. **The bot
learns the specific map it is playing on, without any machine learning** — the
"adaptive" here is a heuristic update rule, not a learned model.

**3. `h` survives across ticks, which ARA\* cannot.** Our board changes every tick,
so ARA\*'s reuse is only valid inside one deliberation. RTAA\*'s `h` table is designed
for exactly this: the environment changes and the heuristic keeps improving anyway.

**4. Implementation is genuinely small.** One A\* with an expansion cap, one loop over
CLOSED. Perhaps 60 lines on top of a pathfinder we already need.
