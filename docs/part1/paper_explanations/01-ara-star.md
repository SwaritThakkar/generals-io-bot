# ARA\* — Anytime A\* with Provable Bounds on Sub-Optimality
Likhachev, Gordon & Thrun · **NIPS 2003** · 8 pages · AIMA ch. 3
`papers/likhachev2003_ARAstar_anytime_astar_NIPS.pdf`

## Overview

A\* gives you an optimal path but only tells you *when it finishes*. If your deadline
arrives first, you have nothing. **Weighted A\*** fixes the speed by inflating the
heuristic — `f(s) = g(s) + ε·h(s)` with `ε > 1` — which finds a path much faster and
guarantees it costs at most `ε ×` optimal.

ARA\*'s contribution: run weighted A\* repeatedly with **decreasing** `ε`, and make
each run **reuse the previous run's work** instead of starting from scratch. You get
a usable answer almost immediately, it improves steadily, and at every instant you
know exactly how suboptimal it is.

## The algorithm, in depth

### The idea it rests on: local inconsistency

A state `s` is **locally inconsistent** when its `g`-value has just been lowered and
it has not been expanded since. The paper's key observation:

> **OPEN is exactly the set of locally inconsistent states.**

Lowering `g(s)` breaks the relationship `g(s') = min_{s''∈pred(s')}(g(s'') + c(s'',s'))`
for `s`'s successors, and expanding `s` repairs it — pushing the inconsistency down to
the children. So a search is just *propagating inconsistency until none is left*.

That reframing is what makes reuse possible: to resume with a new `ε`, you do not
restart, you just carry the inconsistent states forward.

### The catch, and INCONS

With `ε > 1` the heuristic may be inconsistent, so plain weighted A\* can expand a
state **many times**. ARA\* forbids re-expansion within one iteration — that is what
keeps each iteration cheap and preserves the `ε` bound.

But a state whose `g` improves *after* it was already expanded is genuinely
inconsistent and must not be forgotten. It goes into a third set, `INCONS`.

```
fvalue(s) = g(s) + ε · h(s)

ImprovePath():
  while fvalue(s_goal) > min over s in OPEN of fvalue(s):
      remove s with smallest fvalue from OPEN
      CLOSED ← CLOSED ∪ {s}
      for each successor s' of s:
          if s' unvisited:  g(s') ← ∞
          if g(s') > g(s) + c(s,s'):
              g(s') ← g(s) + c(s,s')
              if s' ∉ CLOSED:  insert s' into OPEN with fvalue(s')
              else:            insert s' into INCONS        ← the trick
```

```
Main():
  g(s_goal) ← ∞;  g(s_start) ← 0
  OPEN = CLOSED = INCONS = ∅
  insert s_start into OPEN
  ImprovePath()
  ε' ← min(ε, g(s_goal) / min over s in OPEN ∪ INCONS of (g(s) + h(s)))
  publish the current ε'-suboptimal solution
  while ε' > 1:
      decrease ε
      move all states from INCONS into OPEN         ← reuse, instead of restarting
      re-sort OPEN by the new fvalue
      CLOSED ← ∅
      ImprovePath()
      ε' ← min(ε, g(s_goal) / min over s in OPEN ∪ INCONS of (g(s) + h(s)))
      publish the current ε'-suboptimal solution
```

### The bound you actually get

Note that the published bound is **not** `ε`. It is

```
ε' = min( ε ,  g(s_goal) / min_{s ∈ OPEN ∪ INCONS} (g(s) + h(s)) )
```

The denominator is a lower bound on the optimal cost, so this is often much tighter
than `ε`. You frequently have a near-optimal answer long before `ε` reaches 1.

## Ideas for our game

**1. The tick budget is exactly ARA\*'s setting.** We must return a move in 500 ms,
every tick, forever. An algorithm that returns nothing until it finishes is useless
here. The right structure is:

```
deadline ← now + 400ms
best_move ← greedy move            # always have an answer
for ε in [3.0, 2.0, 1.5, 1.2, 1.0]:
    if now > deadline: break
    best_move ← search(ε), reusing the previous iteration
return best_move
```

**2. It is the right tool for the opening.** Maximising land at tick 50 is a
single-agent, deterministic, adversary-free shortest-path problem — a textbook A\*
target. Run ARA\* offline per map and we get *"this opening is at most 5% worse than
optimal"*, which is a measurable, defensible result for the report.

**3. The bound is a reportable number.** Most bots can only say "it plays well".
ARA\* lets us say "within 12% of optimal, in 400 ms". That is worth more in a write-up
than any amount of hand-tuning.

**Caveat.** ARA\* assumes a fixed goal and a static graph. Our board changes every
tick, so the reuse is only valid *within* one tick's deliberation, not across ticks.
Across ticks we need RTAA\* instead — see `02-rtaa-star.md`.
