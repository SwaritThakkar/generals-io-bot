# Depression Avoidance in Real-Time Heuristic Search
Hernández & Baier · **AAAI/SoCS 2012** · 6 pages · AIMA §4.5
`papers/hernandez2012_realtime_adaptive_astar_depression_avoidance.pdf`

## Overview

This paper is about **the specific way real-time search fails**, and a cheap fix.

A **heuristic depression** is a connected region of the state space where `h`
underestimates the true cost badly, relative to the `h` values on the region's border.
Picture a bowl in the heuristic landscape. A greedy agent rolls in because everything
inside looks close to the goal, and then cannot get out: each move it makes raises
one state's `h` a little, so it wanders the bowl raising values one at a time until
the whole bowl is finally filled in. This is sometimes called *scrubbing*, and it can
waste hundreds of moves.

The fix — **depression avoidance** — is one sentence: *prefer to move to states that
are not known to be inside a depression.*

## The algorithm, in depth

### Detecting a depression for free

You do not need a separate detector. The paper's observation:

> **If a state's heuristic has been updated, that state is inside a depression.**

Because `h` starts admissible and updates only ever *raise* it (proved in the RTAA\*
paper — see `02-rtaa-star.md`), a raised `h` means the original estimate was too
optimistic *there*. That is the definition of a depression. So detection costs one
extra bit per state:

```
h0(s) ← h(s)                        # remember the original heuristic
...
after an update:  marked(s) ← (h(s) > h0(s))
```

### Two ways to use the mark

**mark-and-avoid.** When choosing which frontier state to head for, prefer unmarked
states. Only fall back to marked ones if every option is marked. In the search itself
this becomes a tie-break, or a secondary key: order the frontier by `(marked, f)`
instead of `f`.

**move-to-border.** Weaker but more nuanced. Rather than avoiding marked states
outright, prefer the ones that appear closest to the depression's *border* — i.e.
head for the exit rather than refusing to be in the bowl at all. This matters when
the agent is already deep inside a depression and every neighbouring state is marked.

The paper combines these with RTAA\* to get **aRTAA\***, and reports better-quality
solutions for the same time budget.

### Why it is a good fit for our course

It is a two-line change to an algorithm we are already implementing, it needs no new
theory, and it targets a failure mode we have already *measured* rather than one we
have read about.

## Ideas for our game

**1. We have already hit this, with data.** Our first bot (`strategies/greedy_eval.py`)
picks the move whose resulting board scores highest — one step of hill climbing. It
wins 20% against the scripted `expander`. When we instrumented it we found:

```
greedy general captured    12/20
expander general captured   3/20
timeout, greedy ahead       1/20
timeout, expander ahead     4/20
```

Land was *dead even* (27–27 at tick 300). It was not out-expanded; it walked into
positions that looked good one move out and were fatal two moves out. **That is a
heuristic depression**, and this paper is the named, published treatment of it.

**2. What a depression looks like on our board.** Concretely:
- A corridor of neutral tiles that dead-ends against mountains. Every step in looks
  like progress (land goes up); the whole excursion was wasted.
- Tiles adjacent to a large enemy stack. They score well until the stack takes them
  back, and we re-lose them repeatedly — the textbook scrubbing pattern.

**3. The concrete change.** Once `h(tile)` = "estimated ticks to own this tile" is
being maintained by RTAA\*, mark any tile whose `h` we have had to raise, and make the
move-selection prefer unmarked tiles. Tiles we keep losing get marked and stop
attracting our army. **This should be a ~10-line change with a measurable effect**, and
it is a clean experiment: run 200 games with the mark on and off and report win rate.

**4. Why this beats adding another evaluation term.** We already tried patching the
evaluation function with a "danger" term and it made the bot **worse** — 20% → 15%.
The reason is instructive: that term was nearly constant across our candidate moves,
and *a term that does not vary between neighbours cannot change an argmax*. Depression
marking is different — it is attached to states we are choosing between, so it
actually discriminates. **Reporting that failed experiment is worth as much as the
fix**, because it explains why the fix is shaped the way it is.
