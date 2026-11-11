# Metareasoning — Deciding How Much To Think
O'Ceallaigh & Ruml · **SoCS 2015** · 9 pages · `papers/oceallaigh2015_metareasoning_realtime_heuristic_search_SoCS.pdf`
Cserna, Ruml & Frank · **ICAPS 2017** · 5 pages · `papers/cserna2017_planning_time_to_think_metareasoning_ICAPS.pdf`
Russell & Wefald · **UAI 1990** · 7 pages · `papers/russell1990_finegrained_decision_theoretic_search_control_UAI.pdf`
AIMA ch. 3–4 (search control)

> This direction came from our own observation: **most ticks in a Generals.io game are
> obvious.** There is nothing to decide, so searching them deeply is wasted budget that
> should have been spent on the handful of ticks that actually matter. That intuition
> has a name and a literature.

## Overview

Ordinary search asks *"which action should I take?"*. Metareasoning asks the level-up
question: *"which computation should I perform — and is any computation worth doing at
all right now?"*

Russell & Wefald (UAI 1990) gave the framework. Treat each possible expansion as an
action with a cost (time) and a benefit (**the probability that performing it changes
which move you finally play**). Expand the node with the highest *value of computation*.
Crucially: **a computation that cannot change your answer has zero value, however deep
it is.**

O'Ceallaigh & Ruml apply this to real-time search and identify two concrete decision
points, both of which are exactly the parameters RTAA\* leaves undefined.

## The two decisions, in depth

Recall the RTAA\* skeleton from `02-rtaa-star.md`:

```
lookahead  ← any integer > 0        ← how much to think.  Undefined by the paper.
movements  ← any integer > 0        ← how much to commit. Undefined by the paper.
```

The paper's whole subject is choosing these two numbers *dynamically*.

### Decision 1 — the identity action

An **identity action** keeps the agent in the same state while it thinks longer. The
question: *should I act now, or stay put and deliberate?*

Standard real-time search never considers this — it assumes acting is always right.
But if the current best move is only marginally better than the alternatives, another
round of lookahead may reveal a much better one, and standing still costs only time.

The paper measures everything in **GAT (goal achievement time)** = planning time +
execution time. Under GAT, the identity action is not free — but it is often cheaper
than committing to a bad move and having to undo it.

### Decision 2 — prefix commitment

After lookahead, you have a path to the best frontier state. **How much of it do you
execute before re-planning?**

The existing algorithms are both extremes and neither adapts:

| algorithm | commits to |
|---|---|
| LRTA\* (Korf 1990) | exactly 1 action |
| Real-time A\* variants (2013) | the entire path to the frontier |

Committing to one action means re-planning constantly. Committing to the whole path
means acting on stale information. The paper chooses dynamically: **commit further
when the plan is robust, commit less when it is fragile.** It reports reduced GAT on
several benchmark domains.

### Russell & Wefald's earlier answer: DTA\*

Decision-Theoretic A\* weighs the time a search will take against the expected
improvement in the decision. The unit of computation there is a single node
expansion — hence "fine-grained". The rule: **stop searching when the expected benefit
of another expansion falls below its cost.**

## Ideas for our game

**1. Generals.io has a real identity action: `pass`.** Action index 0 on every cell is
a legal no-op. So Decision 1 is available to us literally, not by analogy — we can
stand still and think. This is unusual; most domains have to simulate it.

**2. Most ticks are obvious, and we can detect that cheaply.** Concretely: run the
1-ply batched evaluation we already have (0.6 ms for all candidate moves) and look at
the spread of scores.

```
if best_score − second_best_score > margin:
        play it immediately            # the decision is not close; deep search is waste
else:
        spend the remaining ~490 ms searching
```

That single rule is Decision 1, it is maybe 10 lines, and it should reclaim most of
our budget. During expansion phases — long stretches where one move is clearly best —
we currently burn the full budget for nothing.

**3. Prefix commitment maps onto move queueing.** The game lets a player queue a path
of moves. A gathering stack walking 12 tiles through our own territory is a robust
plan: commit the whole path and skip planning for 12 ticks. A stack approaching
contested territory is fragile: commit one move and re-plan. **That is Decision 2,
exactly, and the game's own queue mechanic implements it for us.**

**4. Where to spend the reclaimed budget.** The ticks that matter are the ones where
the sandwich between "I move first" and "they move first" is wide, or where the top
few moves score within noise of each other. Those are precisely the ticks where our
bot is currently losing games.

**5. This is our most defensible original contribution.** The metareasoning literature
targets single-agent pathfinding. Applying identity actions and prefix commitment to a
**simultaneous-move adversarial** game, with a domain that supplies a genuine no-op and
a genuine move queue, is not something we have found in the literature. It is a
one-month-sized idea, it is measurable (GAT-equivalent: win rate at a fixed ms budget),
and it needs nothing beyond AIMA chapters 3–5.
