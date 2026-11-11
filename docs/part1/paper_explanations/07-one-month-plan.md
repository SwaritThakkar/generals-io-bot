# One-month plan

**Baseline to beat:** `greedy_eval.py`, 20% win rate vs the scripted `expander` bot,
losing 12/20 by general capture.

**Measurement, fixed for the whole month.** Every change is a 200-game headless match
against the *previous* version and against `expander`, on the same seeds:

```bash
python server/local_server.py --p0 strategies/new.py --p1 strategies/old.py \
    --games 200 --headless
```

Nothing gets kept unless it wins. Keeping every version and running them against each
other is the only honest progress metric before facing an outside opponent.

---

## Week 1 — the tick loop, and fix the measured bug
**Papers:** RTAA\* (02), depression avoidance (03)

| build | detail |
|---|---|
| `h(tile)` table | "estimated ticks to own this tile", initialised as BFS distance from our territory |
| RTAA\* update | bounded A\* per tick, then `h[s] ← g[s̄] + h[s̄] − g[s]` over CLOSED |
| depression marking | one bit per tile: has its `h` ever been raised? Prefer unmarked tiles |
| 2-ply defensive check | for each candidate move, check only the opponent replies that capture our general — 4 cells, not 400 moves |

**Why this order.** The 2-ply defensive check alone addresses 12 of our 20 losses, and
it is a restricted minimax (AIMA §5.2) with the opponent's move list cut to the moves
that end the game. It is cheap and it targets a failure we have measured.

**Target:** beat `expander` more often than not. Anything above 50% is a real result
given we start at 20%.

## Week 2 — depth
**Papers:** ProbCut (04), plus asymmetric abstraction (`../papers/moraes2018…`)

| build | detail |
|---|---|
| opponent abstraction | collapse their ~400 moves to ~4 intents: defend / expand / attack-general / take-castle. Search 400 × 4 |
| alpha-beta + transposition table | the board is deterministic, so hashing works cleanly |
| ProbCut calibration | generate a few thousand self-play positions, run 1-ply and 3-ply on each, regress → `a`, `b`, `σ` |
| iterative deepening | so the search is interruptible at 500 ms |

**Deliverable for the report:** the ProbCut regression plot (1-ply vs 3-ply values with
the fitted line and σ band). That is a real figure from a real experiment.

**Target:** 3-ply within budget; beat Week 1.

## Week 3 — the assignment layer
**Paper:** min-conflicts (05)

| build | detail |
|---|---|
| CSP formulation | variables = our stacks; domains = {expand R, gather, defend T, take castle C}; constraints = no two stacks on one target, general's threat must be covered, army must exceed the task |
| min-conflicts repair | seed with **last tick's assignment**, repair only the variables now in conflict |
| unsatisfiability as a feature | if no assignment covers a threat, that tile is *provably* lost — feed it to the evaluation |

**Target:** general captures conceded should drop sharply. Measure that number
directly, not just win rate.

## Week 4 — metareasoning, and write-up
**Papers:** metareasoning (06)

| build | detail |
|---|---|
| Decision 1: identity action | 1-ply batched scan (0.6 ms); if `best − second > margin`, play immediately and skip the deep search |
| Decision 2: prefix commitment | commit a whole queued path when the plan is robust (walking through our own territory), one move when it is fragile (approaching contact) |
| budget reallocation | spend the reclaimed time on ticks where the top moves score within noise |

**The experiment that makes the report.** Fix the total time budget; vary how it is
allocated across ticks. If uniform allocation loses to metareasoned allocation at the
*same* total compute, that is a clean, publishable-shaped result and the strongest
thing we will have.

**Target:** same or better strength at materially less compute per game.

---

## Risks, stated honestly

| risk | mitigation |
|---|---|
| ProbCut calibration is noisy in a game this different from chess | it is optional; alpha-beta + abstraction alone should reach 3 ply |
| CSP layer conflicts with the search layer | keep them separate — CSP proposes candidate tasks, search chooses among them |
| Week 4 has no time left for write-up | the ProbCut plot (W2) and the capture-rate drop (W3) are already reportable; W4 is upside |
| Everything slips | Week 1 alone takes us from 20% to (target) >50%. That is a defensible project on its own. |

## What "done" looks like

1. A bot that beats the scripted baseline decisively, with the win rate over 200 games.
2. Three graphs: win rate vs version, ProbCut regression, strength vs compute budget.
3. Two negative results honestly reported — the danger-term regression (20% → 15%) and
   whatever else fails. These explain *why* the design is shaped as it is.
4. Every algorithm traceable to a paper and an AIMA chapter.
