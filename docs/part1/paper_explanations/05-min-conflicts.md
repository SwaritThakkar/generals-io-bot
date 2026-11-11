# Min-Conflicts — Heuristic Repair for CSPs
Minton, Johnston, Philips & Laird · **AAAI 1990** · 8 pages · AIMA §6.4
`papers/minton1990_minconflicts_heuristic_repair_CSP_AAAI.pdf`

## Overview

Standard CSP solving (AIMA §6.3) is **constructive**: start with nothing assigned and
extend a partial assignment, backtracking when stuck.

Min-conflicts is **repair-based**: start from a *complete* assignment that is probably
broken, and fix it. Repeatedly pick a variable that is in conflict and give it the
value that leaves the fewest conflicts. Break ties randomly.

The results are famous: n-queens for a million queens in seconds, and it was the
scheduler used for Hubble Space Telescope observations — a real system where a week of
observations had to be planned under heavy constraints.

## The algorithm, in depth

### The heuristic

> **Min-Conflicts heuristic:** Given a variable in conflict, choose the value that
> minimises the number of other variables that will need to be repaired. Break ties
> randomly.

```
min_conflicts(csp, max_steps):
    assignment ← a complete assignment          # random, or greedy, or last tick's
    for i in 1..max_steps:
        if assignment is consistent: return assignment
        var  ← a randomly chosen variable that is currently in conflict
        val  ← the value for var minimising conflicts(var = val, assignment)
        assignment[var] ← val                   # break ties randomly
    return failure
```

### Why it works so well

The intuition the paper gives: a broken complete assignment is usually **almost
right**. The information in "here is a solution to a slightly different problem" is
enormous, and constructive search throws it away by starting from nothing. Repair
keeps it.

This is also why it is not complete — it can cycle, and it cannot prove
unsatisfiability. Random tie-breaking and restarts are what keep it out of loops.

### The second, less-quoted use

The paper makes a point that is easy to miss: min-conflicts is **also a value-ordering
heuristic inside backtracking search**. In "informed backtracking", you use the
initial assignment as a guide — try the value the current assignment suggests first,
and repair variables in an order given by the heuristic. So it is not only a local
search; it is a way to inject a starting guess into a systematic search.

## Ideas for our game

**1. Our real per-tick decision is an assignment problem.** Not "what is the single
best move" but *"which of my stacks does what"*:

```
Variables    one per stack we control (or per task that needs doing)
Domain       { expand toward region R, gather to general, defend tile T,
               take castle C, intercept enemy stack S }
Constraints  · two stacks should not be sent to the same target
             · if a threat exists, some stack must be able to reach it in time
             · a stack's army must exceed what its task requires
```

That is a CSP, and it changes only slightly from tick to tick.

**2. Repair is the only affordable option at 2 ticks/second.** Solving this from
scratch every 500 ms is wasteful, because **last tick's assignment is almost always
nearly right** — one stack has arrived, one new threat has appeared, everything else is
unchanged. Min-conflicts is precisely designed for that situation: seed with last
tick's assignment and repair the few variables now in conflict. This is the same
insight as ARA\*'s reuse and RTAA\*'s persistent `h`, applied to the assignment layer.

**3. Unsatisfiability is a signal, not just a failure.** If we cannot find any
assignment covering a threat, the position is *provably* losing that tile. That is a
much stronger statement than an evaluation function's opinion, and it should feed
straight into the evaluation as a term. **Using constraint propagation as an
evaluation feature is an idea we have not seen in the RTS-search literature** and is
worth writing up.

**4. It gives us the defence layer we currently do not have.** Our measured failure is
12 general captures in 20 games. A CSP whose constraints include *"the general's
threat must be assigned a defender"* addresses that directly and declaratively —
rather than by adding another weighted term to an evaluation function, which we tried
and which made things worse.

**5. It is honest about our syllabus.** This is AIMA §6.4 exactly, with the original
paper behind it. Nothing here needs anything the course has not covered.
