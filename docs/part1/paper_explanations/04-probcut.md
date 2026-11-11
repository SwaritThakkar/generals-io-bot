# ProbCut — Forward Pruning in Alpha-Beta
Buro · `papers/buro_probcut_forward_pruning_alphabeta.pdf` · AIMA §5.3–5.4

## Overview

Alpha-beta prunes a subtree only when it can **prove** the subtree cannot affect the
result. That soundness is expensive: the branching factor stays high, and depth is
what actually makes a game-playing program strong.

ProbCut prunes when a subtree is **probably** irrelevant. Before searching a subtree
deeply, do a cheap **shallow** search. If the shallow result is far enough outside the
current `(α, β)` window, conclude that the deep search would almost certainly also
fall outside, and skip it entirely.

It is deliberately **unsound** — it can prune the best move. The point is that the
depth you buy is worth far more than the mistakes you make, and a single parameter
lets you dial the risk.

## The algorithm, in depth

### The statistical model

Let `v` be the value a depth-`d` search would return, and `v'` the value a shallower
depth-`d'` search returns. These are strongly correlated, and the paper models the
relationship as linear with Gaussian noise:

```
v  =  a · v'  +  b  +  e ,        e ~ N(0, σ²)
```

`a`, `b` and `σ` are fitted **offline by linear regression** over thousands of
positions, for each `(d', d)` pair you intend to use. This is the only "training" in
the method, and it is ordinary least squares — not machine learning in the sense your
course excludes.

### The pruning condition

We want to prune when `P(v ≥ β)` is high. Substituting the model and solving for `v'`:

```
v ≥ β   ⟺   a·v' + b + e ≥ β   ⟺   v' ≥ (β − b − e)/a
```

With a cut threshold `T` (in standard deviations), prune on a **fail-high** when

```
v'  ≥  (T·σ + β − b) / a
```

and symmetrically, prune on a **fail-low** when

```
v'  ≤  (−T·σ + α − b) / a
```

Straight from the paper's code:

```c
#define S 4                  // depth of the shallow search

bound = round((  T * sigma + beta  - b) / a);
if (shallow_search(S, bound - 1, bound) >= bound)   return beta;    // probably ≥ β

bound = round(( -T * sigma + alpha - b) / a);
if (shallow_search(S, bound, bound + 1) <= bound)   return alpha;   // probably ≤ α
```

Note the shallow searches use a **null window** (`bound-1, bound`) — they only need to
answer "above or below?", never an exact value, which makes them very cheap.

### Choosing T

`T` is the whole risk dial:
- large `T` → prune only when very confident → close to plain alpha-beta, safe, slow
- small `T` → prune aggressively → much deeper, occasionally throws away the best move

Typical values are around 1–1.5 standard deviations. **Multi-ProbCut** extends this
with different `(d', d)` pairs and different thresholds at different depths.

## Ideas for our game

**1. Our branching factor makes sound pruning insufficient.** Legal moves range from
~40 to ~400 per side. Even perfect alpha-beta ordering gives roughly `b^(d/2)` — with
`b ≈ 200`, three ply is already out of reach in 500 ms. Forward pruning is not a
refinement for us, it is the only route to depth.

**2. Our shallow search is nearly free, which is unusual and helpful.** Our engine
runs on JAX and we can already simulate **every candidate move in one batched call**
(measured: 0.6 ms for a full candidate set). So "evaluate all children at depth 1"
costs about one engine call. That is exactly the cheap estimator ProbCut needs, and we
get it for free from work already done.

**3. Fitting `a`, `b`, `σ` is a self-contained experiment.** Generate a few thousand
positions from self-play, run a 1-ply and a 3-ply search on each, regress one on the
other. This is a clean, reportable piece of work with a graph at the end of it — and a
good slide.

**4. It composes with everything else.** ProbCut sits inside alpha-beta, which sits
inside the tick loop. It does not conflict with RTAA\* or with the plan abstraction.

**Honest caveat for the report.** ProbCut is unsound: with `T = 1` we will sometimes
prune the winning move. The mitigation is standard — never prune within `k` ply of a
terminal (general capture), because that is precisely where an error is unrecoverable.
