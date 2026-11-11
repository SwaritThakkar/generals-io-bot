# Research directions — a search-only agent for fogless Generals.io

Everything here stays inside AIMA chapters 3–6 (Russell & Norvig [R&N]): uninformed
and informed search, local search, online/real-time search, adversarial search, and
constraint satisfaction. No learning of any kind.

Section numbers cited as §x.y refer to AIMA 4th edition; 3rd-edition numbers are
given where they differ.

---

> **Note on the PDFs.** This document was written when `../papers/` held 26 papers.
> That folder now holds only the post-2015 set — see
> [`TOP-PAPERS.md`](TOP-PAPERS.md) for the current recommendations. The citations
> below are all still valid references; to get the PDFs back, run
> `bash ../papers/restore-removed.sh`.

## 0. The game as a search problem (§3.1)

| | |
|---|---|
| **States** | `s = (owner: V→{−1,0,1}, army: V→ℕ, castle ⊆ V, general: {0,1}→V, t)` |
| **V** | passable cells, `\|V\| ≈ 0.8·H·W ≈ 423` on 23×23 |
| **Actions** | `A_i(s) = {pass} ∪ {(v,d,h) : owner(v)=i, army(v)≥2, v+d ∈ V}` |
| **Branching** | `\|A_i(s)\| = 1 + 2·Σ_{v: owner(v)=i, army(v)≥2} deg(v)` — 40–400 in practice |
| **Transition** | deterministic in the **joint** action `(a₀,a₁)` |
| **Goal** | `owner(general(1−i)) = i` |
| **Cost** | 1 tick per action; episode capped at `max_ticks` |

Two properties make this unusually tractable, both consequences of your modification:

1. **The state is fully observable.** No belief state, so §4.4 (partially observable
   search) and the whole fog apparatus in Human.exe [Straka26 §VI-A] are irrelevant.
2. **The transition is deterministic.** Randomness enters only at map generation.
   Transposition tables, retrograde analysis and exact endgame proofs are all on the table.

The one residual difficulty: **actions are simultaneous**, so this is not a
sequential game and plain minimax (§5.2) is unsound. Section 3 below handles that
rigorously without leaving chapter 5.

---

## 1. The production calculus — derive your numbers before you design anything

This is not from a paper. It falls straight out of the rules, and it should drive
every evaluation weight you pick.

Let `L_i(t)` = tiles owned, `P_i(t)` = producers owned (general + castles). From the
rules (`general_growth_interval = 2`, `land_growth_interval = 50`):

```
army accrual rate     ρ_i(t) = P_i(t)/2 + L_i(t)/50     [army per tick]
one producer          0.5   army/tick
one land tile         0.02  army/tick
⇒ 1 castle ≡ 25 land tiles of production
```

Now price the two purchases. A walking stack of size `a` entering an empty neutral
tile arrives with `a−1`, captures, and leaves 1 unit behind; so **land costs
1 army per tile, permanently parked**. A neutral castle costs its garrison
`g ∈ [40,50]`.

```
spend A army on land   → A tiles      → 0.02·A army/tick
spend A army on castle → 1 producer   → 0.50    army/tick
break-even:  0.02·A = 0.5  ⇒  A = 25
```

**Since castles cost 40–50, buying land dominates buying a castle on pure
production by a factor of ≈1.8, whenever `g` free tiles are reachable.**

That is a falsifiable claim, and you have the harness to test it:

> **Experiment 1.** Two identical expanders, one of which takes any castle it can
> afford, one of which never does until free land runs out. 200 games.
> `python server/local_server.py --p0 a.py --p1 b.py --games 200 --headless`

Reasons the claim might fail, each worth isolating: castles are defensible anchors;
tiles are recapturable for 2 army each while a castle is not; free land runs out in
the midgame; taking a castle denies it to the opponent (worth `0.5` army/tick *to
them*, so the swing is 1.0). The interesting research question is **where the
crossover tick is**, and it is measurable.

---

## 2. Direction A — The opening, solved exactly (§3.4, §3.5)

For roughly the first 34 ticks the two players cannot interact: generals spawn ≥17
apart, and a stack needs army before it can travel. **The opening is therefore a
single-agent, deterministic, fully-observable shortest-path problem** — the purest
possible chapter-3 exercise, and it is worth real Elo.

Objective: maximise `L(50)`, land at the first land tick.

**An admissible bound.** A lone general holds `a(T) = 1 + ⌊T/2⌋` army at tick `T`.
Every captured tile parks exactly 1 unit and consumes 1 tick of a stack's time, and
the general must retain ≥1. Hence

```
L(50) ≤ 1 + ⌊50/2⌋ = 26          (1 for the general's own tile)
```

More usefully, from a state at tick `t` with `k` tiles and total army `A`:

```
h(s) = min( 1 + ⌊50/2⌋ − k ,  50 − t )        [tiles still obtainable]
```

which is admissible (you cannot park more units than you will ever produce, nor
make more captures than you have ticks). That is exactly the ingredient
depth-first branch-and-bound needs, and it is the same construction Churchill &
Buro use for StarCraft build orders — an admissible bound from resource
accumulation rates [Churchill11].

**Keep the search space small** by restricting to the policy shape *"launch the
general's accumulated stack at ticks t₁ < t₂ < … < tₘ, routing each greedily"*.
Then the decision vector is just `(m, t₁…tₘ)` and DFBB over it is trivial. Report
the optimal opening per map. Human openings reach ~23–25 land; the bound says 26.

> **Experiment 2.** Optimal-opening bot vs. `expander`, first 50 ticks only,
> compare `L(50)` across 200 maps. If your DFBB is right you should never lose.

## 3. Direction B — Simultaneous moves, handled soundly (§5.2, §5.3)

Plain minimax assumes you move second and *see* the opponent's move; it will
systematically overestimate your position. You do not need game theory (ch. 17) to
fix this. Define two **serialisations**, both computable with ordinary alpha-beta:

```
V_me(s)   : I commit first, opponent responds knowing my move     (worst case for me)
V_opp(s)  : opponent commits first, I respond knowing their move   (best case for me)
```

Because information can only help the player who moves second,

```
V_me(s)  ≤  V_sim(s)  ≤  V_opp(s)
```

Two things follow, and both are yours to exploit:

1. **Play `argmax_a V_me`.** This is a maximin / safety-first policy: the value you
   compute is *guaranteed* regardless of what the opponent does. A pessimistic
   bound you can actually rely on beats an optimistic estimate you cannot.
2. **When `V_me(s) = V_opp(s)`, the value is exact** and independent of move order —
   so the node can be returned immediately without expanding the other
   serialisation. Sandwich width `V_opp − V_me` is a free, principled measure of
   *how much simultaneity matters at this node*; wide sandwiches are precisely the
   tactical positions deserving deeper search. **This is a good place to look for a
   novel contribution**: an adaptive-depth rule driven by sandwich width.

For the theoretically exact treatment (solving a matrix game at each node rather
than serialising) see [Saffidine12]; for the pragmatic serialisation plus playout
evaluation, transposition tables and script-based move ordering, see [Churchill12a],
which reaches 8v8 combat decisions in under 5 ms.

## 4. Direction C — The race heuristic (§3.5, §5.4) ★ the fogless-specific idea

This is the direction I would push hardest, because **it only works without fog.**

Define `T_att(i)` = the earliest tick at which player `i` can capture the opponent's
general, assuming the defender does nothing new. Then

```
Eval_race(s) = T_att(1−i) − T_att(i)
```

If you can kill them sooner than they can kill you, you are winning the race,
regardless of who has more land. With fog, `T_att` is unknowable — you do not know
the enemy general's garrison or location. **Without fog it is a shortest-path
computation**, and you can therefore *evaluate the actual objective* rather than a
material proxy.

**Computing `T_att`.** Along a path `π = v₀…v_k` a stack loses 1 unit per hop
(parked) plus the garrison of each enemy tile it crosses. The enemy general grows
deterministically. So the launch is feasible iff

```
a_{v₀}(t) − Σ_{j=1..k} (e_j + 1)  >  g(t + k),      g(t') = g(t) + ⌊(t'−t)/2⌋ + land ticks
```

and `T_att(i) = min { t + k }` over start tiles, paths and launch times satisfying it.

This is a **resource-constrained shortest path**, solved by A\* (§3.5) over
augmented nodes `(cell, army-remaining)` with the admissible heuristic
`h = BFS distance to the general` — admissible because every hop costs ≥1 tick.
For the raw distance layer, jump point search [Harabor11] gives optimal grid paths
with an order of magnitude fewer expansions, using pure symmetry-breaking (no
preprocessing, no learning).

**Why it should work.** The sparse terminal reward is the known weakness of every
material-based evaluator: a bot that hoards land loses to one that quietly walks a
stack at your base. `T_att` measures the thing that actually ends games. The paper
you have reports that its trained agent's most interpretable attention head is a
*posterior over the enemy general's location* [Straka26 §VII-E] — that is, an
expensive learned approximation of something you can compute exactly.

## 5. Direction D — Real-time search with a tick budget (§4.5)

You get one action per tick, forever, on a graph too large to solve. This is AIMA's
**online search agent** setting (§4.5, 3rd ed §4.5), and LRTA\* is the book's answer:

```
H(s) ← max( H(s),  min_a [ c(s,a,s′) + H(s′) ] )
```

Maintain `H` over tiles — "ticks of work to make this tile mine" — and update it
every tick. Two properties matter here: it is **anytime** (expand until the budget
is gone, then commit), and it **improves on repeated traversal**, which is exactly
what a 500-tick game on one fixed map gives you.

Refinements worth reading, in order: RTAA\* updates `h` for every expanded state in
one sweep after a bounded-lookahead A\* [Koenig06]; Adaptive A\* is the two-page
`h`-update trick underneath it [Koenig05]; the systematic comparison of which
variant to reach for is [Koenig09]. Because territory flips constantly, your edge
costs change every tick — that is the incremental-search regime, and [Koenig09]
covers it directly. Min-Max LRTA\* extends the whole framework to an adversary
[Koenig01], bridging this direction to §3 above.

## 6. Direction E — Reformulate the action space (§3.1) ★★ highest leverage

§3.1 stresses that *the choice of state space and actions is a design decision*.
Here it is the decision that matters most. With `|A_i| ≈ 400`, a 3-ply search is
6.4×10⁷ nodes; you have 0.5 s. Search over **plans** instead:

```
Π = { expand-nearest, gather-to-general, rush-enemy-general,
      defend-threat, take-castle, cut-territory }
```

Each `π ∈ Π` is a deterministic policy `S → A`. Search over *which plan to commit to
for the next k ticks*. Branching drops from ~400 to 6; with depth 3 and k = 8 that
is 6³ = 216 nodes × 8 simulated ticks — trivially inside budget, and it buys you a
24-tick horizon instead of a 3-tick one.

This is Puppet Search [Barriga15]: take scripts, expose a few choice points, search
over those. Related: hill-climbing over per-unit script assignment [Churchill13]
(which is §4.1 local search applied to a game tree), the three-layer
goals→plans→actions decomposition [Stanescu14], and the shipped commercial
architecture in [Churchill15].

**Why this specifically beats Human.exe.** [Straka26 §VI-A] describes it as running
"a fixed priority list of about thirty precomputed plans… executes the first one
that applies". It has a plan library and a *hardcoded arbiter*. It cannot ask *"if I
expand now, can they punish me in six ticks?"* — only *"does the defend rule fire?"*.
Same library, searched instead of ordered, is a strictly stronger decision procedure.
See `../papers/human-exe-decomposed.md`.

## 7. Direction F — Evaluation function design (§5.4)

§5.4.2's weighted linear form, `Eval(s) = Σ wᵢ φᵢ(s)`, with features derived from §1:

```
φ₁ = ΔL           land difference
φ₂ = ΔA           army difference
φ₃ = Δρ           production-rate difference   ← the derivative; §1 gives you the weights
φ₄ = T_att(opp) − T_att(me)                    ← §4, the race term
φ₅ = frontier size difference                  ← expansion potential
φ₆ = defence slack at my general               ← §8's CSP gives you this for free
```

Note `φ₃`'s weights are *derived*, not tuned: §1 says a producer is worth 25 tiles.
Deriving weights from the rules rather than fitting them is what keeps this
learning-free and is a defensible contribution in its own right.

Two chapter-5 warnings apply sharply here:

- **Horizon effect (§5.4.2).** A bot searching 3 ply will happily walk a stack next
  to a contested tile and evaluate before the exchange resolves.
- **Quiescence (§5.4.2).** Define a position as quiescent iff no stack of size ≥ τ
  sits within 1 tick of an enemy-owned or contested tile, and extend the search
  through non-quiescent positions. Cheap to test, and it will fix a large class of
  blunders.

## 8. Direction G — Defence as a CSP (§6.1–6.4)

Each tick, the threats are exactly knowable (no fog). Formulate defence as a CSP:

```
Variables    X₁…X_m, one per threat (an enemy stack, its target, and its ETA)
Domains      D(X_j) = { my stacks s : dist(s, target_j) ≤ eta_j  ∧  army(s) > threat_j }
Constraints  AllDiff(X₁…X_m)                        a stack defends one thing
             general-cover: the general's threat, if any, must be assigned
```

Solve with backtracking plus MRV, degree tiebreak and forward checking (§6.3), or
AC-3 first (§6.2). Two payoffs:

1. The assignment tells you which stack to move — a concrete action.
2. **Unsatisfiability is an evaluation feature.** If AC-3 wipes out a domain, you
   are provably losing that tile, and `φ₆` should say so. Using *constraint
   propagation as an evaluation term* is an idea I have not seen in the RTS-search
   literature and is worth writing up.

When you must re-solve every tick, min-conflicts (§6.4) repairs last tick's
assignment in near-constant time instead of re-solving from scratch.

If you later route several stacks simultaneously and they interfere, that is the
same two-level structure as conflict-based search — high level resolves conflicts,
low level replans one agent under added constraints [Sharon15, Sharon12]. Use
[Stern19] to state precisely which MAPF variant your problem is.

## 9. Direction H — Proving forced wins (§4.3) ★ the most novel angle

§4.3 (searching with nondeterministic actions) gives AND–OR trees: OR nodes are my
choices, AND nodes are outcomes I must handle *all* of. Model the opponent's
simultaneous move as the nondeterminism. A solution subtree is then a **guaranteed**
win, not an estimate.

```
Solve( s, depth ) :
    if captured(s)                  return WIN
    if depth = 0                    return UNKNOWN
    for each a ∈ A_me(s):                                        (OR)
        if for all b ∈ A_opp(s):  Solve( result(s,a,b), depth−1 ) = WIN   (AND)
             return WIN
    return UNKNOWN
```

Feasible only because the state is fully observable and the transition deterministic —
i.e. **only in your variant.** Restrict to the endgame (one stack, `k ≤ 8` ticks,
opponent moves pruned to plausible defences) and it is small. A proved win short-
circuits the evaluation function entirely, which is worth more than any eval
refinement: it converts "probably ahead" into "wins by force in ≤ k".

Natural extension: run it as a *safety check* on your own general every tick — "can
they force a capture within k?" — and if so, override the plan search. That is a
correctness guarantee no evaluation function can give you.

---

## Suggested order

| # | Build | Directions | Measurable against |
|---|---|---|---|
| 1 | Optimal opening (DFBB) | A | `L(50)` vs `expander` |
| 2 | Race-heuristic greedy bot | C, F | beats `expander` head-to-head |
| 3 | Plan library + 3-ply serialised search | E, B | beats #2 |
| 4 | Quiescence + sandwich-adaptive depth | B, F | beats #3 |
| 5 | CSP defence layer | G | fewer general captures conceded |
| 6 | AND-OR endgame prover | H | converts won positions faster |

Everything is measurable with what is already built:

```bash
python server/local_server.py --p0 strategies/new.py --p1 strategies/old.py --games 200 --headless
python viewer.py --p0 strategies/new.py --p1 strategies/old.py --tick 0
```

Keep every version. Elo between your own checkpoints is the only progress metric
that means anything before you face an external opponent.

---

## References

**Textbook**

- **[R&N]** S. Russell and P. Norvig. *Artificial Intelligence: A Modern Approach*,
  4th ed. Pearson, 2021. Ch. 3 (solving problems by searching), 4 (search in complex
  environments — §4.1 local search, §4.3 nondeterministic actions/AND–OR, §4.5 online
  search agents/LRTA\*), 5 (adversarial search — §5.2 minimax, §5.3 alpha–beta,
  §5.4 imperfect real-time decisions/evaluation functions/quiescence), 6 (CSPs —
  §6.2 AC-3, §6.3 backtracking with MRV and forward checking, §6.4 min-conflicts).

**This game**

- **[Straka26]** M. Straka, V. Lisý and M. Schmid. Superhuman AI for Generals.io
  Using Self-Play Reinforcement Learning. arXiv:2606.23348, 2026. — the paper in this
  repo; §IV rules, §VI-A the Human.exe description, §VII-E the general-location head.
- **[Straka25]** M. Straka and M. Schmid. Artificial Generals Intelligence: Mastering
  Generals.io with Reinforcement Learning. arXiv:2507.06825, 2025. — zero v3.
- **[Bhatia23]** A. Bhatia, A. L. Davis, S. Ghosh and G. Sukthankar. Generally Genius:
  A Generals.io Agent Development and Data Collection Framework. *AIIDE* 19(1):400–406,
  2023. — the only paper analysing a rule-based agent in this game (Flobot).

**Adversarial search under simultaneity**

- **[Saffidine12]** A. Saffidine, H. Finnsson and M. Buro. Alpha-Beta Pruning for Games
  with Simultaneous Moves. *AAAI*, 2012.
- **[Churchill12a]** D. Churchill, A. Saffidine and M. Buro. Fast Heuristic Search for
  RTS Game Combat Scenarios. *AIIDE*, 2012. — ABCD.
- **[Churchill12b]** D. Churchill and M. Buro. Incorporating Search Algorithms into RTS
  Game Agents. *AIIDE Workshop*, 2012.

**Action-space abstraction**

- **[Barriga15]** N. A. Barriga, M. Stanescu and M. Buro. Puppet Search: Enhancing
  Scripted Behavior by Look-Ahead Search with Applications to RTS Games. *AIIDE*, 2015.
- **[Churchill13]** D. Churchill and M. Buro. Portfolio Greedy Search and Simulation for
  Large-Scale Combat in StarCraft. *CIG*, 2013.
- **[Churchill15]** D. Churchill and M. Buro. Hierarchical Portfolio Search: Prismata's
  Robust AI Architecture for Games with Large Search Spaces. *AIIDE*, 2015.
- **[Stanescu14]** M. Stanescu, N. A. Barriga and M. Buro. Hierarchical Adversarial
  Search Applied to Real-Time Strategy Games. *AIIDE*, 2014.

**Resource-constrained sequencing**

- **[Churchill11]** D. Churchill and M. Buro. Build Order Optimization in StarCraft.
  *AIIDE*, 2011. — DFBB with an admissible bound from accumulation rates.

**Real-time and incremental search**

- **[Koenig01]** S. Koenig. Minimax Real-Time Heuristic Search. *Artificial Intelligence*
  129(1–2):165–197, 2001.
- **[Koenig05]** S. Koenig and M. Likhachev. Adaptive A\*. *AAMAS*, 2005.
- **[Koenig06]** S. Koenig and M. Likhachev. Real-Time Adaptive A\*. *AAMAS*, 2006.
- **[Koenig09]** S. Koenig and X. Sun. Comparing Real-Time and Incremental Heuristic
  Search for Real-Time Situated Agents. *JAAMAS* 18(3):313–341, 2009.

**Pathfinding and abstraction**

- **[Harabor11]** D. Harabor and A. Grastien. Online Graph Pruning for Pathfinding on
  Grid Maps. *AAAI*, 2011. — jump point search.
- **[Harabor14]** D. Harabor and A. Grastien. Improving Jump Point Search. *ICAPS*, 2014.
- **[Sturtevant05]** N. Sturtevant and M. Buro. Partial Pathfinding Using Map Abstraction
  and Refinement. *AAAI*, 2005.
- **[Bulitko07]** V. Bulitko, N. Sturtevant, J. Lu and T. Yau. Graph Abstraction in
  Real-time Heuristic Search. *JAIR* 30:51–100, 2007.

**Multi-agent routing**

- **[Sharon12]** G. Sharon, R. Stern, A. Felner and N. R. Sturtevant. Conflict-Based
  Search for Optimal Multi-Agent Path Finding. *AAAI*, 2012.
- **[Sharon15]** G. Sharon, R. Stern, A. Felner and N. R. Sturtevant. Conflict-based
  search for optimal multi-agent pathfinding. *Artificial Intelligence* 219:40–66, 2015.
- **[Stern19]** R. Stern et al. Multi-Agent Pathfinding: Definitions, Variants, and
  Benchmarks. *SoCS*, 2019.
- **[Sturtevant06]** N. Sturtevant. Improving Collaborative Pathfinding Using Map
  Abstraction. *AIIDE*, 2006.

**Spatial reasoning and context**

- **[Critch20]** L. Critch and D. Churchill. Combining Influence Maps with Heuristic
  Search for Executing Sneak-Attacks in RTS Games. *IEEE CoG*, 2020.
- **[Ontanon13]** S. Ontañón, G. Synnaeve, A. Uriarte, F. Richoux, D. Churchill and
  M. Preuss. A Survey of Real-Time Strategy Game AI Research and Competition in
  StarCraft. *IEEE TCIAIG* 5(4):293–311, 2013.

**Out of scope, listed for completeness**

- **[Chung05]** M. Chung, M. Buro and J. Schaeffer. Monte Carlo Planning in RTS Games.
  *CIG*, 2005. — sampling-based. AIMA 4th ed. covers MCTS in §5.4, so it is arguably
  in scope, but confirm with your instructor before relying on it.

PDFs for everything above are in [`../papers/`](../papers/); the annotated index is
[`../papers/README.md`](../papers/README.md).
