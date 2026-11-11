# Human.exe — what actually exists, and how to beat it

## There is no Human.exe paper

I checked GitHub, its ladder profile (`generals.io/profiles/human.exe is a bot` — a
JavaScript shell serving no data to crawlers), and community channels. **No paper,
no source, no writeup.** It is a closed community bot.

The tell is in the paper you already have: every other agent gets a citation —
Flobot `[15]`, HASP `[16]`, zero v3 `[17]` — and Human.exe gets none. It is
introduced as "the community-developed Human.exe".

**So §VI-A of `../2606.23348v1.pdf` is the most detailed public description in
existence.** It is one paragraph. Fortunately it is a dense one, and it names every
algorithm. Below is that paragraph decomposed into something you can build.

## Where it sits

| agent | OpenSkill |
|---|---|
| Average Joe (the RL agent) | 118 |
| strongest active human | 101 |
| zero v3 (prior SOTA, CNN + PPO) | 74 |
| **Human.exe** | **70** |
| top-100 human cutoff | 68 |

Human.exe is roughly a top-100 human, and it was the top bot on the ladder until
zero v3 (which only beats it 54.8% over 529 games). **Beating it with pure search
is a real result, not a warm-up.**

---

## The paragraph, decomposed

| what the paper says | the algorithm | your situation |
|---|---|---|
| "an explicitly maintained belief state… a belief model handles the fog of war… enemy armies projected along likely routes… backward search through the fog… a conservation accounting layer" | particle-filter-ish belief tracking | **Delete all of it.** ~40% of the described machinery is fog handling, and your mod removes fog. You get its hardest subsystem for free. |
| "a fixed priority list of about thirty precomputed plans… executes the first one that applies" | rule arbiter (AIMA ch. 2 reflex agent with state) | **This is its weakness — see below.** |
| "most of the work happens in a perception pass at the start of the turn" | recompute derived features once per tick, cache | Straightforward. Distance fields, frontier sets, threat maps. |
| "army consolidation is solved as a rooted prize-collecting Steiner tree, with a binary search over the cost basis to stay within a turn budget" | **rooted PCST**, NP-hard, solved approximately; binary search on the prize/cost tradeoff to hit a move budget | "Gather my scattered army to one point in ≤ K ticks, collecting the most army per tick spent." |
| "territorial expansion is solved as a min-cost flow on a contracted graph whose nodes are contiguous regions of tiles" | **min-cost flow** on a region-contracted graph | Contract the board into connected regions, push "expansion effort" through it. The contraction is what makes it tractable. |
| "the candidate paths from the two are combined with a multiple-choice knapsack dynamic program" | **MCKP** — pick one option from each group under a shared budget | Gather-plans and expand-plans each produce candidates with (cost in ticks, value in army/land); MCKP picks the best compatible set. |
| "short-range engagements are resolved with a simultaneous-move minimax" | matrix-game minimax at each node | Exactly `saffidine2012_alphabeta_simultaneous_moves_AAAI.pdf`. |

Every one of those is in scope for you: graph algorithms, DP, and adversarial
search. No learning anywhere.

---

## Two structural advantages you have

**1. No fog.** Its belief model — route projection, branch hedging, backward search
through fog, conservation accounting — is a large fraction of its complexity and
buys it nothing against you, because you both see everything. You skip building it
*and* it loses the edge it gave against humans.

**2. Its plan choice is not searched.** "A fixed priority list… executes the first
one that applies" is a hardcoded arbiter. It cannot reason about *"if I expand now,
can they punish me in 6 ticks?"* — it only asks *"does the defend rule fire?"*.

That is exactly the gap Puppet Search closes: keep the plans, but **search over
which plan to commit to**, evaluating each by simulating a few ticks ahead against
the opponent's plausible plans. Same plan library, a decision procedure with
lookahead instead of a priority list.

> Read `barriga2015_puppet_search_AIIDE.pdf` (7 pages) with this framing. It is the
> single highest-leverage paper in the directory for beating Human.exe.

---

## What you *can* actually read

Since Human.exe is closed, the nearest readable rule-based generals.io agents:

- **Flobot** — a real open-source rule-based generals.io bot, and the subject of an
  AIIDE 2023 paper. Much weaker than Human.exe (the paper you have notes Human.exe
  "wins close to 100% of games" against Flobot-derived agents), but it is readable
  and paper-backed.
  - Paper: `bhatia2023_generally_genius_flobot_generalsio_AIIDE.pdf` — analyses
    Flobot's win rate by starting quadrant and proposes a pathfinding fix.
  - Source: https://github.com/flo-lan/generals.io-Bot (JavaScript, last pushed 2019)
- **harrischristiansen/generals-bot** — a larger Python collection of automated
  clients with a live viewer. https://github.com/harrischristiansen/generals-bot
- **zero v3** — `straka2025_artificial_generals_intelligence_zerov3.pdf`. Learned,
  so out of scope to copy, but §on the environment and the bot landscape is useful
  context, and it is the agent Human.exe lost to.

## Honest caveat

Human.exe's real strength is very likely in details the paragraph does not
record — opening book, exact plan priorities, tuned constants, engine-level move
timing. You are reverse-engineering from one paragraph. Treat the decomposition
above as *a* design consistent with the description, not *the* design.

The upside: you are not trying to clone it. You are trying to beat it, in a variant
where its biggest subsystem is irrelevant.
