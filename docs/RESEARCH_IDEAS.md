# Search and strategy ideas log

This file is the running ledger for every new bot idea raised or tested in this
project. Results are kept with the candidate rather than used to erase failed
experiments. No model becomes `final.py` unless the tournament shows a repeatable
gain over v11 in seat-balanced games.

## Current game facts

- Both players choose actions simultaneously; the transition is deterministic.
- A move can be `PASS`, so a source can collect while other stacks keep acting.
- Generals and owned castles add one army every two ticks; every owned tile adds
  one army every fifty ticks. A producer is worth 25 tiles of production before
  capture cost and denial value are counted.
- The board is fully visible and the viewer accepts queued paths. A queue is a
  commitment that can buy the bot more decision time, but it can also go stale.
- Search must model both the value of waiting and the opponent's opportunity to
  expand, take the same castle, intercept, or attack home.

## Idea register

| ID | Idea | Search model / source | Status and test |
|---:|---|---|---|
| 1 | **Bucket legal moves before adversarial search.** Group actions by source role, target role, send fraction, and tactical contact; keep one or a few representatives per group. Search the resulting compact simultaneous matrix. | Action abstraction in AIMA §3.1; `strategies/RESEARCH-DIRECTIONS.md` §6; Churchill/Buro RTS portfolio search. | Implemented as `v26_bucketed_action_probcut.py`: keeps five bucket representatives in addition to the v25 macro candidates and keeps its ProbCut fit/model separate. Opening smoke test passed. We stopped the unrelated round robin after 89/250 games; those rows remain archived without standings. v26 has not had a focused matchup yet. |
| 2 | **Run A* inside each strategic bucket.** Use weighted A* to propose target-specific route representatives; try ε=2, 1.5, then 1 as an anytime refinement while retaining the best result. | Existing `strategies/wastar.py`; AIMA §3.5; ARA* paper in `papers/`. | v26 starts with exact/weighted A* representatives; staged refinement remains a separate ablation. |
| 3 | **ProbCut after bucketing.** Fit shallow/deep values on the compressed matrix, then spend deeper search only on rows whose predicted upper bound can still change the choice. Calibrate per search horizon/context. | Buro, *ProbCut: An Effective Selective Extension of the α-β Algorithm*; `to_show/04-probcut.md`. | v26 applies the cutoff to bucket rows; calibration and tournament required. Never prune a one-move general capture. |
| 4 | **Event-driven producer race, not a fixed number of PASS plies.** Track a source, target, route, launch-ready event, first-contact event, and opponent race. Fast-forward deterministic production between events; branch on launch, intercept, defend, or switch target only when the board creates a decision. Keep the selected plan across ticks while its source and target remain valid. | Deterministic production rules; semi-Markov / durative-action planning; Puppet Search scripts. | Designed from v24/v25 traces: waits could oscillate or be recomputed to a different target. Implement as a separate candidate after v26. |
| 5 | **Compare castle investment with land expansion.** Estimate the units spent, expected time to capture, time remaining to earn back the cost, and denial value. A castle is not automatically better than cheap land; choose the crossover from current board facts. | Production calculus in `strategies/RESEARCH-DIRECTIONS.md` §1. | Add to event-race candidate; evaluate against no-castle and castle-rusher controls. |
| 6 | **Give both players a portfolio of scripts.** Compare expand, wait-at-producer, capture-castle, charge-general, defend, intercept, and pass as simultaneous macro-actions. Expose choice points and replan at events instead of locking a whole fixed script. | Puppet Search (Barriga, Stanescu & Buro, AIIDE 2015); `strategies/RESEARCH-DIRECTIONS.md` §6. | v22–v25 prototypes tested; the fixed-horizon versions were not promoted. Keep event-driven form as the next design. |
| 7 | **Use safe queued moves as computation time.** Queue a short prefix only while every source is owned and the route is outside enemy reach. Search the predicted endpoint while those actions execute; discard the forecast if local state changes. | `ideas.txt` Ideas 2–3; Cserna & Ruml, Slo’RTS (ICAPS 2017); viewer queue. | v18 prototype lost its seed batch. Revisit after v26/v27 with invalidation counts. |
| 8 | **Keep human input responsive during bot search.** Separate input-queue synchronization from the match-state lock so key presses never wait for `agent.act()`. | Direct control-path analysis of `viewer.py` → `Match.advance()` → `Match._action_for()`. | Implemented in `server/local_server.py`; verify with a deliberately slow mock bot. |
| 9 | **Allocate search by decision uncertainty.** Use bucket-score ties or value-of-computation estimates to spend time where representative moves are hard to rank, and act quickly when one bucket dominates. | Russell & Wefald value of computation; `ideas.txt` Idea 1. | Candidate feature only; needs fixed-compute tournament against uniform allocation. |
| 10 | **Add a forcing-move / quiescence extension.** If the bucket matrix has a general capture, adjacent general threat, castle race, or launch interception, search through that event before evaluating. | Chess quiescence and selective extension; Buro ProbCut caveat in `to_show/04-probcut.md`. | Required safeguard for v26; test tactical miss rate explicitly. |
| 11 | **Let game phase emerge from available plans.** Compare producer surplus, safe frontier, castle ROI, and attack ETA instead of switching on an arbitrary tick number. | Rule-derived features in `strategies/RESEARCH-DIRECTIONS.md` §§1, 7. | Planned for event-race model; no phase thresholds added yet. |
| 12 | **Test seat and map effects with focused paired games.** Swap seats on each chosen seed and keep per-game outcomes. | Experimental design; prior batches show strong seat effects. | We cancelled the 250-game round robin after 89 games. Partial rows remain in `models/tournaments/rr_v1_v25.jsonl`; no standings, summary, or promotion will be inferred from them. Use focused rusher tests instead. |
| 13 | **Improve path commitment only after bucketing.** Compare 1/3/6 queued actions for each chosen bucket at equal compute, and count stale-queue cancellation. | `ideas.txt` Idea 3; prefix commitment / real-time search. | Planned ablation; not inferred from v18's two losses. |
| 14 | **Learn ProbCut uncertainty by context, not one global line.** Features can include shallow bucket value, target type, wait ETA, source producer, opponent ETA, and bucket margin; validate on held-out map seeds. | Multi-ProbCut (Buro); JAX MLP already available in repo. | v20–v25 fits retained; v26 gets its own pairs/model to avoid mixing horizons. |
| 15 | **Use opponent-specific threat buckets.** Cluster replies into pass/expand, same-target race, source interception, home defense, and direct general attack. Do not spend response branches on many equivalent moves. | Simultaneous-move matrix search; `strategies/RESEARCH-DIRECTIONS.md` §3. | Included in v26 bucket categories; inspect representative coverage in tournament logs. |
| 16 | **Evaluate target denial and route fragility.** An objective's value includes how much it slows the opponent and whether a queued route can be invalidated by a local capture. | Exact visibility and deterministic transition; queue protocol. | Included as bucket metadata; future opponent model ablation. |

| 17 | **Live simultaneous race matrix.** Recompute both players’ expansion, rush, defense, and intercept scripts from the current state every tick; compare joint choices rather than extending a stale turn-by-turn line. | Simultaneous/durative action abstraction; Puppet Search and AHTN planning. | v27 implemented the live matrix and v29 added a selective second response ply. v27’s focused 6-game direct-rusher set was 4–2; v29 lost the seed 92020 seat-1 direct-rusher game. Kept as separate candidates. |
| 18 | **Route-aware reserve guard and charge plan.** Project a producer stack’s wait-to-launch and test whether the home reserve can still meet a rush along the actual route; generate the bot’s own producer-to-general wait/charge plan. | Producer race calculus + A*; direct-rusher trace analysis. | v27/v28 candidate. A false reserve threat can freeze expansion; v29’s selective reply search was too slow (~0.56s on one sampled node) and did not fix seat-1 loss. |
| 19 | **Countergrowth race projection.** During both launch wait and march, project general/castle production and land production on every defended route tile; do not assume that waiting only strengthens the attacker. Replan against observed state each tick. | Deterministic game dynamics; AHTN’s simultaneous/durative planning model. | v30 candidate implemented the projection but lost both seed-92020 direct-rusher seat tests (152 and 147 ticks); stopped before the larger batch because the initial evidence was negative and the nested search was slow. This is a failed candidate, not a promotion. |
| 20 | **Attack route choice under projected growth.** Evaluate multiple paths using arrival-time garrisons rather than selecting a route solely from current garrisons; retain a lower-risk fallback route if the cheapest path crosses a growing producer. | A* with time-dependent edge costs; route fragility / race slack. | Hypothesis derived from the v30 failure; not yet implemented. |
| 21 | **Check home reserve through the opponent’s wait window.** A producer-based threat is not harmless merely because it needs more charging ticks. Evaluate the home departure against projected home income and route attrition through launch and arrival; replan on the next observed tick. | Exact production and simultaneous movement rules; counterexample in `_safe_to_depart_home` (`wait > 0` returned safe). | Implemented as v31 on top of v30. One direct-rusher test (seed 92020, challenger seat 0) still lost at tick 152 after 124s; second seat aborted due excessive decision time. Not a fix yet. |
| 22 | **Learn a branch rejector, not a move policy.** Use the existing simultaneous response matrix and A* route proposals; bucket legal actions by role first. Train a small classifier on exact shallow matrix labels to identify optional branches whose worst-case result is dominated. Keep PASS, A* principal route, waits/charges, defense/intercepts, producer/general captures, and one representative fallback protected. Uncertain means keep. | Finkelstein & Markovitch, *Learning to Play Chess Selectively by Acquiring Move Patterns*; ProbCut; `to_show/04-probcut.md`; `strategies/RESEARCH-DIRECTIONS.md` §§3, 6. Learned planning pruning warns that bad pruning rules can remove all solutions. | Implemented as `models/v32_learned_branch_pruning.py` + `train_learned_pruner.py`. Dataset: 4,065 extra-branch examples from 180 held-out-by-map seeds; labels from exact one-ply simultaneous response matrices. At a 0.90 cutoff, held-out precision was 92.3% (65 predicted skips); at a conservative high-precision gate it selected too few branches to matter. Live sample count: 5.11 bucketed candidates per side vs 5.06 after the experimental classifier. The bucket-only ablation had the exact same six rusher results as learned pruning (5/6), so the learned cutoff has no measured playing-strength contribution yet. Keep experimental and do not promote. |
| 23 | **Use weighted A* alternatives as route branches, then let the learned cutoff reject redundant routes.** Compare ε=1, 1.5 and 2 route proposals while retaining the current A* principal route. | Weighted A* / ARA* in `strategies/wastar.py` and the supplied search directions. | Not implemented: the learned pruner has not yet shown sufficient selective precision to safely prune additional route variants. |
| 24 | **Time-dependent resource-constrained A*.** A route edge consumes one movement unit and the target garrison grows until combat; keep Pareto labels `(elapsed ticks, army remaining)` and only return a target when the route is actually affordable. | Resource-constrained shortest path; ARA*/weighted A* in the supplied `papers/`; exact transition order in `genmod/env.py`. | Implemented in `strategies/wastar_timegrowth.py`; on 72 sampled states it cut mean expansions 31.6→22.6 (ε=1), but target-found rate was 45.8%→44.4%. The v33 bot scored 3/6 versus the direct rusher, below v32’s 5/6. It fixes an infeasible-general-path bug but is not a strength upgrade. |
| 25 | **Rank objectives by producer payback rather than raw objective value.** Estimate remaining production value, travel time, and army surviving at capture; retain enemy-general capture as a forcing exception. | Projected producer income from the game rules and target values in `strategies/wastar.py`. | v35 payback variant scored 3/6 in the same direct-rusher seed/seat/cap suite; epsilon 1.5 changed no v33 outcomes (also 3/6). Payback expanded 143.7 nodes on average vs 31.6 for baseline in the 72-state audit, so reject this version. |
| 26 | **Simultaneous macro-scenario rollout (non-minimax).** Generate hold, expand, castle-wait, general-charge and tactical scripts for both sides; apply each script pair simultaneously through the exact simulator; choose the own script by weighted expected score plus lower-tail risk, execute one tick, then replan. | Simultaneous move games; scenario/planning horizon. Contrast with AHTN/Puppet Search and regret/Exp3 MCTS. | v33 112-tick horizon and v34 224-tick horizon are in `models/`. Across two seeds, both scored 1 win/1 draw/2 losses in four seat-swapped games each, with a 220-tick cap. Games take 15–65 seconds; extending the horizon did not change these outcomes. No promotion. |
| 27 | **Simultaneous-move MCTS with regret matching or Exp3 at each node.** Sample joint actions from each side’s independent bandit policy; update regrets from rollout returns instead of solving every response matrix or serializing moves. Use macro-action progressive widening to include long waits and rush scripts. | Lisý et al., *Convergence of MCTS in Simultaneous Move Games* (NeurIPS 2013); Bošanský et al., *Algorithms for Computing Strategies in Two-Player Simultaneous Move Games* (AIJ 2016). | Research alternative identified; not implemented yet. This is the most direct next search-family comparison after the scenario rollout. |
| 28 | **Producer-to-General launch envelope.** Enumerate wait durations and candidate routes; calculate when each producer can launch a viable charge and what force reaches the General. | Exact `genmod.step`, production phase, route simulation. | Proposed|
| 29 | **Deadline and slack race evaluator.** Compare attack arrival against the defender's earliest effective mobilization/interception time. | Temporal planning and race analysis. | Proposed; test false alarms and missed lethal attacks on saved charge replays. |
| 30 | **Strategic macro-state graph with event jumps.** Search between launch-ready, first-contact, capture, and threat-deadline events while preserving opponent choices during skipped time. | Semi-Markov/event-driven planning; AHTN and Puppet Search. | Proposed; abstraction correctness and skipped-event audits required. |
| 31 | **Audit Pareto time/resource route labels.** Keep nondominated arrival-time and remaining-army routes, including defender growth and exact combat. | Resource-constrained shortest paths; `strategies/wastar_timegrowth.py`. | Proposed audit of existing route work; no gameplay claim. Test feasible-goal recall and exact replay agreement. |
| 32 | **Robust route set and principled route switching.** Retain fast, strong, low-risk, and fallback routes; switch only when a meaningful event changes plan value. | Multiobjective pathfinding and receding-horizon planning. | Proposed; log route switches and oscillations. |
| 33 | **Producer break-even horizon.** Compare castle capture cost and delay against future production, denial, route utility, and missed attack timing. | Game income rules and economic payback. | Proposed; compare castle-first with direct charge from matched states. |
| 34 | **Simultaneous-move MCTS with regret matching/Exp3.** Sample a joint action at each simulated tick and update each side's strategy from rollout return; progressively widen macro-actions. | Lisý et al. (NeurIPS 2013) | Not implemented; use exact joint transitions and test against simultaneous restricted-game solutions. |
| 35 | **Double-oracle / PSRO over macro scripts.** Solve a small simultaneous script payoff matrix, add best-response scripts, and repeat when new responses expose weaknesses. | Restricted-game solving and policy-space response oracles. | Not implemented; compare with fixed opponent populations at equal engine calls. |
| 36 | **Event-driven hierarchical plans.** Search tasks such as secure producer, accumulate, charge, intercept, and resume expansion, with explicit interruptions. | AHTN, Puppet Search, ABCD; supplied RTS papers. | Not implemented; must preserve simultaneous actions while jumping between decision events. |
| 37 | **Adversarial receding-horizon MPC.** Search simultaneous plan pairs and replan from the observed state; replace fixed rollout priors with an adversarial or calibrated opponent set. | Model-predictive control; v33/v34 scenario rollout ablations. | Not implemented; first diagnose v33/v34 losses as horizon, script coverage, evaluation, or compute failures. |
| 38 | **Adversary population and counterexample growth.** Maintain diverse expansion, direct-rush, delayed-charge, castle-rush, defend, and intercept scripts; add a response after each loss. | PSRO / adversarial scenario coverage. | Proposed; reserve held-out variants and wait times for evaluation. |
| 39 | **Strategic-state dynamic programming.** Abstract the full board into producers, charge-source bands, route reachability, reserve, and event countdowns. | State abstraction and exact small-game validation. | Proposed; measure abstraction aliasing before trusting the value model. |
| 40 | **Factor feasibility, safety, and utility.** First establish that a route can work, then whether it is safe, then how valuable the objective is. | Heuristic search diagnostics and tactical validation. | Proposed; avoid collapsing independent model errors into one opaque score. |
| 41 | **One evaluator for attack and defense.** Use the same exact transition and terminal objective when scoring a charge and its interception. | Zero-sum simultaneous action evaluation. | Proposed; verify every predicted defensive success by exact replay. |
| 42 | **Route-control responses.** Consider taking the launch source, cutting a route, holding a choke point, or forcing diversion instead of only defending the General. | Graph cuts, race planning, tactical search. | Proposed; compare with home-reserve logic on matched threat states. |
| 43 | **Multiobjective route labels.** Preserve ETA, surviving force, defender growth, contested tiles, and interruption risk as separate dimensions. | Pareto search. | Proposed; test whether the chosen route survives exact tactical replay. |
| 44 | **Producer surplus and mobilization value.** Value army and income by whether they can arrive before the decisive event, not by raw stock alone. | Temporal resource planning. | Proposed; compare candidate features against exact charge outcomes. |
| 45 | **Potential fields with depression avoidance.** Add progress-aware treatment of repeated route/state basins without permanently penalizing strategically useful return paths. | RTAA*, depression-avoidance paper in `papers/`. | Proposed; log loops and verify penalties do not block necessary retreats. |
| 46 | **Race-derived game phase.** Derive opening/midgame/endgame from attack slack, reachable producer surplus, remaining targets, and timeout proximity rather than fixed turn thresholds. | Temporal heuristics. | Proposed; ablate against the current phase-free evaluation. |
| 47 | **Timeout-aware strategic value.** Distinguish General captures from land/army tiebreak outcomes and protect a lead when terminal capture is too risky. | Terminal/tiebreak rules in the game engine. | Proposed; report timeout wins separately from captures. |
| 48 | **Cost-sensitive learned branch rejector.** Train a classifier to omit branches only when deeper-label regret is low; protect critical waits, charges, captures, and defenses. | Learned pruning and branch-risk calibration. | Proposed extension to v32; require critical-branch recall and rusher false-prune metrics, not precision alone. |
| 49 | **Contextual, conservative ProbCut.** Fit shallow/deep error by event type and horizon, and prune only when a held-out residual bound cannot change the decision. | Buro ProbCut; existing `probcut_*` artifacts. | Proposed; audit catastrophic false cuts in exhaustive tiny states and rusher positions. |
| 50 | **Role-aware buckets with diversity guarantees.** Preserve branches with distinct source, target, arrival time, force, and response profile instead of only top heuristic actions. | Action abstraction; v32 bucket-only ablation. | Proposed; measure regret and best-response coverage before ML pruning. |
| 51 | **Value-of-computation scheduling.** Spend extra search where action rankings are uncertain, the threat is near, or tactical coverage is incomplete. | Metareasoning papers in `papers/` and `to_show/06-metareasoning.md`. | Proposed; report decision stability, value per engine call, and tail latency. |
| 52 | **Safe queue as a thinking window.** Run background search during validated low-risk prefixes and invalidate predictions when relevant state changes. | Viewer queue and metareasoning. | Proposed follow-up; preserve the viewer's independent input/queue lock. |
| 53 | **Incremental search and route reuse.** Cache route/evaluation/transposition work, then selectively invalidate it after local changes. | ARA*, Anytime Dynamic A*, D* Lite. | Proposed; measure cache hit rate and invalidation cost before retaining the complexity. |
| 54 | **Symmetry canonicalization.** Reuse evaluations and training data across player perspective and valid board symmetries. | Game symmetry and transposition reduction. | Proposed; validate action directions, terrain, seat, and income-phase transforms. |
| 55 | **Counterexample-driven regressions.** Save every important rusher loss as a state/replay with threat visibility, launch time, last safe response, and model error. | Scientific debugging and adversarial testing. | Proposed process improvement; add a regression before changing policy for each counterexample. |
| 56 | **Exact tiny-board oracle.** Solve reduced simultaneous games exhaustively to check abstractions, MCTS, learned cutoffs, and payoff evaluation. | Matrix-game solution / exact restricted games. | Proposed; use as an algorithm test separate from full-match outcomes. |
| 57 | **Paired ablations and held-out adversaries.** Change one factor at a time, swap seats, hold seeds/caps fixed, and evaluate unseen wait lengths and scripts. | Experimental design; strong seat effects in prior logs. | Required for each candidate; small batches diagnose but do not establish a reliable rating. |
| 58 | **Reproducible candidate artifacts.** Version every bot, model, config, command, per-game log, and short result summary. | Repository preservation and model ledger. | Required process; keep `final.py` unchanged until repeated matched gains justify promotion. |


## Tournament rules

- Do not run the cancelled 250-game all-model round robin. Use focused test
  suites chosen to distinguish the current hypothesis.
- Compare a candidate with the current best and relevant controls on the same
  seeds, caps, and board size; swap seats for each seed.
- Save every game, including losses and timeout tiebreaks, under
  `models/tournaments/` as JSONL with seed, seat, winner, ticks, runtime, errors,
  and useful decision diagnostics.
- Report results by opponent and seat. Small batches are diagnostics, not
  reliable ratings; require repeated gains across seeds before promoting.
- Run long-game, direct-rusher, and castle-rusher suites separately so each
  answers a clear question.

## References to the repo's papers

- `papers/buro_probcut_forward_pruning_alphabeta.pdf` and `to_show/04-probcut.md`
- Finkelstein & Markovitch, *Learning to Play Chess Selectively by Acquiring Move Patterns*,
  <https://journals.sagepub.com/doi/10.3233/ICG-1998-21204>
- Fern, Khardon & Tadepalli, *Learning Pruning Rules for Heuristic Search Planning*,
  <https://fai.cs.uni-saarland.de/hoffmann/papers/hsdip14d.pdf>
- Barriga, Ontañón & Buro, *Combining Strategic Learning with Tactical Search in Real-Time Strategy Games*,
  <https://arxiv.org/abs/1709.03480>
- `strategies/RESEARCH-DIRECTIONS.md` (A*, weighted A*, ARA*, simultaneous
  response matrices, action abstraction, producer economics)
- Barriga, Stanescu & Buro, *Puppet Search* (AIIDE 2015),
  <https://ojs.aaai.org/index.php/AIIDE/article/view/12779>
- Cserna, Ruml & Frank, *Planning Time to Think: Metareasoning for On-line
  Planning with Durative Actions* (ICAPS 2017),
  <https://www.cs.unh.edu/~ruml/papers/slorts-icaps17.pdf>

## v36: fundamentals-first rebuild (2026-11-02)

Model, from `genmod/env.py`: one move per player per tick; an all-in walk of
d steps loses d units plus garrisons while the defending general gains about
d/2. Two banking generals therefore cannot rush each other; a rush only wins
against a drained general or a much larger economy. Findings while building
`v36_scripted_guard.py` (each fixed a traced loss):

- Earlier "wins" against `general_rusher` were almost all land tiebreaks at
  220/300/400 caps; record capture vs tiebreak separately (`tools36/eval36.py`).
- Static distance maps that treat castles as walls missed routes through the
  attacker's own castle. Live reverse Dijkstra over walk costs fixed it.
- Counting a nearby stack as reinforcement fails when the policy then walks it
  away. The general must hold alone; other moves must pass `survives()`, an
  exact simultaneous rollout of every immediate launch against `Defender`.
- Our own stacks are not walls: an attacker routes around them. Threat walks
  cap defender-owned cells at 3 units (land), and the race check removes the
  marching stack.
- Launch routes must be recomputed on the board after our move (capturing a
  castle can open a shortcut) and ranked by army minus distance, not size.
- The general never relies on being refilled (strict invariant); otherwise a
  stack can oscillate out and back while each step looks verifiable.
- Opening: tiles pay back in 50 ticks, castles in ~100; never waste a tick.
- Late game: threats can merge, so the margin subtracts merge potential and
  the pad grows with the largest hostile stack (fixed a 2000-tick loss).
- Negative: forward "anchor" strike bases and stack-sourced castles reduced
  expander results (ablation `abl_*` logs); disabled.

## v37: phase planner (2026-11-02)

Hypotheses: (1) most raw moves are equivalent, so plan over ~6 macros and merge
plans with identical rollout outcomes; (2) phases detected from the board
should set the plan set, horizon and value model; (3) choosing by the
position H ticks later, extrapolated to the cap, beats one-step priorities;
(4) a small learned value model scores leaves.

Result: identical-outcome merging cut 6.1 candidate plans to 4.6 on average.
Value model held-out log-loss vs land-only baseline: opening 0.473 vs 0.522,
end 0.088 vs 0.111, mid 0.372 vs 0.359 (no gain in mid-game). Held-out suite:
expander 19/20 vs v36's 16/20; rushers 40/40 for both.

## v38: counterfactual kill window (2026-11-02) — not an upgrade

Hypothesis: v37 waits too long because its attack filters (static margin with
reinforcement, pessimistic race, four-reply rollouts incl. turtle) are
stricter than real defence. v38 launches when exact rollouts against
`Defender` and v36's turtle both capture and an enemy counter-rush does not
win first; it assembles army into a stack at >=50% readiness and raids enemy
tiles once neutral land is gone.

Result vs v37 (seeds 93001-93010, seats swapped, cap 500): 11-9 twice, but
seat-determined (seat 0 won 9/10, seat 1 2/10); one capture. Trace analysis:
against an opponent that banks its general, the general can only spare about
1.5*d - pad (~30) units while the enemy general holds hundreds, so a kill is
impossible without gambling home. Between near-identical bots the land
tiebreak and map asymmetry decide. Not promoted.

## v39: fixes from recorded human wins (2026-11-03)

`viewer.py` now saves every human-vs-bot game to
`models/tournaments/human_games/*.npz`; `tools36/analyse_human.py` replays
one in sim36 and re-runs the bot each tick (v37 reproduced all 403/377
recorded moves exactly). Both human wins: an early tiny hostile walk made
v37 "mid phase" and the planner chose gather_home for ~300 ticks (land 27 vs
63, 59 vs 64); the human then hit the general in two waves (first trades,
second or a merged stack finishes) while their own general sat at 1-12 for
15-20 ticks, uncountered.

Human-style stress bot `adversaries38.GatherRusher` (expand until t0, sweep
into one stack, all-in when the walk beats general + growth): v37 lost 14/60
(t0 = 60/120/200, cap 1000, sim36).

Ablations (losses /60 vs GatherRusher): two-wave cumulative margin 44 (too
pessimistic, turtles); counter-strike on drained general 19 (fires when the
enemy is not really drained); breach mode for castle-walled territory had no
effect (misdiagnosis: v37 had out-expanded 133-29). Root cause of that loss:
the defence rule "hit an adjacent attacker we can beat" moved the general;
moves are simultaneous, the stacks passed each other and the attacker took
the emptied general. v39 adds swap-safe defence (general never steps toward
an adjacent enemy stack; `swap_safe` flag, off for v36/v37): 10/60. A
swap-aware safety check (enemy reply into the vacated cell) gave 12/60 and
was reverted.

Harness (seeds 93001-93010, cap 500): rushers 40/40, expander 18/20 (same as
v37); vs v37 8-12 (seat 0 6/10, seat 1 2/10; mean land -6). Not promoted.

## v40: paper-guided ranked planner (2026-11-03) — failed, kept as an idea

Papers read in full (text extracted from the PDFs): Barriga, Stanescu & Buro
2017 (CNN policy distilled from Puppet Search: 73% top-1, 95% top-2, 3 ms);
Barriga et al. 2015 Puppet Search; Moraes & Lelis 2018 asymmetric action
abstractions (two-step GAB, N=4 unrestricted units, 0.92 vs PGS); Lelis
2017 SSS/SSS+ (PGS+ evaluation: the candidate script acts first, then a
default script); Lisy et al. 2013 SM-MCTS convergence (regret matching/Exp3
converge, UCB does not); Ontanon 2017 CMAB/NaiveMCTS; Straka, Lisy & Schmid
2026 (Generals.io; Human.exe uses Steiner-tree gathering, min-cost-flow
expansion, simultaneous minimax for short fights; tempo before tick%50).

v40 (`v40_ranked_planner.py`, `core40.py`, `gen_labels40.py`,
`train_ranker40.py`): teacher = plan x six-reply matrix of PGS+ rollouts
solved by regret matching (8,071 positions from 240 self-play games, CPU
simulation on 12 cores); ranker = dilated CNN + global pooling + plan
features trained on the Mac GPU (MPS, EMA weights). Held-out ranker vs
random top-2 regret: opening 0.0054 vs 0.0172, mid 0.0070 vs 0.0128, end
0.0036 vs 0.0057. Play time: ranker top-k, regret matching over the kept
plans, 150 ms anytime budget, asymmetric tactical step on contact.

Result: worse than v37. vs v37 4-16 (seat 1 0/10, -65 land); expander
10/20 (v37 18/20); castle rusher 18/20; gather rusher 17/60 losses (v37
14). Ablations were not run. Not promoted; keep as an idea.

v42 (`v42_collect_race.py`, promoted to final 2026-11-03): collection race
margin min_T G + growth(T) + D_us(g, T - 2 - T//10) - D_them(g, T), with
D = greedy budgeted gather (walk lengths sum <= T; Human.exe's
prize-collecting Steiner tree, simplified). Lessons while building it:
counting a departing stack as returning (general moves) and counting a
castle-bound stack before its garrison is paid both lost generals; fixed by
standalone checks on general moves and projecting castle walks to their end.
v43 (`v43_collect_strike.py`): + collect onto the forward castle and strike,
+ per-game parameter adaptation (kappa, castle buffer, race slack). Sim vs
v37 cap 1000, seeds 93011-93040: 40W/20L, 17 wins by capture. Not promoted.

v44 (`v44_reserve_counter.py`): v42 + tiles counted once in the race, a
general reserve (half of every hostile walk, now and after 30/60 ticks of
growth, and of the army the enemy can gather in 30/60 moves), and a
counter-strike when home loses the race. Sim, cap 1000, seeds 93001-93010:
vs v42 13-7, vs v37 19-1; cap 500 seeds 93011-93020 vs scripted bots 82/100
(v42 75/100; SpreadCollector 17-3 vs v42 12-8). Not promoted (v42 kept as final).

v45 (`v45_gated_opponent.py`): v44 + opponent-gated tree (castle gate: full
castle play only if the opponent holds / masses on castles, else cheap
castles only; threat gate: reserve 0.6 + counter if they attacked or walked
their general out, else 0.4). Sim, same seeds: vs v44 10-10, vs v42 12-8,
vs v37 16-4, scripted 79/100. Ablations vs v37 / v44: cheap castle limit 60
-> 17-3 / 10-10; threat gate off -> 15-5 / 9-11. Equal to v44 within noise;
against a castle-less opponent (v37) skipping castles costs a little.

v46 (`v46_opponent_learner.py`): v45's gates driven by learning the opponent.
Papers and how each is used:
- Synnaeve & Bessiere, "A Bayesian Model for Opening Prediction in RTS Games
  with Application to StarCraft", AIIDE 2011; Weber & Mateas, "A Data Mining
  Approach to Strategy Prediction", CIG 2009 -> in-game posterior over
  opponent types (v37's script weights, updated every tick from the observed
  move) plus observed facts (castles, attacks, general walking out).
- Ganzfried & Sandholm, "Game Theory-Based Opponent Modeling in Large
  Imperfect-Information Games", AAMAS 2011; "Safe Opponent Exploitation",
  ACM TEAC 2015 -> best-respond to the inferred type only inside v44's
  safety layer (race margin, reserve, verify()).
- Tavares, Azpurua, Santos & Chaimowicz, "Rock, Paper, StarCraft: Strategy
  Selection in Real-Time Strategy Games", AIIDE 2016; Auer, Cesa-Bianchi,
  Freund & Schapire, "The Nonstochastic Multiarmed Bandit Problem", SIAM J.
  Comput. 2002 (Exp3); Lisy et al. NIPS 2013 -> cross-game Exp3 over four
  parameter arms, persisted when V46_MEMORY=path (live `act` only).
- Churchill & Buro, "Build Order Optimization in StarCraft", AIIDE 2011 ->
  income race: castles while the projected lead at the cap is not safe.
Result (sim, cap 1000, seeds 93001-93010): vs v44 11-9, vs v42 14-6, vs v37
17-3; cap 500 seeds 93011-93020 scripted 81/100 (v44 82, v45 79). Equal to
v44 within noise in-harness; cross-game Exp3 is untested (needs repeated
games against one opponent). Slowest move 1.28 s. Not promoted.

v47 (`v47_bayes_phases.py`): v46 + naive-Bayes opponent classifier over 8
types (rusher, gather_rusher, castle_rusher, expander, hoarder,
spread_collector, general_walker, passive), updated on every inferred
opponent move (kind / toward / from general / size) plus a state feature
every 10 ticks; forgetting 0.95, tempered x0.5, 5% floor. Phases from army
spread (clash, end, mass, spread, expand, build; 3-tick hysteresis). Policy:
posterior-weighted (reserve, castle mode, counter, collect-and-strike) with
phase modifiers. Classifier check (seed 93003): expander -> expander t100-200;
SpreadCollector -> spread_collector t200; banking Rusher reads "passive" until
it launches. Sim: vs v46 8-12, vs v44 11-9, vs v37 17-3 (cap 1000); scripted
82/100 (cap 500). Slowest move 1.7 s. Not promoted.

v48 (`v48_guard_split.py`, kept as is, in the arena): v46 + a home
guard. Guard margin = min_T G + growth(T) + (our best T-2 tiles within T-2 of
home) - (their best tiles within 1.5T of our general, gathered then walked,
minus the walk cost). The guard binds only while the enemy mobilises (biggest
free stack >= max(20, half our home defence) or +50% in 15 ticks, latched 30
ticks): moves may not take it under its pad (small moves tolerated), and the
GUARD branch pulls the best reachable stack home. When calm, the attack force
collects onto the forward castle and strikes (v43's collect-and-strike).
Sim, cap 1000, seeds 93001-93010: vs v46 8-12; vs GatherRusher t0=200 14-6,
t0=120 18-2, SpreadCollector 16-4 (v46 at cap 500: 11-9, 15-5, 17-3). The first
version (all enemy tiles counted, guard always on) went 57/60 vs gather rushers
but 8-32 vs v46 by turtling (land 62 vs 337).

v49 (`v49_kill_window.py`, in the arena): v46 + a kill window. v46 only charges
when ONE stack's walk beats the enemy general, so a spread army never adds up
to an attack. Every tick v49 runs the collection race on the enemy's home from
our side: attack(T) = our walks onto their general, lengths summing to <= T,
each tile once, their cells paid in full plus land growth; or collect-then-
strike: our land army walked onto a forward stack for H moves, then one
all-in walk. defence(T) = their general + growth + everything they can collect
home in T moves (their walks through our cells capped at 3). The smallest T
with attack > defence + 2 + 10% is taken if our home still holds over T+3 with
the committed stacks gone (v44 race profile, reserve for a general half-send)
and a rollout wins against every core36 reply (rusher, castle rusher, pass,
turtle Econ). Once verified the walks run to the end; only home danger or the
remaining hits no longer beating their general + growth cancel it (re-planning
every tick aborted every plan half-way: v49a, 18-22 vs v46).
Sim cap 500: vs v46 32-28 (seeds 93001-93030; captures 14 won / 5 lost);
scripted spread/expand/econ/gather120 65/80 (v46 55/80), captures 43 vs 30;
rusher 39-0-1, castle_rusher 39-1, spread_late 31-9, gather60 34-6 (seeds 93011-93030).
4.1 ms/move on the arena bench. Not promoted.

v50 (`v50_26modified.py`, in the arena; models fitted by `fit_v50.py` ->
`v50_model.json`): v46 with the cited papers' algorithms implemented:
Human.exe (Straka et al. 2026 sec. VI-A) rooted prize-collecting Steiner-tree
gather with binary search on the cost basis; min-cost flow expansion on
contracted neutral regions + multiple-choice knapsack DP; simultaneous-move
matrix game for contact fights solved by regret matching (Lisy et al. 2013);
Puppet ABCD (Barriga et al. 2015 Alg. 1, alternating serialization of
Churchill et al. 2012) over macro scripts vs opponent scripts; SES "risk what
you've won" (Ganzfried & Sandholm 2015) for exploiting the opponent model;
Synnaeve & Bessiere 2011 Bayesian timing model (Gaussian event times +
survival term) and Weber & Mateas 2009 kNN on masked event-time vectors,
multiplied with v46's move likelihood; Churchill & Buro 2011 DFS
branch-and-bound over castle sequences; Tavares et al. 2016 Nash arm over the
arm x type payoff matrix, best response once the type is predicted (Exp3
across games as in v46). Fit: 320 sim games, statistics and regret matching
on the Mac GPU (MPS); Nash = pure "greedy" (expander column binds).
Ablation vs v46, 20 games each, one component on: none 11-9, PCST 13-7,
MCF 5-15, skirmish 11-9, ABCD+SES 10-10, Bayes+kNN 13-7, B&B 9-11, Nash 10-10.
MCF stayed 6-14 after fixing its cost model, a pick-rule bug and a
general ping-pong: greedy one-tile-per-move expansion captures more per tick
on this board, so MCF + MCKP are off by default (V50_ON=mcf).
Result (cap 500): vs v46 35-25 (seeds 93001-93030, seats 17-13 / 18-12;
captures 6 won / 3 lost); scripted spread 14, expand 12, econ 15, gather120
19 (v46: 15, 11, 12, 17), rushers 40/40. Not promoted.

v51 (`v51_fast_defence.py`, promoted to final 2026-11-05) and v52
(`v52_fast_defence_only.py`, v51 without the classifier). Traced v46 loss to
GatherRusher t0=200: a 103-stack walked 22 -> 1 tiles over 20 ticks while v46
kept expanding with 2-army tiles, defended only on single ticks when the race
margin dipped, and collected one stack per tick from distance 4. Fix: approach
detector (hostile mobile stack >= max(15, 0.2 G) closing >= 2 tiles in 5
ticks) commits to defence while need = max_T<=ETA+5 D_them(T) - G - growth(T)
+ 0.15 x attacker > 0: PCST gather rooted at the general with budget ETA - 2,
stacks of 10+ next to the general merge in (any size within 6). Results:
GatherRusher t0=200 19-1 (v46 12-8), t0=120 20-0 (18-2), t0=60 19-1 (16-4),
SpreadCollector 13-7 (14-6), rushers 40/40; vs v46: v51 32-28, v52 32-28.
Classifier: 14 board features -> 10 types (MLP 64-64, MPS); held-out
per-tick accuracy 25% / 53% / 70% / 67% by tick band, online posterior 39% /
66% / 70% at ticks 100 / 200 / 300; main confusions rusher~turtle (both bank
at home), late gather~expander (identical before the sweep), econ~planner.
