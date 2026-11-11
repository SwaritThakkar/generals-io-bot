# Bot versions and match harness

Before replacing `final.py`, copy it here as the next numbered model. Keep all
prior versions in the test pool.

| Model | Approach |
|---|---|
| `v1_wastar_frontier.py` | Initial submission: weighted A* and BFS frontier fallback. |
| `v2_castle_wait.py` | Producer stack waits for a reachable castle before launch. |
| `v3_one_ply_matrix.py` | 3x3 simultaneous move matrix with production and siege-readiness evaluation. |
| `v4_conflict_matrix.py` | Matrix search only when candidate moves can collide. |
| `v5_conflict_queue.py` | Current candidate: conflict search plus short safe path queues. |
| `v6_siege_defense.py` | Adds incoming general siege detection and defense actions. |
| `v7_siege_tree.py` | Adds a simultaneous response tree for siege candidates. |
| `v8_charge_guard_matrix.py` | Adds explicit PASS in the response matrix and protects the home general while a producer stack charges; still expands with other stacks. |
| `v9_narrow_charge_guard.py` | Baseline candidate: based on v7, with an explicit PASS matrix branch and a short charge-window guard on the home general. |
| `v10_experimental_two_ply.py` | Deeper tactical matrix experiment; archived after it ran slowly and lost its four-game v9 head-to-head. |
| `v11_launch_intercept.py` | Current candidate: adds an opponent-launch-source capture option, lowers the missed-small-rush detector threshold, and computes producer charge time with a closed-form tick count. |
| `v12_probcut_charge.py` | Adds a shortest-path wait/charge plan toward the enemy general; first comparison tied v11 across swapped seats. |
| `v13_probcut_charge.py` | Adds a charge-window matrix and learned shallow/deep ProbCut. The first two-ply conversion bug was fixed during iteration; treat early runs as invalid. |
| `v14_charge_eta_probcut.py` | Values the full wait-plus-travel ETA of a general attack, then searches a three-ply principal variation. |
| `v15_charge_intercept_probcut.py` | Adds a large-stack trade against a producer/path while the opponent is still charging. |
| `v16_deeper_probcut.py` | Uses a 1.5σ ProbCut bound and deepens up to four charge rows through a three-ply principal variation. |
| `v17_depth4_probcut.py` | Four-ply principal variation; achieved no measured paired gain and was much slower. |
| `v18_queued_prefetch.py` | Queues up to six safe owned-territory moves, predicts the queued endpoint, and searches that future state in a background worker. Forecasts are discarded when the live endpoint changes locally. |
| `v19_waiting_defense_probcut.py` | Values home-general strength while the opponent is charging and retains the bounded ProbCut search. |
| `v20_ml_probcut.py` | Learns a nonlinear shallow/deep correction from 1,000 tactical boards; tests explicit castle-first wait/launch execution. |
| `v21_ml_probcut_search_only.py` | Isolates the learned ProbCut estimate while retaining the previous policy flow. |
| `v22_macro_wait_probcut.py` | Searches hold/castle/general scripts through 96 exact ticks; retained as the first temporal abstraction experiment. |
| `v23_macro_expand_probcut.py` | Adds full-route expansion, early producer value, seat-swapped macro simulation, and a separate 96-tick ProbCut fit trained on 23×23 boards. |
| `v24_long_wait_probcut.py` | Extends long producer waits and macro ProbCut; trace review found changing wait/target plans can oscillate. |
| `v25_multicastle_wait_probcut.py` | Tries multiple castle waits before the long general charge and uses the v25 learned ProbCut model. |
| `v26_bucketed_action_probcut.py` | Adds up to five legal-move representatives selected from source-role × target-role × send-fraction buckets to the v25 macro candidate set; separate ProbCut artifacts. Smoke-tested, focused test pending. |
| `v27_live_race_matrix.py` | Replans simultaneous action portfolios each tick and compares live joint responses; adds route-based charge and reserve guard. Targeted direct-rusher set: 4/6; castle rusher 3/6; v11 1/6. |
| `v28_live_race_economy.py` | v27 with tapered rush urgency and stronger land/castle economy. Same score totals as v27 on the same small matchup suite; no promotion. |
| `v29_live_replanning_matrix.py` | Adds a second simultaneous response ply at latent producer/general race nodes. Lost the seed-92020 direct-rusher seat-1 game at tick 322; one sampled decision took ~0.56s. |
| `v30_countergrowth_race.py` | Projects defender production during attacker wait and travel along the chosen route. Failed both seed-92020 direct-rusher seats (losses at ticks 152 and 147); full suite stopped. Retained as a failed experiment. |
| `v31_wait_window_reserve.py` | Adds a projected home-reserve check during producer charge windows. Lost seed-92020 as seat 0 at tick 152; took 124s. The other seat was stopped during an overlong decision. Failed and too slow. |
| `v32_learned_branch_pruning.py` | Adds role-bucketed action branches and a small MLP that only rejects optional branches. Trained on 4,065 shallow exact-matrix labels from 180 maps. At the experimental threshold, validation precision was 92.3%; hard tactical/script branches are protected. It scored 5/6 vs `general_rusher` on three matched seeds/caps, but the bucket-only ablation also scored 5/6. The learned pruning has not shown a gain; experimental only. |
| `v32_bucketed_ablation.py` | Same added bucket representatives as v32, with the learned rejector disabled. Scored 5/6 against `general_rusher` on the same seeds and tick caps as v32, showing the measured gain came from widening the bucketed response set, not from the learned cutoff. |
| `v33_timegrowth_astar.py` | Pareto-label A* over elapsed time and remaining army; pays hostile/neutral garrisons and projects defender production. Scored 3/6 vs the direct rusher; reduced expansions on the pathfinding audit but not match losses. |
| `v34_timegrowth_astar15.py` | Same resource-aware route heuristic at ε=1.5. Same 3/6 as v33 on the matched rusher seeds; no win change. |
| `v35_payback_timegrowth_astar.py` | Ranks goals by producer payback, travel, and surviving army. Scored 3/6 vs the direct rusher and expanded much more than the other A* variants. |
| `v33_simultaneous_scenario_rollout.py` | Replaces matrix minimax choice with simultaneous macro-script rollouts scored by opponent-script priors and lower-tail risk. 112-tick horizon; 1W/1D/2L on two seeds, four seat-swapped games. |
| `v34_long_horizon_scenario_rollout.py` | Extends scenario-rollout horizon to 224 ticks. 1W/1D/2L on the same four direct-rusher games; slower and no improvement over v33. |
| `v36_scripted_guard.py` | Rebuilt from the game's fundamentals on `sim36.py` (exact pure-Python replica of the JAX step, checked by `tools36/check_sim36.py`) and `core36.py`. The general alone must beat every hostile all-in walk after its growth during the walk; any weaker move must survive exact simultaneous launch rollouts; own rushes pass a pessimistic race check. ~1–2 ms per move. Held-out seeds 93001–93010 (500 cap, seats swapped): rusher 20/20, castle rusher 20/20, expander 16/20 (losses are land tiebreaks); v11 12/12, v32 bucket-only 12/12. |
| `v37_phase_planner.py` | v36 safety layer plus `core37.py`: dynamic phases (opening/mid/end from the board), ~6 macro plans instead of raw moves with identical-outcome merging, H-tick simultaneous lookahead against four opponent models (24/40/60 ticks by phase, to the cap when near it), per-phase logistic value model (`value37.json`, `train_value37.py`) with income extrapolated to the cap, Bayesian opponent-model weights, robust mean/worst-case choice. Held-out: rusher 20/20, castle rusher 20/20, expander 19/20. |
| `v38_kill_window.py` | v37 plus counterfactual attack rollouts (vs `Defender` and v36 turtle, with counter-rush check), strike assembly and enemy-tile raids. 11–9 vs v37 twice but seat-determined; experimental, not promoted. |
| `v39_counter_planner.py` | v37 plus swap-safe defence (the general never steps toward an adjacent enemy stack), found by replaying recorded human wins. Human-style gather rusher: 10/60 losses vs v37's 14/60; scripted bots same as v37; vs v37 8–12 (seat-dominated, −6 land). Other v39 ideas (two-wave margin, counter-strike, breach mode) are ablation flags, off. |
| `v40_ranked_planner.py` | Failed idea: GPU-trained plan ranker (b reduction), regret-matching plan game, PGS+ rollouts, asymmetric tactical step. Lost to v37 4–16, expander 10/20. Kept for reference. |
| `castle_wait.py` | Test opponent that waits at a producer, then routes to neutral castles. |
| `../strategies/wastar.py` | Existing weighted A* baseline. |
| `../strategies/greedy_eval.py` | Existing one-ply evaluation baseline. |
| `../genmod/bots.py` | Existing expander, random, and pass baselines. |

Example seat-swapped comparison:

```bash
python server/local_server.py --p0 final.py --p1 models/v1_wastar_frontier.py --games 20 --headless --seed 26000
python server/local_server.py --p0 models/v1_wastar_frontier.py --p1 final.py --games 20 --headless --seed 26000
```

The two commands matter because map seeds can favor one seat.

## Current candidate results

Results are small head-to-head batches, not rating estimates. The seeds and
both seat orders are recorded below so later candidates can be compared on the
same positions.

| Opponent | Seeds | Games | `final.py` wins |
|---|---:|---:|
| `expander` | 28000 | 12 | 12 |
| `strategies/wastar.py` | 29000 | 8 | 4 |
| `v1_wastar_frontier.py` | 25000, 26000 | 14 | 8 |
| `v2_castle_wait.py` | 32000 | 4 | 3, 1 draw |
| `v3_one_ply_matrix.py` | 33000 | 4 | 3 |
| `v4_conflict_matrix.py` | 34000 | 4 | 3 |
| `castle_wait.py` (queued castle-collector test policy) | 31000 | 8 | 7 |

The one-ply matrix applied at every state performed poorly and ran much slower.
The current model runs it only when candidate moves can collide. It queues up to
three actions on low-risk routes and replans on later turns or near enemy territory.

## Charge and all-in stress test

`castle_general_rusher.py` waits at a producer, takes a castle when affordable,
then repeatedly routes an all-in stack toward the opposing general. Two games
per seat order were run per seed (400 tick cap):

| Seed | `final.py` as p0 | `final.py` as p1 |
|---:|---:|---:|
| 52000 | v8: 1/2 | not run |
| 53000 | v8: 1/2 | v8: 2/2; v9: 1/2 |
| 54000 | v8: 0/2; v9: 0/2 | v9: 2/2 |
| 56000 | v8: 0/2 | v8: 2/2; v9: 2/2 |

The v11 launch-intercept candidate matched v9 on the same rusher cases: as p1,
it scored 1/2 at seed 53000, 2/2 at 54000, and 2/2 at 56000. In a four-game
head-to-head at seed 59000 it split 2–2 with v9. It adds a tactically relevant
counteraction, but these small batches do not show an overall win-rate gain.

The v9 candidate split 2–2 against v7 over four games at seed 57000,
versus v8's 1–3 result in the same batch. The results show a strong seat effect
against the stress bot, so compare both seat orders when evaluating changes.

## ProbCut and queue experiments

`calibrate_probcut_charge.py` generates 80 randomized producer-wait positions,
uses the exact JAX transition/evaluation path to collect shallow/deep values,
and writes the fit beside the candidate. The v19 fit contains 105 paired rows:
slope 0.955, intercept 2.044, residual σ 2.768, correlation 0.974. This is a
small tactical calibration set, not the thousands of positions recommended by
Buro; the cutoff stays isolated to producer wait/launch nodes. Earlier fits are
preserved in `probcut_charge_fit_v*.json` and `archive/`.

The deeper candidates did not establish a win-rate improvement over v11 in the
seat-swapped batches so far. v14 split 1–1 at seeds 64000, 65000, and 66000;
v16 split 2–2 over the two seat orders at seed 67000 and 1–1 at seed 68000;
v17 split 1–1 at seed 68000. v18 lost both games at seed 69000. The rusher
results were comparable to v11: v14 scored 2/2 defending at seed 54000, v16
scored 2/2 at seed 56000, and v19 split 1–1 at seed 53000. Since those results
do not show a consistent gain, `final.py` remains v11; all candidates and fits
are kept here for further tests.

The long-search candidates averaged 1–5 ticks per second in these batches,
depending on whether a charge window was active. The viewer's queue can run
their safe prefixes; `v18_queued_prefetch.py` additionally computes the next
decision while those moves execute and validates the forecast before using it.

## Long-wait macro search and learned ProbCut

The v20 model was trained on 1,000 randomized 9×13 charge positions (2,042
rows). Its nonlinear residual model reduced held-out MAE from 2.018 to 1.787,
but explicit castle-first execution lost 7/8 games to v11 at seed 72100. The
separate v21 search-only candidate tied v11 4–4 in the same seed batch with
both seats. These shallow/deep fits do not search through a long producer wait.

The v23 search treats hold, castle capture, general charge, and expansion as
durative scripts, then simulates up to 96 exact simultaneous ticks per script
pair using the JAX rules. A separate MLP ProbCut model predicts the long-horizon
minimax row values; it uses a held-out residual bound, and it always searches
castle/general scripts and immediate general captures. v23's seat-balanced,
700-board 23×23 fit contains 1,899 nonterminal macro rows. Held-out MAE is
33.25 versus 36.50 for the linear baseline (8.9% lower). Raw training pairs
are retained at `archive/probcut_macro_pairs_v23_23x23_700.npz`; earlier fits
are also in `archive/`.

Across three paired seeds (74100, 74110, 74120; 12 games with seats swapped),
v23 split 6–6 with v11. It is not promoted because it has not shown a measured
win-rate gain. The final rusher sequence audit is recorded by
`evaluate_wait_sequence.py` when complete. `final.py` remains v11.

The offline trainer and macro simulator use JAX's default device, so they can
run on a GPU when the runtime provides a supported GPU-backed JAX build. This
workspace exposed only `cpu:0` (no CUDA or Apple MPS). JAX also documents that
macOS GPU execution is unsupported; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Cancelled round robin archive

We stopped the v1–v25 250-game schedule after 89 completed games. Partial
per-game rows remain in `tournaments/rr_v1_v25.jsonl` for audit, but no standings
or CSV are treated as results. The active work uses paired, targeted rusher
stress tests; do not resume the full round robin unless requested.

## Learned branch-pruning experiment

`train_learned_pruner.py` creates held-out-by-map training data from exact
one-ply simultaneous response matrices. The tiny CPU MLP predicts only whether
an optional bucket branch is dominated; it never selects the played move.
The inference candidate always keeps the A* principal action, pass/wait,
charge, defense/interception, and immediate producer/general captures. The
high-precision artifact is kept disabled because it pruned too few branches;
the experimental 0.90 cutoff is retained separately. The six-game paired
comparison against `general_rusher.py` was 5/6 for both learned and bucket-only
variants, while v27 was 3/6 on those same seeds and caps. More varied teacher
positions and deeper labels are needed before the classifier can be credited
with any strength gain. `final.py` remains v11.

## Heuristic and simultaneous-search alternatives

The time-dependent A* planner is a better path-feasibility model: it rejects
routes whose stack cannot survive hostile garrisons and their projected growth.
On 72 sampled states it reduced average node expansions from 31.6 to 22.6, but
the v33/v34 rusher result was 3/6, below v32's 5/6. The producer-payback score
also reached only 3/6 and expanded 143.7 nodes on average. The per-state audit is
in `tournaments/astar_heuristic_node_stats.jsonl`.

The scenario-rollout pair evaluates simultaneous hold, expand, castle, charge,
and tactical scripts with exact joint transitions, then replans after one tick;
it does not use a minimax recurrence. Extending the horizon from 112 to 224
ticks did not improve the first two paired seeds (both variants 1W/1D/2L).
Matches took roughly 15–65 seconds. A useful next algorithm comparison is
simultaneous-move MCTS with regret matching or Exp3 and macro-action widening.
`final.py` remains v11.

## Promotion: v37 is now `final.py` (2026-11-02)

`final.py` is a copy of `v37_phase_planner.py` (it re-exports v11's castle
helpers so `castle_general_rusher.py` and `castle_wait.py` are unchanged).
The previous final bot is preserved as `v11_launch_intercept.py`.

Evidence (seats swapped, 23×23):

| Suite | v36 | v37 |
|---|---|---|
| vs v11, seeds 93001–93010, cap 500 | 12/12 (6 seeds) | 20/20 (4 captures) |
| held-out rushers + expander, cap 500 | 55/60 | 58/60 |
| rushers + expander, 4 seeds, cap 2000 | 21W/3L | 22W/1D/1L |
| v37 vs v36 head-to-head, cap 500 | 8 | 12 |

Known weaknesses: expander seed 93009 seat 1 at cap 2000 captures the general
(both v36 and v37); v37's slowest move was 676 ms (mean 14.9 ms).

## Promotion: v42 is now `final.py` (2026-11-03)

`final.py` is a copy of `v42_collect_race.py` (same v11 re-exports). v42 loads
`v37_phase_planner.py` and `v39_counter_planner.py` from `models/`, so both
must stay. The previous final (v37) is unchanged in `v37_phase_planner.py`.

v42 = v37 + (1) a symmetric collection-race margin instead of the standalone
home margin (general moves keep the standalone check), (2) a castle layer
(half / all-but-one sends, checked on the board projected past the walk with
a buffer of 0.2 x the largest hostile stack), (3) strike race check on the
race margin.

| Suite | v37 | v42 |
|---|---|---|
| v42 vs v37, real harness, seeds 93001–93010, cap 1000 | — | 19W/1L (10-0, 9-1) |
| v42 vs v37, sim, seeds 93011–93040, cap 500 | — | 32W/28L |
| held-out suite cap 500 (v41-era castle rule) | 58/60 | not re-run |

Known weaknesses (human game 20261103_154954): spends its general on
castles/strike so the army sits diluted over land; a human general walking
out and collecting (84 -> 137) beats the single-stack defence; the race
estimate double-counts land tiles shared by home walks. Most wins at cap
1000 are timeout tiebreaks: it does not collect and strike (v43 work).

## Promotion: v46 is now `final.py` (2026-11-03)

Copy of `v46_opponent_learner.py`. Chain loaded by file from
`models/`: v45, v44, v42, v37, v39. Sim evidence: vs v42 14-6, vs v44 11-9,
vs v37 17-3 (cap 1000); scripted bots 81/100 (cap 500). Previous final (v42)
backed up outside the repo; source unchanged in `v42_collect_race.py`.

## Promotion: v51 is now `final.py` (2026-11-05)

Copy of `v51_fast_defence.py` (end of the coding part of the
project). Chain loaded by file from `models/`: v46, v45, v44, v42, v37, v39,
plus `v51_classifier.json` (opponent classifier, trained on the Mac GPU by
`train_v51_classifier.py`). v51 = v46 + an approach detector that commits to
defence and collects army INTO the general (PCST gather sized to the
attacker's ETA, adjacent stacks merge in) + a 10-type learned opponent
classifier (online posterior 66% / 70% accurate at ticks 200 / 300 on
held-out maps). Sim, cap 500, seeds 93001-93030: vs v46 32-28 (2 captures
each way). Fast defence (same as v52), cap 600, seeds 93001-93010:
GatherRusher t0=200 19-1 (v46 12-8), t0=120 20-0, t0=60 19-1; SpreadCollector
13-7 (v46 14-6); rushers 40/40. Previous final (v46) backed up outside the
repo; its source is unchanged in `v46_opponent_learner.py`.

## Versions after v40

| Model | Approach |
|---|---|
| `v41_castle_surplus.py` | v37 + a surplus-castle rule from recorded human games (send half of a safe general to a castle). Superseded by v42's castle layer. |
| `v42_collect_race.py` | Collection-race safety margin (what each side can gather onto our general in T moves) and a castle layer. Final bot 2026-11-03 (see above). |
| `v43_collect_strike.py` | v42 + collect onto the forward castle and strike, plus in-game parameter adaptation. vs v37 40W/20L (cap 1000, seeds 93011-93040). Not promoted. |
| `v44_reserve_counter.py` | v42 + each tile counted once, a general reserve, and a counter-strike when home loses the race. vs v42 13-7, vs v37 19-1. |
| `v45_gated_opponent.py` | v44 + opponent-gated castle and threat gates. vs v44 10-10, vs v37 16-4. |
| `v46_opponent_learner.py` | v45's gates driven by the in-game belief over four scripts, an income-race castle gate and an optional cross-game Exp3 bandit (off unless `V46_MEMORY` is set). Final bot 2026-11-03. |
| `v47_bayes_phases.py` | v46 + naive-Bayes opponent classifier over 8 types and army-spread phases. vs v46 8-12. Not promoted. |
| `v48_guard_split.py` | v46 + a home guard (defence vs attack stacks). vs v46 8-12; strong against gather rushers. Not promoted. |
| `v49_kill_window.py` | v46 + a kill window: attack as soon as a collected strike is guaranteed. vs v46 32-28 (cap 500, seeds 93001-93030). Not promoted. |
| `v50_26modified.py` | v46 with the cited papers' algorithms implemented as published (PCST gathering, MCF + MCKP, ABCD/Puppet search, Bayes + kNN, castle B&B, Nash arm), each switchable with `V50_OFF` / `V50_ON`; fits in `v50_model.json` (`fit_v50.py`). Ablations within noise of v46 except MCF (5-15, off by default). |
| `v51_fast_defence.py` | v46 + committed fast defence (approach detector, PCST gather into the general) + ten-style MLP opponent classifier (`v51_classifier.json`, `train_v51_classifier.py`). **Final bot** (`final.py`). |
| `v52_fast_defence_only.py` | v51 without the classifier (v46's opponent model). Used to measure the fast defence: GatherRusher t0=200 19-1 (v46 12-8), t0=120 20-0, t0=60 19-1. |
| `v53_verified_strike.py` | Experimental follow-up, not promoted: intercepts parked enemy stacks near our general with non-general stacks, and every attack (any source stack) must win against every reply (`verify()` on charges, a one-step check of all local enemy replies on contact moves). 18-2 against both besieger set-ups (`adversaries53.py`), 10-10 and 11-9 against v51. |
| `adversaries53.py` | Besieger stress opponent: collects one or two stacks and parks them a few squares from the enemy general, striking when an exact walk wins. Registered in `tools36/h2h42.py` as `siege` and `siege2`. |

