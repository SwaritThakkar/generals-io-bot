# v51: from deeper search to collection-aware defence

## Purpose and scope

This writeup is the research companion to the revised talk. It separates what the cited papers establish from what this project adapted, then explains the game model, the v51 design, and the evidence behind its main change. The presentation concerns the modified, fully visible Generals environment implemented in this repository. It does not claim that v51 is stronger than every later bot in the repository.

## The game model that shaped the research

The board is a 23 × 23 grid. Each side chooses one move at the same tick; the two moves resolve together. A moving stack leaves one soldier behind on every tile it exits, combat subtracts armies, and generals and castles produce one soldier every two ticks. Owned land produces one soldier every fifty ticks. Capturing the opposing general wins immediately; otherwise the end condition uses the game’s land-based result.

The asymmetry between walking and waiting matters more than a generic positional score suggests. A stack of 30 that travels eight squares arrives with 22 if it has no fights en route. During those eight ticks, a stationary general gains four. Thus “attack the biggest stack” is a poor proxy for “can this side win the general race?” The relevant threat is the amount of army that can be assembled at a target before impact. A player can wait on productive ground, collect several stacks, and then send them along a shortest route. A defensive rule that considers only the largest enemy stack misses that sequence; a rule that reacts to a temporary score recovery can also abandon defence too early.

## Why the first search direction did not solve the loss

The initial search programme explored ARA*, RTAA*, depression avoidance, ProbCut, and metareasoning. These are useful techniques for specific bottlenecks: anytime path search, online heuristic learning, recovery from heuristic depressions, selective deepening, and allocating scarce computation. In this project’s visible grid, shortest distances are already cheap and exact, and the failed games were not caused by running out of compute. The measured shallow/deep value correlation was high, while match results did not improve from the tested search variants. The failure was in the state abstraction and threat model: the bot treated army at home as safe and distant army as irrelevant until the attack was nearly at the general.

ProbCut, introduced by Buro for Othello alpha-beta search, estimates a statistical relationship between shallow and deeper search values and prunes a deep search when a confidence bound indicates that it cannot affect the alpha-beta result. It is not a general-purpose “search more moves” method. The project tested analogous shallow/deep filtering, but that did not address delayed army collection, and those variants were not promoted. The same distinction applies to ARA*: Likhachev, Gordon, and Thrun improve a heuristic path solution as time permits; in this game, distance is not the hard unknown. RTAA* (Koenig & Likhachev) updates a heuristic while acting, while depression avoidance (Hernández & Baier) addresses misleading heuristic basins. Neither changes the need to reason about future army arrival. Metareasoning by Russell and Wefald asks whether additional computation is worth its cost; the measured decision cost here was small relative to the tick budget, and more computation did not fix the model error.

This is a problem-specific conclusion, not a rejection of those methods. A different game with hidden terrain, expensive pathfinding, or a useful deep value signal could make them effective.

## The sequence of design changes

### v36: single-stack safety

The first successful repair checked whether an enemy stack could beat the general if it walked in. Candidate moves that emptied the general were rejected whenever the remaining army plus growth could not beat that single arrival. In the repository’s evaluation, v11 lost its general to simple rushers in 25 of 60 games; v36 won 24 of those 25 same-map rematches. This repaired the obvious “walk the whole general away” failure.

The model was still incomplete. It considered threats one at a time. A patient opponent could own more land and castles, keep collecting, and eventually combine its army into one large attack. The largest stack at the current tick did not describe the army it could deliver later.

### v42: collection-aware threat estimation

The next change estimated how much army either side could deliver within a move budget. For each future horizon, the bot asks what the defender can keep or bring home and what the attacker can collect and send. It then uses the worst projected safety margin rather than the best-looking current snapshot. This design was motivated by a recorded human win where a single-stack check read “safe” while the opponent held substantially more production and eventually assembled a larger force.

Exact selection of routes and stacks is combinatorial. v42 used a greedy approximation; v51 also uses greedy collection estimates. The estimate is useful as a trigger and safety signal, but it is not an exact solution to an NP-hard budgeted collection problem. Route overlap, tile ownership, congestion, and simultaneous moves can all make projected delivery differ from realized delivery.

### v51: detect commitment and stay in defence

The remaining exploit was temporal. v46 could recognize danger on one tick, defend, see its margin recover, and resume expansion while the same committed attack was still approaching. v51 adds an incoming-stack detector and latches a defensive mode. A large stack is considered incoming when it has made sufficient progress toward the general during a short observation window. The bot estimates the shortfall at arrival; while the shortfall is positive, every move is assigned to collecting defence.

Its gathering planner adapts the prize-collecting Steiner tree structure described for Human.exe by Straka, Lisý, and Schmid. In the paper’s Generals agent, army on vertices is a prize and movement is a cost; the bot seeks a connected set of routes that gathers army within a move budget. v51 roots its gathering tree at the home general and uses a small, greedy tree-building procedure with a binary-searched price per move. It orders moves from outer branches inward so that armies can accumulate along routes. This is an adaptation: it is not the paper’s full implementation or its learned/self-play system, and the project’s use is specifically to defend against a predicted arrival.

## Research connections and limits of transfer

### Opponent inference: Bayesian plan recognition and classifiers

Synnaeve and Bessière model StarCraft opening prediction as Bayesian inference over strategies, updating a belief as observations arrive. Weber and Mateas use replay-derived game features to predict strategy. v51 combines these types of ideas: a neural classifier maps fourteen board-history features to ten scripted styles; an online belief update smooths those predictions across ticks. The evidence is the fully visible board and recent deltas, not hidden state. The mapping is project-specific, the labels are generated by scripted opponents, and a patient rusher can look like a land-grabber before launch. The classifier reaches about 70% held-out style accuracy by ticks 200–300, but this did not yield a measurable improvement in the tested head-to-head matches: v51 and classifier-free v52 both went 32–28 against v46. That is a useful negative result. Better prediction is not automatically better control.

### Macro-plan search: Puppet Search

Puppet Search (Barriga, Stanescu, and Buro) uses scripted behaviours with choice points and searches over those choices, allowing real-time planning without expanding every low-level RTS action. v51 similarly compares a small menu of plans—expand, hold, take a nearby castle, gather, strike, or rush—by simulating them against a handful of opponent scripts. It scores the resulting states with a learned win estimator and combines belief-weighted expected score with a worst-case score. This is inspired by macro-action search, but should not be described as a faithful Puppet Search implementation: the project’s plan generator, opponent scripts, simulator, and aggregation rule are its own.

### Safe opponent exploitation

Ganzfried and Sandholm study safe exploitation in repeated zero-sum games. Their formal guarantees depend on a game-theoretic value and a constrained deviation from a safe strategy. v51 uses the idea as a design principle: opponent belief may change how greedily the bot values castles or how much reserve it keeps, but it does not disable the general-safety gate. That is not the paper’s algorithm or a theorem-backed safety guarantee. The actual safety property depends on the quality of v51’s finite-horizon threat estimate and its reply checks.

### Economy and investment timing

Churchill and Buro’s StarCraft build-order optimization schedules production and construction under resource constraints. v51 borrows the future-income comparison: castles cost army and travel now but produce over the remaining game, so the bot estimates payback and checks the post-move home defence. The game mechanics and decision variables differ; v51 does not implement the StarCraft build-order optimizer.

### Self-play Generals reference

Straka et al. describe Human.exe alongside a self-play reinforcement-learning agent for Generals.io. The project uses the paper as a source for the hand-written agent’s collection formulation. v51 does not reproduce the paper’s self-play system, training setup, or performance claims. The transfer is narrow: model the value of armies that can be routed into a target before a deadline.

## How v51 chooses

Every tick, v51 reads the board, updates opponent-style probabilities, checks for a capture, checks for an incoming committed attack, and either defends or evaluates plans. Proposed moves pass a safety check. A fast defence is sticky: it is not cancelled merely because a single margin reading recovers while the attacker is still approaching. Other actions—castles, expansions, or attacks—are considered only when the relevant safety and timing checks pass.

In ordinary play the planner simulates a small menu of macro plans for a finite horizon against several simple opponent scripts. A learned value model converts the resulting board into a win estimate. The belief over opponent styles changes plan weights; a worst-case component prevents the average against the current belief from dominating all other outcomes. This is a practical compromise, not full simultaneous-move minimax. The opponent is represented by a few deterministic scripts, and the rollout horizon cannot see every later response.

The style classifier uses a 14–64–64–10 feed-forward network: fourteen normalized features, two 64-unit hidden layers, and ten style outputs. Training data consist of v46 games against ten scripted styles over 33 maps, sampled every other tick; the split is by map, with 25 maps for training and eight held out. The reported counts are 99,437 training rows and 33,318 held-out rows. Accuracy is low early when opponent styles look alike, rising to 70% at ticks 200–300. A separate early-rush classifier reached 92%. These are classification measures, not win rates.

## Evidence and limits

In one traced seed (93003), a gather-rusher waited until tick 200, emptied its general at tick 213, and sent a stack down a shortest route. v51 recognized the approach at tick 219, gathered its army, and survived the impact at tick 237 with 38 soldiers remaining. v46 lost the same position at tick 237, 77 against 82. v51 later took a castle and captured the opponent at tick 407. This replay demonstrates the mechanism; it is not a statistical estimate by itself.

Across the recorded match suite, v51’s fast defence substantially improved results against the delayed gather-rusher. The 600-tick variant won 19 of 20 games against the rusher that began gathering at tick 200, compared with 12 of 20 for v46; against the tick-120 gather-rusher, the counts were 20/20 and 18/20; against the tick-60 gather-rusher, 19/20 and 16/20. It went 40/40 against the two direct rushers tested. Against a spread-and-collect opponent, it scored 13/20 versus v46’s 14/20, exposing the opportunity cost of staying home while the opponent takes production. In 60 games versus v46, v51 scored 32–28; classifier-free v52 produced the same record. Therefore the measured upgrade is specific: committed defence counters the gather-and-charge sequence. The opponent classifier was not shown to improve overall match results.

The main limitations are model error and strategic opportunity cost. Enemy collection is approximated greedily, opponent scripts cover only a few behaviours, and defence can concede land or castles to a player who spreads out. The attack checker can be conservative, so v51 may wait too long against a careful player. The replay and match suite support a targeted improvement, not a claim of universal dominance.

## References

- Barriga, N., Stanescu, M., & Buro, M. “Puppet Search: Enhancing Scripted Behavior by Look-Ahead Search with Applications to RTS Games.” AIIDE, 2015. [AAAI proceedings](https://ojs.aaai.org/index.php/AIIDE/article/view/12779).
- Buro, M. “ProbCut: An Effective Selective Extension of the Alpha-Beta Algorithm.” ICCA Journal, 1995. [Paper](https://journals.sagepub.com/doi/pdf/10.3233/ICG-1995-18202).
- Churchill, D., & Buro, M. “Build Order Optimization in StarCraft.” AIIDE, 2011. [AAAI proceedings](https://ojs.aaai.org/index.php/AIIDE/article/view/12435).
- Ganzfried, S., & Sandholm, T. “Safe Opponent Exploitation.” ACM Transactions on Economics and Computation, 2015. [Author-hosted paper](https://www.cs.cmu.edu/~sandholm/safeExploitation.ec12.pdf).
- Hernández, C., Baier, J. A., Uras, T., & Koenig, S. “Avoiding and Escaping Depressions in Real-Time Heuristic Search.” *Journal of Artificial Intelligence Research*, 43, 523–570, 2012. [Paper](https://doi.org/10.1613/jair.3590).
- Koenig, S., & Likhachev, M. “Real-Time Adaptive A*.” AAMAS, 2006.
- Likhachev, M., Gordon, G. J., & Thrun, S. “ARA*: Anytime A* with Provable Bounds on Sub-Optimality.” NeurIPS, 2003. [Proceedings](https://papers.nips.cc/paper_files/paper/2003/hash/ee8fe9093fbbb687bef15a38facc44d2-Abstract.html).
- Russell, S., & Wefald, E. “Principles of Metareasoning.” Artificial Intelligence, 1991.
- Straka, M., Lisý, V., & Schmid, M. “Superhuman AI for Generals.io Using Self-Play Reinforcement Learning.” arXiv:2606.23348, 2026. [Paper](https://arxiv.org/abs/2606.23348).
- Synnaeve, G., & Bessière, P. “A Bayesian Model for Opening Prediction in RTS Games with Application to StarCraft.” AIIDE, 2011. [Paper](https://arxiv.org/abs/1111.3735).
- Weber, B., & Mateas, M. “A Data Mining Approach to Strategy Prediction.” CIG, 2009. [Paper](https://alumni.soe.ucsc.edu/~bweber/pubs/cig_2009.pdf).
