# Making a Bot for Generals.io using Search Algorithms and Offline Learning

**DAC-203 – Artificial Intelligence, IIT Roorkee**

A game-playing agent for a two-player, simultaneous-move, fully observable version of
[Generals.io](https://generals.io) on a 23×23 board. The final bot, **v51** (`final.py`), is built
from methods of the course textbook (AND-OR search, depth-limited look-ahead with a learned
evaluation function, Bayesian filtering, expected utility and decision making under uncertainty,
logistic regression and a small neural network) and from four papers on real-time strategy games
(Synnaeve & Bessière 2011, Barriga et al. 2015, Straka et al. 2026, Churchill & Buro 2011). It never
plays a move after which its general can be taken, chooses among a few scripted plans by simulating
them against four imagined opponents, commits to a Steiner-tree defence when an attack is coming,
and attacks only when the attack is proven. The project went through about fifty versions, all kept
in `models/`.

| Document | Link |
|---|---|
| Project report (PDF) | [`project_report/report.pdf`](project_report/report.pdf) |
| Project report (Markdown) | [`project_report/report.md`](project_report/report.md) |
| Final presentation | [`presentation/v51-talk.pdf`](presentation/v51-talk.pdf) (slides with speaker notes: [`presentation/v51-talk.pptx`](presentation/v51-talk.pptx)) |

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Setup](#2-setup)
3. [Play the game against the bot](#3-play-the-game-against-the-bot)
4. [Use the bot from code](#4-use-the-bot-from-code)
5. [Reproduce the results](#5-reproduce-the-results)
6. [How the versions are organised](#6-how-the-versions-are-organised)
7. [Results summary](#7-results-summary)
8. [Limitations](#8-limitations)
9. [Credits and references](#9-credits-and-references)

---

## 1. Repository layout

```
final.py            the final bot (v51); entry point act(state, player)
play_arena.py       launches the pygame arena: play any bot, tutorial, leaderboard
viewer.py           older pygame viewer on the JAX engine (watch / play / rewind)
arena/              the arena app: screens, bot registry, cached maps, tutorial videos, results store
arena_results/      human-vs-bot games recorded with the arena (games.jsonl + replays)
genmod/             the game engine in JAX (reference rules, map generator, renderer)
models/             every bot version v1–v53, shared cores (sim36, core36, core37), training scripts, fitted models
models/tools36/     evaluation harness: head-to-head runner, simulator check, game analysis
models/tournaments/ recorded tournaments and the human games used for analysis
server/             local match server and the bot protocol (BotState) used by the arena
strategies/         early search agents (weighted A*, greedy) used by the first versions
tests/              pytest suite for the game rules
project_report/     project report: LaTeX source, figures, PDF, Markdown
presentation/       final presentation (pptx/pdf, writeups, build scripts); earlier/ holds the first two decks
docs/               how-to-play notes, arena guide, research ledger, v46 explainer, project notes
papers/             reference list with links (arXiv papers included)
legacy/             Part-I figure tools, early demos, and the optional Next.js web app
```

## 2. Setup

Tested with **Python 3.12** on macOS (Apple Silicon).

```bash
pip install -r requirements.txt
```

- `jax` runs on the CPU; no GPU is needed to play or to evaluate.
- `torch` is only needed to retrain the opponent classifier (`models/train_v51_classifier.py`),
  which asserts that the Apple GPU backend (MPS) is available.
- Run every command below from the repository root.

## 3. Play the game against the bot

```bash
python play_arena.py                    # 1280x800 window (resizable)
python play_arena.py --window 1440x900
```

1. Type your name.
2. Pick **Final · champion (v51 fast defence)** from the bot list (it is listed first; every
   other version is also playable).
3. Choose the side for game 1, the length (300 / 600 / 1000 ticks) and the speed, then press Enter.
   Every match is a pair: game 2 is played on the same map with the seats swapped.
4. First time? Open **How the game works** (F1) and **Controls** (F2) from the home screen.

**Controls in game:** click one of your tiles, arrow keys / WASD queue moves, Shift + arrow sends
half the stack, `c` clears the queue, `p` pauses, `[` / `]` change speed, Esc leaves (counts as a
quit). Results are written to `arena_results/`; the leaderboard (F3) ranks bots and players. More
detail: [`docs/ARENA.md`](docs/ARENA.md) and [`docs/PLAYING.md`](docs/PLAYING.md).

A headless check that drives every screen and saves screenshots (it does not touch real results):

```bash
SDL_VIDEODRIVER=dummy python play_arena.py --smoke --shots /tmp/arena_shots   # prints "smoke ok" (~13 s)
```

## 4. Use the bot from code

`final.act(state, player) -> int` takes the full board for one tick (a `server.bot_protocol.BotState`:
`owner`, `army`, `mountain`, `castle`, `general` grids, `general_idx`, `tick`, …) and returns one
action: `0` = pass, otherwise `cell * 9 + k` where `k = 1..4` moves the stack up / down / left / right
leaving one soldier behind and `k = 5..8` sends half. Minimal example (run from the repo root):

```python
import sys
sys.path[:0] = [".", "models", "models/tools36"]
import final
from arena.bots import bot_state
from arena36 import start                  # a harness map as a simulator state

ctx, sim = start(93001, 600)
for tick in range(5):
    a0 = final.act(bot_state(sim), 0)      # final bot plays seat 0
    a1 = final.act(bot_state(sim), 1)      # ... and seat 1
    sim.step2(a0, a1)
```

The bot keeps its per-game memory (plan, beliefs) between calls and starts fresh when the tick goes
backwards (a new game).

## 5. Reproduce the results

All experiments run in `models/sim36.py`, an exact pure-Python copy of the JAX rules, on the
harness maps (`genmod.reset` with a seed). `models/tools36/h2h42.py CAND OPP...` plays every seed
in `--seeds A B` (inclusive) from both seats, in parallel worker processes, and prints wins/draws/losses,
the per-seat split, castles held at the end and the candidate's mean / max time per move. Opponent
names: `gather60`, `gather120`, `gather200` (collect, then charge at that tick), `spread`,
`spread_late`, `expand`, `rusher`, `castle_rusher`, `econ`, `siege`, `siege2`, or any bot file.
Timings below are wall-clock on our laptop (Apple Silicon, 12 worker processes).

### 5.1 Simulator check and tests

```bash
python models/tools36/check_sim36.py      # "ok: 40 games, 10400 joint transitions identical" (~2 s)
python -m pytest -q tests                 # 17 passed (~7 s)
```

### 5.2 Fast defence against collect-and-charge bots

The defence table was measured on **v52** (`models/v52_fast_defence_only.py`), which is
v51 with the same fast defence but without the opponent classifier.

```bash
python models/tools36/h2h42.py models/v52_fast_defence_only.py gather200 --cap 600   # 19-1  (~12 s)
python models/tools36/h2h42.py models/v46_opponent_learner.py  gather200 --cap 600   # 12-8  (~9 s)
```

Both reproduce the report's numbers exactly (95% against 60%, 20 games each, seeds 93001–93010). The full table:

```bash
python models/tools36/h2h42.py models/v52_fast_defence_only.py \
    gather200 gather120 gather60 spread spread_late rusher castle_rusher --cap 600
```

We ran this table on two seeds (`--seeds 93001 93002`, 4 games per opponent, 52 s); the full
10-seed run takes a few minutes.

### 5.3 v51 against the previous final bot v46

```bash
python models/tools36/h2h42.py models/v51_fast_defence.py models/v46_opponent_learner.py \
    --seeds 93001 93030 --cap 500                                                    # ~64 s
```

The report gives 32–28, and re-running this command reproduces it; one other run gave 33–27, because a single game can flip between runs (for example through Python hash ordering).
`final.py` is a copy of v51 and plays identically (checked on seed 93001).

### 5.4 Retrain the learned win estimate (logistic regression)

```bash
python models/train_value37.py            # ~3 min on one CPU core; overwrites models/value37.json
```

Plays 200 self-play games between seven scripted policies (train seeds 95000–95159, held-out
95160–95199, caps 300/500/800), records features every 5 ticks and fits one logistic model per phase
by Newton's method. **Note:** the scripted policies in `core36` / `core37` were improved after v37 was
trained, so a re-run today plays slightly different games (28,140 / 7,084 rows instead of
27,208 / 6,986) and gives different weights; the qualitative result is the same (opening better than
land alone, mid-game not better, end game ≈ 96–98% correct). Keep a copy of the shipped
`models/value37.json` if you want the exact weights the final bot uses.

### 5.5 Retrain the opponent classifier (Apple GPU)

```bash
python models/train_v51_classifier.py --help
python models/train_v51_classifier.py --train 93101 93125 --test 93126 93133    # overwrites models/v51_classifier.json
```

v46 plays each of the 10 opponent styles on 33 maps from both seats (cap 500) on CPU worker
processes, then a 14-64-64-10 MLP is trained with Adam for 60 epochs on MPS. Expected (as shipped):
99,437 training rows, 33,318 test rows; per-board accuracy 24.6% / 53.3% / 69.9% / 67.2% for ticks
0–100 / 100–200 / 200–300 / 300–500; filtered belief 38.8% / 66.2% / 70.2% at ticks 100 / 200 / 300.
(We checked `--help`; the full run was not repeated for this README.)

## 6. How the versions are organised

- Every version is a single file `models/vNN_name.py` exposing `act(state, player)` (v36+ also
  expose `Planner`, used by the simulator harness). Later versions load their predecessors by file
  name, so `models/` must stay flat.
  v36+ share `sim36.py` (simulator), `core36.py` (safety layer, scripts) and `core37.py` (plans,
  phases, value model).
- [`models/README.md`](models/README.md) has one line per version with its idea and measured result,
  plus a section for each promotion to `final.py` (v37 → v42 → v46 → v51).
- [`docs/RESEARCH_IDEAS.md`](docs/RESEARCH_IDEAS.md) is the research ledger: every idea, the paper it
  came from, and what happened when we tested it.
- The project report, Section 4 ("Trials and Error"), tells the story of the three changes of direction.

## 7. Results summary

| Experiment (simulator) | Result |
|---|---|
| v11 (search-era) against two rush bots, 60 games | lost its general in 25 |
| v36 (safety rule) on the 25 maps v11 lost | won 24 |
| v36 against the direct rusher | 20 / 20 |
| Fast defence against the bot that collects and charges at tick 200 | 95% (v46: 60%) |
| … charges at tick 120 / tick 60 | 100% / 95% (v46: 90% / 80%) |
| … against a bot that spreads out and takes castles | 65% (v46: 70%) |
| … against rushers | 40 / 40 |
| v51 against v46 (previous final), 60 games | 32–28 (±1 game between runs) |
| Learned win estimate, held-out accuracy | opening 75% (land alone 71%), mid-game no better than land, end game 98% |
| Opponent classifier, filtered belief on unseen maps | 39% / 66% / 70% at ticks 100 / 200 / 300 (chance 10%) |
| Decision time | about 10 ms per move on average |

### Against human players (v51, the final bot)

Games played by classmates against v51 in the arena (`play_arena.py`), recorded automatically in
[`arena_results/games.jsonl`](arena_results/games.jsonl) with replays in `arena_results/replays/`.
Board 23×23, one tick every 350–500 ms, humans played both seats.

| | Games | v51 won | Humans won |
|---|---|---|---|
| Completed games | 14 | **11** (10 by capturing the general, 1 on land at the tick limit) | 3 (all by capture) |
| Player A | 10 | 7 | 3 |
| Player B | 2 | 2 | 0 |
| Player C | 2 | 2 | 0 |

- **v51 won 11 of 14 completed games (79%)** against three human players; it captured the human
  general between ticks 268 and 655.
- All three human wins came from one player (Player A, 3 of their 10 games), at ticks 382–638.
- Two further games ended when the human quit (at ticks 8 and 355); they are not counted above.
- Count them yourself:
  `python -c "import json; g=[json.loads(l) for l in open('arena_results/games.jsonl')]; v=[d for d in g if d['bot_id'].endswith('v51_fast_defence') and d['end']!='quit']; print(len(v), sum(d['winner']=='bot' for d in v))"`

## 8. Limitations

- **Late attacker:** it attacks only proven attacks, which in practice means an emptied enemy general.
- **Weak against spreading and castles:** defence has priority over castles.
- **Hand-set constants:** pad, reserve fractions, thresholds and horizons were set from lost games,
  not tuned systematically.
- **Narrow opponent model:** four scripts and ten styles.
- **Full board:** the bot reads the whole board; the original game's fog of war would need an extra
  inference layer.
- A few ticks that verify an attack take much longer than average (the worst we measured was
  about 1.7 s); in the arena the board keeps ticking while the bot thinks.

## 9. Credits and references

DAC-203 course project, IIT Roorkee.

- G. Synnaeve and P. Bessière. A Bayesian model for plan recognition in RTS games applied to StarCraft. AIIDE 2011.
- B. G. Weber and M. Mateas. A data mining approach to strategy prediction. IEEE CIG 2009.
- N. A. Barriga, M. Stanescu and M. Buro. Puppet Search: Enhancing scripted behavior by look-ahead search with applications to real-time strategy games. AIIDE 2015.
- M. Straka, V. Lisý and M. Schmid. Superhuman AI for Generals.io using self-play reinforcement learning. arXiv:2606.23348, 2026.
- D. Churchill and M. Buro. Build order optimization in StarCraft. AIIDE 2011.
- S. Russell and P. Norvig. Artificial Intelligence: A Modern Approach, 4th ed. Pearson, 2020.

The five search papers of the first two presentations (ARA\*, RTAA\*, depression avoidance, ProbCut,
metareasoning) and further reading are listed with links in [`papers/README.md`](papers/README.md).
