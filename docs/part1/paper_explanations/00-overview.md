# Search-only AI for perfect-information Generals.io

## The project

Generals.io is a real-time strategy game on a 23×23 grid. Both players expand
territory, grow armies, and try to capture the opponent's general. **We modified it by
removing the fog of war** — both players see the entire board.

That modification is not cosmetic. It changes which algorithms are *sound*:

| with fog | without fog (our variant) |
|---|---|
| belief state over hidden armies | the state **is** the board |
| minimax unsound — you cannot evaluate what you cannot see | minimax applies directly |
| cannot prove anything about the opponent | forced wins become provable |

**The only hidden information left is simultaneity**: both players commit a move for
the same tick without seeing the other's.

### The constraints we work under

- **0.5 s per move.** Real-time. An algorithm that returns nothing until it finishes
  is useless.
- **Branching factor 40–400 per side**, from an action space of `23 × 23 × 9 = 4761`.
- **Games last 300–1000 ticks.** Sparse terminal reward: win or lose.
- **Search and CSP only.** No reinforcement learning, no neural networks, no machine
  learning. AIMA chapters 3–6.

That last constraint is less limiting than it sounds. The strongest non-learning agent
on the public ladder, **Human.exe**, is built entirely from classical optimisation, and
it was the top bot until an RL agent trained on 4× H200 GPUs beat it. See
`../papers/human-exe-decomposed.md`.

## Where we are

We have a working JAX simulator of the modified game (fast enough to batch every
candidate move in one call), a pygame viewer, and a first bot.

**Bot 1 — `strategies/greedy_eval.py`.** One step of hill climbing (AIMA §4.1.1):
score every reachable board, play the best. **20% win rate** against a scripted
expander baseline. Diagnosed:

```
greedy's general captured   12/20     ← the actual problem
expander's general captured  3/20
timeout, greedy ahead        1/20
timeout, expander ahead      4/20
```

Land was even (27–27 at tick 300). It expands fine and gets killed, because it
simulates the opponent as passing and therefore **cannot see an attack coming**.

We also tried patching this with a "danger" term in the evaluation, and it made things
**worse** (20% → 15%). The reason is instructive and shapes the rest of the plan: that
term barely varied between candidate moves, and *a term that does not vary between
neighbours cannot change an argmax*.

## The five papers, and why each one

| # | paper | venue | AIMA | what it fixes for us |
|---|---|---|---|---|
| [01](01-ara-star.md) | ARA\* | NIPS 2003 | ch. 3 | must answer within 500 ms, always, with a quality bound |
| [02](02-rtaa-star.md) | RTAA\* | AAMAS 2006 | §4.5 | act every tick forever on a graph we cannot solve |
| [03](03-depression-avoidance.md) | Depression avoidance | AAAI 2012 | §4.5 | walking into positions that look good one move out — **our measured bug** |
| [04](04-probcut.md) | ProbCut | — | §5.3 | branching 400 makes sound pruning insufficient |
| [05](05-min-conflicts.md) | Min-conflicts | AAAI 1990 | §6.4 | which stack defends what, re-decided every tick |
| [06](06-metareasoning.md) | Metareasoning | SoCS 2015 / ICAPS 2017 / UAI 1990 | ch. 3–4 | **most ticks are obvious — stop wasting the budget on them** |

Paper 06 came from our own observation about the game and turned out to be an
established subfield. It is where we think our original contribution lies.

## The through-line

Four of the six papers are the same idea in different clothes — **do not throw away
what you already computed**:

- ARA\* carries `INCONS` across iterations instead of restarting the search
- RTAA\* carries the `h` table across ticks instead of re-estimating
- Min-conflicts repairs last tick's assignment instead of re-solving
- Metareasoning skips computation that cannot change the answer

For an agent that must decide 2× per second for 1000 ticks on a board that barely
changes between them, that is the correct organising principle.

**Next:** [`07-one-month-plan.md`](07-one-month-plan.md).
