# genmod — a JAX simulator for perfect-information Generals.io

A from-scratch, `jit`/`vmap`/`scan`-friendly reimplementation of the Generals.io
environment from **"Superhuman AI for Generals.io Using Self-Play Reinforcement
Learning"** (Straka, Lisý & Schmid, arXiv:2606.23348), with the paper's rules and
**one modification: the fog of war is removed.** Both players see the entire board.

The original fogged observation is still implemented, so you can train and
evaluate the same agent under both regimes with identical dynamics.

```bash
python demo.py --out replay.html        # play a game, write a scrubbable replay
python -m pytest tests -q               # 17 rule-level tests
python benchmark.py --envs 1024 4096    # throughput
```

The simulator requires `jax` and `numpy`. Nothing else.

**Want to play it?** See **[PLAYING.md](PLAYING.md)**:

```bash
python viewer.py --p0 expander --p1 my_agent.py:act    # native pygame window
python server/local_server.py --open                   # same thing in a browser
```

Plus a Vercel web app for human matches — share a link, or play a bot that runs
entirely in your tab.

---

## The modification

`EnvConfig.visibility` selects what a player observes. **The dynamics are
identical in all three modes** — only the observation tensor changes.

| mode | what a player sees |
|---|---|
| `"full"` *(default — the mod)* | Everything: all terrain, all armies, both generals, every castle, from turn 0. |
| `"opponent"` | The opponent's territory and armies are always exposed, but neutral terrain must still be explored. |
| `"fog"` | The original game: own cells plus their Moore (8-cell) neighbourhood. |

Under `"full"` the paper's fog bookkeeping collapses: `seen`, `enemy_seen`,
`structures_in_fog`, `last_enemy_army_seen` and `last_enemy_army_age` carry no
information, so they are replaced by exact current-frame channels. The
persistent-memory arrays are never written in this mode, and XLA eliminates them.

**Strategically this is a different game.** Deception, general-location inference
and fog-routed flank attacks all disappear; the paper's value-token attention
head that tracks a posterior over the enemy general (their Fig. 9) has nothing
left to do. What remains is a simultaneous-move, perfect-information RTS: tempo,
expansion-vs-defence tradeoffs, castle timing, and reading a *visible* opponent's
commitments one turn at a time. Both players still choose without seeing the
other's move for the current turn — that simultaneity is the only remaining
source of imperfect information, and it is what keeps the game non-trivial.

---

## Rules implemented

Taken from Section IV of the paper.

**Map.** `H × W` grid (default 23×23). 20% of cells are mountains. 9–11 neutral
castles, each garrisoned with 40–50 units. Generals are placed at least
`min_spawn_dist` (17) BFS steps apart, and every player is guaranteed a neutral
castle within BFS radius 6. Lowering `max_spawn_dist` gives you the paper's
spawn-distance curriculum (Section V-C) for free.

**Production.** Generals and owned castles gain +1 every 2 turns. Every owned
cell gains +1 every 50 turns. On a turn that is both, producers gain +2.

**Movement.** Each turn a player picks one owned cell and one cardinal direction,
and sends either *all but one* unit or *exactly half* (floor) — or passes. The
action space is `H × W × 9`, matching the paper's policy head.

**Combat.** Resolved by subtraction. The two stacks cancel one-for-one and the
larger side survives with the difference, taking the cell. **Ties leave the cell
with its previous owner.** A neutral garrison must be paid down in full before
the cell changes hands; draining it to exactly zero does *not* capture it.

**Winning.** Capture the opponent's general. Reward is sparse, terminal and
zero-sum: ±1.

**Free-for-all.** `num_players > 2` works: capturing a general transfers all of
that player's cells to the captor at half strength, and the fallen general
becomes a castle. 2v2 team play is *not* implemented.

### Simultaneous resolution

The paper specifies simultaneous moves but not the tie-breaking mechanics, so
this is the concrete rule used here — one pass, fully vectorised, order-independent:

1. **Departures.** Every legal mover's units leave their source cell first. A
   source always retains at least one unit and keeps its owner.
2. **Arrivals.** Each player's incoming stack is added to their *force* at the
   destination; a player already holding the destination folds their remaining
   garrison into the same force.
3. **Resolution.** At each cell the largest force survives with
   `largest − second-largest`. If that remainder is zero, the cell keeps its
   previous owner (with zero army). If the cell was neutral, the remainder is
   then absorbed by the neutral garrison; only a strict surplus captures it.

Two consequences worth knowing, both deliberate:

- **Head-on collision:** two stacks entering the same empty cell annihilate by
  difference, exactly like an attack.
- **Swap:** if two players move into each other's source cells, both armies pass
  through and each fights the reduced garrison left behind. The browser game
  resolves moves sequentially and would collide them; simultaneous resolution
  cannot, and picking a winner by player index would break the symmetry the
  self-play loop depends on.

---

## API

```python
import jax, genmod
from genmod import EnvConfig

cfg   = EnvConfig(height=23, width=23, visibility="full")
state = genmod.reset(jax.random.PRNGKey(0), cfg)

obs   = genmod.observe(state, cfg)             # (N, C, H, W) float32
mask  = genmod.legal_action_mask(state, cfg)   # (N, H, W, 9) bool
state, rewards, done, info = genmod.step(state, actions, cfg)   # actions: (N,) int32
```

`cfg` is a frozen dataclass passed as a **static** argument, so every branch on
it is resolved at trace time.

**Actions** are flat `int32` in `[0, H*W*9)`: `cell * 9 + k`, where `k = 0`
passes, `k = 1 + d` sends all-but-one in direction `d`, and `k = 5 + d` sends
half. Directions are `0=up, 1=down, 2=left, 3=right`. Illegal actions degrade to
a pass rather than erroring, so an unmasked policy is always safe.
`genmod.encode_action` / `decode_action` convert to and from `(row, col, dir, half)`.

**Batching.**

```python
state = genmod.reset_batch(key, cfg, num_envs=4096)
rollout = genmod.make_rollout(cfg, policy, num_steps=512)   # one XLA graph
state, traj = rollout(key, state)                           # auto-resets finished games
```

`make_rollout` fuses map generation, stepping, auto-reset and the policy into a
single compiled graph — that fusion is what the paper's throughput depends on.

### Observation channels

`C = 27 + 2 * history_length` (41 with the default `history_length=7`).
Everything is **perspective-normalised**: "own" always means the player the
observation belongs to, so one policy serves both seats. Army counts use a signed
`log1p` compression so unbounded stacks stay in range.

| # | channel |
|---|---|
| 0–2 | `own_army`, `enemy_army`, `neutral_army` |
| 3–5 | `own_cells`, `enemy_cells`, `neutral_cells` |
| 6 | `mountains` |
| 7–8 | `own_general`, `enemy_general` |
| 9–12 | `castles`, `own_castles`, `enemy_castles`, `neutral_castles` |
| 13–15 | `visible`, `seen`, `playable` |
| 16–17 | `coord_x`, `coord_y` |
| 18–20 | `timestep`, `phase_land_growth`, `phase_production` |
| 21–24 | `own_land_count`, `own_army_count`, `enemy_land_count`, `enemy_army_count` (broadcast) |
| 25–26 | `land_advantage`, `army_advantage` |
| 27–33 | `own_army_delta` over the last 7 turns |
| 34–40 | `enemy_army_delta` over the last 7 turns |

`genmod.channel_names(cfg)` returns the list. `EnvConfig.series_length > 0` adds
the paper's two non-spatial scoreboard tokens via `genmod.scoreboard_series`;
it defaults to off because a visible opponent makes the scoreboard history
largely redundant.

Under `"full"` visibility, channel 13 is all ones and channel 14 all zeros —
kept so a network trained in one mode has the same input shape in the other.

---

## Visualisation

`python demo.py` writes a **self-contained HTML replay** — no server, no CDN,
no dependencies. Scrub with the slider, play at 4–60 turns/second, step with the
arrow keys, and read land/army timelines beside the board. Arrows on the board
show the move each side committed *this* turn, with a white dot marking a
"send half"; both resolve simultaneously.

```python
from genmod.render import record, export_html, render_ansi
from genmod.bots import BOTS

rep = record(cfg, key, [BOTS["expander"], BOTS["random"]])
export_html(rep, "replay.html", subtitle="expander vs random")
print(render_ansi(state, cfg))     # coloured terminal board
```

`record` replays the whole game inside one `lax.scan` and returns numpy arrays
(`owner`, `army`, `castle`, `general`, `moves`, `land`, `armies`), so you can
feed it to any renderer you prefer. `export_html(..., standalone=False)` emits a
body fragment for embedding.

## Bots

`genmod.bots` ships three pure `(key, state, cfg, player) -> action` policies,
all `jit`/`vmap`-safe, so they double as scripted opponents inside training:

- `pass_bot` — always passes.
- `random_bot` — uniform over *legal moves*, passing only when none exist.
- `expander_bot` — takes any adjacent cell it can afford, otherwise marches its
  largest stack down a BFS field toward the nearest unowned cell. It beats
  `random_bot` essentially always and makes for a non-trivial curriculum floor.

## Layout

```
genmod/
  config.py                 EnvConfig — every knob, all static
  state.py                  State PyTree + scoreboard helpers
  scenario.py               build a State from an explicit board, and back
  mapgen.py                 vectorised map generation, grid BFS
  env.py                    reset / step / combat / legal masks / visibility
  obs.py                    observation tensors, channel names
  bots.py                   scripted baseline policies
  rollout.py                batched reset / step / scan rollouts
  render.py                 recording, ANSI board, HTML replay export
  _replay_template.html     the replay viewer
demo.py  benchmark.py  viewer.py  tests/test_rules.py

server/                     local play + the genmod-bot/1 protocol (JAX engine)
web/                        Next.js app for Vercel (TypeScript mirror engine)
shared/board.mjs            canvas renderer used by both front ends
tools/export_golden.py      generates the cross-implementation test vectors
```

The web app runs a TypeScript mirror of `env.py`, because Vercel cannot hold a
JAX process open per match. `web/test/golden.test.ts` replays JAX-generated
games through it and asserts the boards match tick by tick — **change one engine
and you must change the other.** See [PLAYING.md](PLAYING.md).

## Notes on the port

- **Nothing is hidden behind Python control flow.** `EnvConfig` is static, so
  `visibility`, `history_length`, `num_players` and `timeout_result` all branch at
  trace time and cost nothing at runtime.
- **Map generation is branch-free.** Positions are sampled with Gumbel-argmax
  (and Gumbel top-k for the castles) instead of rejection sampling, so there is no
  `while_loop` whose iteration count varies across a batch. BFS is a fixed number
  of min-plus relaxation rounds (`bfs_iters`); raise it if you use a much larger
  grid or a much higher mountain density.
- **Variable map sizes** use a fixed buffer with an active sub-rectangle
  (`random_dims=True`); the padding is mountains, exactly as in the paper.
- **`max_steps` decides a stalled game** by land, then army, then a draw
  (`timeout_result="draw"` gives you the paper's pure win/loss signal instead).
  Real Generals.io has no turn limit; RL needs one.
