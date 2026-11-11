# strategies/

Your agents live here. One file per strategy, each exporting `act(state, player)`.

```bash
# watch it play
python viewer.py --p0 human --p1 strategies/yourbot.py
python viewer.py --p0 strategies/a.py --p1 strategies/b.py --tick 100

# score it
python server/local_server.py --p0 strategies/yourbot.py --p1 expander --games 100 --headless
```

## The contract

```python
def act(state, player):
    """state: BotState (see server/bot_protocol.py). Return an action.

    state.owner / .army / .mountain / .castle / .general  -> (h, w) numpy arrays
    state.general_idx / .alive / .land / .army_total      -> (n,) arrays
    state.legal_actions(player)                           -> list[int]
    state.to_genmod_state(cfg)                            -> a genmod.State
    """
    return 0  # pass
```

Return a flat action index, a `(row, col, dir, half)` tuple, or a list of those to
queue a path. Directions are `0=up, 1=down, 2=left, 3=right`.

Nothing is hidden — this is the perfect-information variant, so `state` is the
complete board for both players.

## Gotcha

Index 0 of every cell's 9 actions is **pass**, and it is legal on every cell. A
flat `argmax` over a masked score will therefore pass forever unless you exclude it.

**Start with [RESEARCH-DIRECTIONS.md](RESEARCH-DIRECTIONS.md)** — eight directions,
each grounded in a specific AIMA chapter, with the maths and full citations.
PDFs are in [`../papers/`](../papers/); the annotated index is
[`../papers/README.md`](../papers/README.md).
