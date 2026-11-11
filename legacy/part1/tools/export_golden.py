#!/usr/bin/env python3
"""Generate golden vectors so the TypeScript engine can be proven equivalent to JAX.

Each case pins an explicit starting board plus a fixed action sequence, and
records the full board after every tick. `web/test/golden.test.ts` replays them
and asserts equality cell by cell.

Map *generation* is deliberately not covered here: the two implementations draw
from the same distribution but not the same PRNG stream. Those constraints are
checked separately in `web/test/mapgen.test.ts`.

    python tools/export_golden.py            # writes web/test/golden.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from genmod import EnvConfig, reset, step                    # noqa: E402
from genmod.config import NUM_CELL_ACTIONS                   # noqa: E402
from genmod.scenario import make_state, state_to_dict        # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "web" / "test" / "golden.json"

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def cfg_to_json(cfg: EnvConfig) -> dict:
    return {
        "height": cfg.height,
        "width": cfg.width,
        "numPlayers": cfg.num_players,
        "mountainDensity": cfg.mountain_density,
        "minCastles": cfg.min_castles,
        "maxCastles": cfg.max_castles,
        "castleArmyMin": cfg.castle_army_min,
        "castleArmyMax": cfg.castle_army_max,
        "minSpawnDist": cfg.min_spawn_dist,
        "maxSpawnDist": min(cfg.max_spawn_dist, 10**9),
        "guaranteedCastleRadius": cfg.guaranteed_castle_radius,
        "generalGrowthInterval": cfg.general_growth_interval,
        "landGrowthInterval": cfg.land_growth_interval,
        "maxTicks": cfg.max_steps,
        "tickMs": 500,
    }


def snapshot(state, cfg) -> dict:
    d = state_to_dict(state, cfg)
    return {k: d[k] for k in ("tick", "done", "winner", "owner", "army", "castle",
                              "general", "generalIdx", "alive")}


def roll(cfg: EnvConfig, state, actions: list[list[int]]) -> dict:
    """Apply a fixed action sequence, recording the board after every tick."""
    initial = state_to_dict(state, cfg)
    ticks = []
    for a in actions:
        state, _r, _d, _i = step(state, jnp.asarray(a, jnp.int32), cfg)
        ticks.append(snapshot(state, cfg))
    return {"config": cfg_to_json(cfg), "initial": initial, "actions": actions, "ticks": ticks}


def act(cfg, r, c, direction, half=False) -> int:
    return (r * cfg.width + c) * NUM_CELL_ACTIONS + 1 + direction + (4 if half else 0)


# --------------------------------------------------------------------------- #
def scenario_cases() -> list[dict]:
    """Hand-built positions covering the rules that are easy to get subtly wrong."""
    cfg = EnvConfig(height=5, width=5, num_players=2, max_steps=100,
                    general_growth_interval=2, land_growth_interval=50)
    h, w = cfg.height, cfg.width
    cases = []

    def base():
        owner = np.full((h, w), -1, np.int8)
        army = np.zeros((h, w), np.int32)
        gen = np.zeros((h, w), bool)
        owner[0, 0], army[0, 0], gen[0, 0] = 0, 1, True
        owner[4, 4], army[4, 4], gen[4, 4] = 1, 1, True
        return owner, army, gen

    def add(name, mutate, actions, c=cfg):
        owner, army, gen = base()
        castle = np.zeros((h, w), bool)
        mutate(owner, army, castle)
        st = make_state(c, owner, army, castle=castle, general=gen)
        case = roll(c, st, actions)
        case["name"] = name
        cases.append(case)

    add("move_all_but_one",
        lambda o, a, k: (o.__setitem__((2, 2), 0), a.__setitem__((2, 2), 10)),
        [[act(cfg, 2, 2, RIGHT), 0]])

    add("move_half_floor",
        lambda o, a, k: (o.__setitem__((2, 2), 0), a.__setitem__((2, 2), 9)),
        [[act(cfg, 2, 2, RIGHT, True), 0]])

    def two_sided(o, a, k):
        o[2, 2], a[2, 2] = 0, 10
        o[2, 3], a[2, 3] = 1, 4

    add("attack_captures", two_sided, [[act(cfg, 2, 2, RIGHT), 0]])

    def tie(o, a, k):
        o[2, 2], a[2, 2] = 0, 6
        o[2, 3], a[2, 3] = 1, 5

    add("tie_keeps_defender", tie, [[act(cfg, 2, 2, RIGHT), 0]])

    def castle40(o, a, k):
        o[2, 2], a[2, 2] = 0, 41
        k[2, 3], a[2, 3] = True, 40

    add("castle_drained_not_taken", castle40, [[act(cfg, 2, 2, RIGHT), 0]])

    def castle45(o, a, k):
        o[2, 2], a[2, 2] = 0, 45
        k[2, 3], a[2, 3] = True, 40

    add("castle_captured", castle45, [[act(cfg, 2, 2, RIGHT), 0]])

    def headon(o, a, k):
        o[2, 1], a[2, 1] = 0, 10
        o[2, 3], a[2, 3] = 1, 6

    add("head_on_collision", headon, [[act(cfg, 2, 1, RIGHT), act(cfg, 2, 3, LEFT)]])

    def swap(o, a, k):
        o[2, 2], a[2, 2] = 0, 10
        o[2, 3], a[2, 3] = 1, 8

    add("mutual_swap", swap, [[act(cfg, 2, 2, RIGHT), act(cfg, 2, 3, LEFT)]])

    add("general_capture_ends_game",
        lambda o, a, k: (o.__setitem__((0, 1), 1), a.__setitem__((0, 1), 9)),
        [[0, act(cfg, 0, 1, LEFT)], [0, 0], [0, 0]])

    add("production_schedule",
        lambda o, a, k: (o.__setitem__((2, 2), 0), a.__setitem__((2, 2), 3)),
        [[0, 0]] * 52)

    # A three-player position: capturing a general hands over the cells at half strength.
    cfg3 = EnvConfig(height=5, width=5, num_players=3, max_steps=100)
    owner = np.full((h, w), -1, np.int8)
    army = np.zeros((h, w), np.int32)
    gen = np.zeros((h, w), bool)
    for p, (r, c) in enumerate([(0, 0), (0, 4), (4, 4)]):
        owner[r, c], army[r, c], gen[r, c] = p, 1, True
    owner[2, 2], army[2, 2] = 1, 30          # player 1 holds a fat cell
    owner[0, 3], army[0, 3] = 2, 20          # player 2 is next to player 1's general
    st3 = make_state(cfg3, owner, army, general=gen)
    case = roll(cfg3, st3, [[0, 0, act(cfg3, 0, 3, RIGHT)], [0, 0, 0]])
    case["name"] = "ffa_general_capture_transfers_cells"
    cases.append(case)

    return cases


def random_cases(count: int = 6, ticks: int = 90) -> list[dict]:
    """Full random maps driven by random actions -- broad coverage of the step function."""
    cases = []
    for seed in range(count):
        size = [9, 11, 13][seed % 3]
        cfg = EnvConfig(height=size, width=size, min_spawn_dist=4,
                        max_spawn_dist=4 * size, bfs_iters=4 * size, max_steps=400)
        key = jax.random.PRNGKey(1000 + seed)
        state = reset(key, cfg)
        rng = np.random.default_rng(seed)
        # Mostly legal moves, with a slice of nonsense to exercise the fallbacks.
        actions = []
        s = state
        for _ in range(ticks):
            a = []
            for p in range(cfg.num_players):
                owned = np.flatnonzero(
                    (np.asarray(s.owner).reshape(-1) == p) & (np.asarray(s.army).reshape(-1) >= 2)
                )
                if len(owned) and rng.random() < 0.85:
                    cell = int(rng.choice(owned))
                    a.append(int(cell * NUM_CELL_ACTIONS + 1 + rng.integers(0, 8)))
                else:
                    a.append(int(rng.integers(0, cfg.num_actions)))
            actions.append(a)
            s, *_ = step(s, jnp.asarray(a, jnp.int32), cfg)
        case = roll(cfg, state, actions)
        case["name"] = f"random_{size}x{size}_seed{seed}"
        cases.append(case)
    return cases


def main() -> None:
    cases = scenario_cases() + random_cases()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": cases}, separators=(",", ":")))
    total = sum(len(c["ticks"]) for c in cases)
    print(f"wrote {OUT.relative_to(pathlib.Path.cwd())}: "
          f"{len(cases)} cases, {total} ticks, {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
