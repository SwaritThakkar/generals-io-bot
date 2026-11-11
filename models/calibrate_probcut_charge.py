"""Fit the charge-line ProbCut model on tactical positions from the game rules.

The samples are generated from valid BotState boards with a producer stack
charging along a clear corridor to a general. Randomized garrisons, production
timing, and producer sizes cover the wait/launch boundary that the extension
actually searches. Run from the repository root with ``python3.12``.
"""
from __future__ import annotations

import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from server.bot_protocol import BotState
from models import v19_waiting_defense_probcut as bot


def position(seed: int, height: int = 9, width: int = 13) -> BotState:
    rng = np.random.default_rng(seed)
    h, w = int(height), int(width)
    mid = h // 2
    owner = np.full((h, w), -1, np.int8)
    army = np.zeros((h, w), np.int32)
    mountain = np.zeros((h, w), bool)
    castle = np.zeros((h, w), bool)
    general = np.zeros((h, w), bool)
    general[mid, 0] = general[mid, w - 1] = True
    owner[mid, 0], owner[mid, w - 1] = 0, 1
    army[mid, 0], army[mid, w - 1] = rng.integers(6, 20), rng.integers(2, 13)
    # The attackers collect at an owned castle; the defender owns a second
    # producer so both sides have a meaningful response/charge branch.
    c0, c1 = max(2, w // 4), min(w - 3, 3 * w // 4)
    castle[mid, c0] = castle[mid, c1] = True
    owner[mid, c0], owner[mid, c1] = 0, 1
    army[mid, c0], army[mid, c1] = rng.integers(2, 13), rng.integers(3, 16)
    # Keep the shortest corridor open, while varying side terrain and neutral
    # garrisons to produce different attrition and shallow/deep values.
    for r in range(h):
        for c in range(w):
            if r == mid:
                continue
            if rng.random() < 0.13:
                mountain[r, c] = True
                army[r, c] = 0
            elif rng.random() < 0.18:
                army[r, c] = rng.integers(1, 5)
    tick = int(rng.integers(1, 50))
    return BotState(
        h, w, 2, tick, owner, army, mountain, castle, general,
        np.asarray([mid * w, mid * w + w - 1], np.int32),
        np.asarray([True, True]), np.asarray([2, 2], np.int32),
        np.asarray([int(army[owner == 0].sum()), int(army[owner == 1].sum())]),
        {"generalGrowthInterval": 2, "landGrowthInterval": 50,
         "maxTicks": 400, "minSpawnDist": 2, "tickMs": 500},
    )


def main() -> None:
    bot._PROBCUT_PAIRS.clear()
    for seed in range(80):
        state = position(seed + 90100)
        # Use the same tactical entry point as play; it generates shallow
        # matrix values and evaluates queued second-ply charge sequences.
        bot.act(state, 1)
    fit = bot._probcut_fit()
    if fit is None:
        raise SystemExit(f"Only {len(bot._PROBCUT_PAIRS)} samples; need at least 12")
    a, b, sigma = fit
    data = np.asarray(bot._PROBCUT_PAIRS)
    corr = float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
    print(f"samples={len(data)} slope={a:.6f} intercept={b:.6f} "
          f"sigma={sigma:.6f} correlation={corr:.4f}")
    print(f"saved={bot._PROBCUT_PATH}")


if __name__ == "__main__":
    main()
