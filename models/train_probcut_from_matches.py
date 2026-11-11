"""Collect charge-line ProbCut samples from matches in the actual JAX engine."""
from __future__ import annotations

import json
import pathlib
import sys

import jax
import jax.numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import genmod
from models import castle_general_rusher as rusher
from models import v13_probcut_charge as bot
from server.local_server import to_bot_state
from server.bot_protocol import normalise


def main() -> None:
    cfg = genmod.EnvConfig(max_steps=400)
    bot._PROBCUT_PAIRS.clear()
    bot._PROBCUT_WRITES = 0
    games = [(53000, 1), (56000, 0)]
    for seed, bot_seat in games:
        state = genmod.reset(jax.random.PRNGKey(seed), cfg)
        while not bool(state.done):
            acts = [0, 0]
            for player in range(2):
                bs = to_bot_state(state, cfg, player)
                agent = bot if player == bot_seat else rusher
                result = agent.act(bs, player)
                actions = normalise(result, bs)
                acts[player] = actions[0] if actions else 0
            state, _, _, _ = genmod.step(state, jnp.asarray(acts, jnp.int32), cfg)
        print(f"seed={seed} bot_seat={bot_seat} winner={int(state.winner)} "
              f"samples={len(bot._PROBCUT_PAIRS)}", flush=True)
    fit = bot._probcut_fit()
    if fit is None:
        raise SystemExit("The matches did not produce enough producer-wait nodes to fit ProbCut")
    a, b, sigma = fit
    bot._PROBCUT_PATH.write_text(json.dumps({"pairs": bot._PROBCUT_PAIRS}, separators=(",", ":")))
    print(f"samples={len(bot._PROBCUT_PAIRS)} slope={a:.6f} intercept={b:.6f} "
          f"sigma={sigma:.6f} saved={bot._PROBCUT_PATH}")


if __name__ == "__main__":
    main()
