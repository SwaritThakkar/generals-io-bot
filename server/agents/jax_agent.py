"""Template for a trained JAX policy.

Load this from the local server's page (or with `--p0 server/agents/jax_agent.py:act`)
and it runs in the same interpreter as your training code: real JAX, real
checkpoints, GPU if you have one. That is the reason the file picker lives in the
local server and not the deployed site -- a browser has no XLA, so a policy like
this can only run here.

Swap `_score` for your network and the rest stays as it is.
"""

from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import genmod  # noqa: E402


@jax.jit
def _score(obs: jax.Array, mask: jax.Array) -> jax.Array:
    """Stand-in for a policy head: (C, H, W) observation -> (H*W*9) logits.

    Replace the body with `params`-driven inference. The only contract is that
    illegal actions come back as -inf.
    """
    own_army = obs[0]  # channel 0 is own_army, log-compressed
    logits = jnp.broadcast_to(own_army[..., None], mask.shape) * 4.0

    # Channel 0 of the mask is "pass", and it is legal on every cell. Leave it in
    # and a flat argmax will pass forever; drop it and the policy must commit.
    mask = mask.at[..., 0].set(False)
    return jnp.where(mask, logits, -jnp.inf).reshape(-1)


def act(state, player: int) -> int:
    """Called once per tick. `state` is a `BotState`: plain numpy, nothing hidden."""
    cfg = state.env_config()
    st = state.to_genmod_state(cfg)

    obs = genmod.observe(st, cfg)[player]              # (C, H, W) float32
    mask = genmod.legal_action_mask(st, cfg)[player]   # (H, W, 9) bool

    logits = _score(obs, mask)
    if not bool(jnp.isfinite(logits).any()):
        return 0  # nothing legal: pass
    return int(jnp.argmax(logits))


if __name__ == "__main__":
    # Also servable over HTTP, if you ever want the deployed site to reach it.
    import argparse

    from server.bot_protocol import serve

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8802)
    serve(act, port=ap.parse_args().port, name="JAX policy")
