"""Batched rollouts: `vmap` over environments, `lax.scan` over time.

The whole loop -- map generation, stepping, observation, auto-reset -- compiles
into a single XLA graph, which is what makes the throughput numbers in the paper
possible.  Nothing here allocates on the host inside the loop.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .config import EnvConfig
from .env import reset, step, step_auto_reset
from .obs import observe
from .state import State

#: A policy maps (key, batched_state, cfg) -> (num_envs, num_players) actions.
Policy = Callable[[jax.Array, State, EnvConfig], jax.Array]


class Transition(NamedTuple):
    rewards: jax.Array   # (T, B, N)
    done: jax.Array      # (T, B)
    winner: jax.Array    # (T, B)


@partial(jax.jit, static_argnames=("cfg", "num_envs"))
def reset_batch(key: jax.Array, cfg: EnvConfig, num_envs: int) -> State:
    return jax.vmap(reset, in_axes=(0, None))(jax.random.split(key, num_envs), cfg)


def make_step_batch(cfg: EnvConfig, auto_reset: bool = True):
    """Return a jitted `(state, actions) -> (state, rewards, done, info)`."""
    fn = step_auto_reset if auto_reset else step
    return jax.jit(jax.vmap(partial(fn, cfg=cfg), in_axes=(0, 0)))


def make_observe_batch(cfg: EnvConfig):
    return jax.jit(jax.vmap(partial(observe, cfg=cfg)))


def make_rollout(cfg: EnvConfig, policy: Policy, num_steps: int, auto_reset: bool = True):
    """Compile a `(key, state) -> (state, Transition)` rollout of `num_steps`."""
    step_fn = step_auto_reset if auto_reset else step
    vstep = jax.vmap(partial(step_fn, cfg=cfg), in_axes=(0, 0))

    def scan_body(carry, _):
        state, key = carry
        key, sub = jax.random.split(key)
        actions = policy(sub, state, cfg)
        state, rewards, done, _ = vstep(state, actions)
        return (state, key), Transition(rewards, done, state.winner)

    @jax.jit
    def rollout(key: jax.Array, state: State):
        key, sub = jax.random.split(key)
        (state, _), traj = jax.lax.scan(scan_body, (state, sub), None, length=num_steps)
        return state, traj

    return rollout


def bot_policy(bot_fns) -> Policy:
    """Turn a per-seat list of `(key, state, cfg, player) -> action` bots into a Policy."""

    def policy(key, state, cfg):
        keys = jax.random.split(key, len(bot_fns))
        per_seat = [
            jax.vmap(partial(fn, cfg=cfg, player=i))(
                jax.random.split(keys[i], state.step.shape[0]), state
            )
            for i, fn in enumerate(bot_fns)
        ]
        return jnp.stack(per_seat, axis=-1)

    return policy


def random_policy(cfg: EnvConfig) -> Policy:
    """Uniform over the whole action space (illegal actions degrade to a pass)."""

    def policy(key, state, cfg):
        b = state.step.shape[0]
        return jax.random.randint(key, (b, cfg.num_players), 0, cfg.num_actions, dtype=jnp.int32)

    return policy
