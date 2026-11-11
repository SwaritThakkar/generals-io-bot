#!/usr/bin/env python3
"""Throughput benchmark: environment steps per second, batched on one device.

    python benchmark.py --envs 256 1024 4096 --steps 200
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # project root

import argparse
import time

import jax
import jax.numpy as jnp

import genmod
from genmod import EnvConfig
from genmod.obs import observe


def bench(cfg: EnvConfig, num_envs: int, num_steps: int, with_obs: bool) -> float:
    key = jax.random.PRNGKey(0)
    state = genmod.reset_batch(key, cfg, num_envs)
    policy = genmod.random_policy(cfg)

    if with_obs:
        vobs = jax.vmap(lambda s: observe(s, cfg))
        base = policy

        def policy(k, s, c):  # force the observation into the graph
            o = vobs(s)
            return base(k, s, c) + (o.sum() * 0).astype(jnp.int32)

    roll = genmod.make_rollout(cfg, policy, num_steps)
    state, out = roll(jax.random.PRNGKey(1), state)   # compile
    jax.block_until_ready(out)

    t0 = time.perf_counter()
    state, out = roll(jax.random.PRNGKey(2), state)
    jax.block_until_ready(out)
    dt = time.perf_counter() - t0
    return num_envs * num_steps / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--size", type=int, default=23)
    ap.add_argument("--obs", action="store_true", help="also build observation tensors")
    args = ap.parse_args()

    cfg = EnvConfig(height=args.size, width=args.size, bfs_iters=4 * args.size)
    print(f"device: {jax.devices()[0]}   grid: {args.size}x{args.size}   "
          f"steps/rollout: {args.steps}   observations: {'on' if args.obs else 'off'}")
    print(f"{'envs':>8}  {'frames/s':>14}")
    for n in args.envs:
        fps = bench(cfg, n, args.steps, args.obs)
        print(f"{n:>8}  {fps:>14,.0f}")


if __name__ == "__main__":
    main()
