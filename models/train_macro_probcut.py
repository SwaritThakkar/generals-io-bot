"""Precompute full wait/charge minimax values and train horizon-specific ProbCut.

The rollout search evaluates simultaneous wait/castle/general scripts for 96
actual game ticks. Run with Python 3.12; JAX uses its default CPU/GPU device.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import jax
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.calibrate_probcut_charge import position
from models import v22_macro_wait_probcut as bot
from models.train_ml_probcut import _fit_mlp, _predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=420)
    ap.add_argument("--seed", type=int, default=93100)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--size", type=int, default=23, help="square match board size")
    args = ap.parse_args()
    print(f"JAX devices: {jax.devices()} horizon={bot._WAIT_HORIZON}", flush=True)
    bot._MACRO_SAMPLES.clear()
    bot._MACRO_FORCE_ALL = True
    start = time.perf_counter()
    for i in range(args.positions):
        st = position(args.seed + i, args.size, args.size)
        mine = bot._plan_candidates(st, 0)
        theirs = bot._plan_candidates(st, 1)
        bot._macro_wait_choice(st, 0, mine, theirs)
        if (i + 1) % 50 == 0:
            print(f"precomputed {i+1}/{args.positions}; rows={len(bot._MACRO_SAMPLES)} "
                  f"elapsed={time.perf_counter()-start:.1f}s", flush=True)
    if len(bot._MACRO_SAMPLES) < 150:
        raise SystemExit(f"Only {len(bot._MACRO_SAMPLES)} rows; increase --positions")

    feats = np.asarray([row[0] for row in bot._MACRO_SAMPLES], np.float32)
    targets = np.asarray([row[1] for row in bot._MACRO_SAMPLES], np.float32)
    split = max(1, int(len(targets) * 0.8))
    train = np.arange(split)
    valid = np.arange(split, len(targets))
    mean = feats[train].mean(axis=0)
    scale = np.maximum(feats[train].std(axis=0), 1e-3)
    x = (feats - mean) / scale
    coef = np.linalg.lstsq(np.column_stack([feats[train, 0], np.ones(len(train))]),
                           targets[train], rcond=None)[0]
    base_train = coef[0] * feats[train, 0] + coef[1]
    base_valid = coef[0] * feats[valid, 0] + coef[1]
    params = _fit_mlp(x[train], targets[train] - base_train,
                      jax.random.PRNGKey(22022), args.steps)
    pred = base_valid + _predict(params, x[valid])
    mae_ml = float(np.mean(np.abs(targets[valid] - pred)))
    mae_linear = float(np.mean(np.abs(targets[valid] - base_valid)))
    q90 = float(np.quantile(np.abs(targets[valid] - pred), 0.90))
    model = {
        "version": 1,
        "features": ["shallow", "producer_delta", "army_delta", "own_stack",
                     "enemy_stack", "own_wait", "own_travel", "enemy_wait",
                     "enemy_eta", "pass_action", "tick"],
        "x_mean": mean.tolist(), "x_scale": scale.tolist(),
        **{k: np.asarray(v).tolist() for k, v in params.items()},
        "linear_a": float(coef[0]), "linear_b": float(coef[1]),
        "residual_q90": q90, "use_ml": mae_ml <= 0.98 * mae_linear,
        "samples": len(targets), "train_samples": len(train),
        "validation_samples": len(valid), "validation_mae_ml": mae_ml,
        "validation_mae_linear": mae_linear, "horizon_ticks": bot._WAIT_HORIZON,
        "board_size": args.size,
        "training_device": str(jax.devices()[0]),
    }
    path = pathlib.Path(__file__).with_name("probcut_ml_v22.json")
    path.write_text(json.dumps(model, separators=(",", ":")))
    archive = pathlib.Path(__file__).with_name("archive")
    archive.mkdir(exist_ok=True)
    data_path = archive / f"probcut_macro_pairs_v22_{args.size}x{args.size}_{args.positions}.npz"
    np.savez_compressed(data_path, features=feats, targets=targets,
                        seed=args.seed, board_size=args.size,
                        horizon=bot._WAIT_HORIZON)
    print(f"samples={len(targets)} train={len(train)} validation={len(valid)}")
    print(f"96-tick MAE: MLP={mae_ml:.4f}, linear={mae_linear:.4f}; "
          f"90% residual={q90:.4f}; enabled={model['use_ml']}")
    print(f"device={jax.devices()[0]} saved={path}")
    print(f"training_pairs={data_path}")


if __name__ == "__main__":
    main()
