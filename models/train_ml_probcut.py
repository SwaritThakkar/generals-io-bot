"""Precompute tactical ProbCut pairs and train a nonlinear JAX value model.

Run from the repository root with ``python3.12 models/train_ml_probcut.py``.
JAX places the batched training work on its default accelerator (GPU when a
GPU-enabled JAX build and supported device are present; CPU otherwise).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.calibrate_probcut_charge import position
from models import v20_ml_probcut as bot


def _fit_linear(x, y):
    design = np.column_stack([x[:, 0], np.ones(len(x))])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return design @ coef


def _fit_mlp(x_train, y_train, key, steps=2400):
    """Small tanh regressor; all dense training batches execute in JAX."""
    n, d = x_train.shape
    k1, k2 = jax.random.split(key)
    params = {
        "w1": jax.random.normal(k1, (d, 24), dtype=jnp.float32) * 0.15,
        "b1": jnp.zeros((24,), jnp.float32),
        "w2": jax.random.normal(k2, (24, 1), dtype=jnp.float32) * 0.1,
        "b2": jnp.zeros((1,), jnp.float32),
    }
    opt = optax.adam(learning_rate=0.003)
    state = opt.init(params)
    X, Y = jnp.asarray(x_train), jnp.asarray(y_train[:, None])

    @jax.jit
    def train_step(params, state, xb, yb):
        def loss_fn(p):
            pred = jnp.tanh(xb @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]
            err = pred - yb
            # Huber loss protects the fit from terminal-like score outliers.
            ae = jnp.abs(err)
            loss = jnp.mean(jnp.where(ae < 2.0, 0.5 * err * err, 2.0 * (ae - 1.0)))
            loss += 1e-4 * (jnp.sum(p["w1"] ** 2) + jnp.sum(p["w2"] ** 2))
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    batch = min(64, n)
    rng = np.random.default_rng(43020)
    for _ in range(steps):
        ids = rng.integers(0, n, batch)
        params, state, _ = train_step(params, state, X[ids], Y[ids])
    return {k: np.asarray(v) for k, v in params.items()}


def _predict(params, x):
    return np.asarray(np.tanh(x @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=420,
                    help="number of randomized valid producer-wait boards")
    ap.add_argument("--seed", type=int, default=90100)
    ap.add_argument("--steps", type=int, default=2400)
    args = ap.parse_args()

    print(f"JAX devices: {jax.devices()}", flush=True)
    bot._ML_SAMPLES.clear()
    bot._PROBCUT_PAIRS.clear()
    t0 = time.perf_counter()
    for i in range(args.positions):
        st = position(args.seed + i)
        bot.act(st, i % 2)
        if (i + 1) % 50 == 0:
            print(f"precomputed {i + 1}/{args.positions}; pairs={len(bot._ML_SAMPLES)} "
                  f"elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    if len(bot._ML_SAMPLES) < 100:
        raise SystemExit(f"Only {len(bot._ML_SAMPLES)} deep pairs; increase --positions")

    feats = np.asarray([p[0] for p in bot._ML_SAMPLES], dtype=np.float32)
    target = np.asarray([p[1] for p in bot._ML_SAMPLES], dtype=np.float32)
    # Deterministic held-out suffix avoids randomly leaking nearly identical
    # neighboring generated seeds across train and validation.
    order = np.arange(len(target))
    split = max(1, int(len(order) * 0.8))
    train, valid = order[:split], order[split:]
    x_mean = feats[train].mean(axis=0)
    x_scale = np.maximum(feats[train].std(axis=0), 1e-3)
    x = (feats - x_mean) / x_scale
    linear_model = np.linalg.lstsq(
        np.column_stack([feats[train, 0], np.ones(len(train))]), target[train], rcond=None)[0]
    baseline_train = feats[train, 0] * linear_model[0] + linear_model[1]
    baseline_valid = feats[valid, 0] * linear_model[0] + linear_model[1]
    # Learn a nonlinear correction to the standard shallow/deep regression.
    params = _fit_mlp(x[train], target[train] - baseline_train,
                      jax.random.PRNGKey(2020), args.steps)
    pred = baseline_valid + _predict(params, x[valid])
    linear = baseline_valid
    ml_mae = float(np.mean(np.abs(target[valid] - pred)))
    linear_mae = float(np.mean(np.abs(target[valid] - linear)))
    residual_q90 = float(np.quantile(np.abs(target[valid] - pred), 0.90))
    # Use the neural model only if held-out error beats the existing simple
    # shallow/deep line by a meaningful margin; otherwise retain safe fallback.
    use_ml = ml_mae <= 0.98 * linear_mae
    model = {
        "version": 1, "features": ["shallow", "producer_delta", "army_delta",
            "own_stack", "enemy_stack", "own_wait", "own_travel", "enemy_wait",
            "enemy_eta", "pass_action", "tick"],
        "x_mean": x_mean.tolist(), "x_scale": x_scale.tolist(),
        **{k: np.asarray(v).tolist() for k, v in params.items()},
        "linear_a": float(linear_model[0]), "linear_b": float(linear_model[1]),
        "residual_q90": residual_q90, "use_ml": use_ml,
        "samples": int(len(target)), "train_samples": int(len(train)),
        "validation_samples": int(len(valid)), "validation_mae_ml": ml_mae,
        "validation_mae_linear": linear_mae,
        "training_device": str(jax.devices()[0]),
    }
    path = pathlib.Path(__file__).with_name("probcut_ml_v20.json")
    path.write_text(json.dumps(model, separators=(",", ":")))
    print(f"samples={len(target)} train={len(train)} validation={len(valid)}")
    print(f"validation MAE: MLP={ml_mae:.4f}, linear={linear_mae:.4f}; "
          f"90% abs residual={residual_q90:.4f}; enabled={use_ml}")
    print(f"device={jax.devices()[0]} saved={path}")


if __name__ == "__main__":
    main()
