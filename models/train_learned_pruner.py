#!/usr/bin/env python3
"""Train a conservative branch rejector from exact shallow matrix labels.

The model predicts only ``safe_to_prune`` for extra action-bucket branches.
It does not predict or return a game move. Hard tactical/script branches are
excluded from training and protected by v32 at inference.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import genmod
from genmod import EnvConfig
from genmod.bots import expander_bot
from models import v26_bucketed_action_probcut as buckets
from models import v27_live_race_matrix as teacher
from models.learned_pruning import FEATURES, branch_features, cheap_score
from server.bot_protocol import BotState


def _bot_state(st, cfg):
    owner, army, mountain = (np.asarray(st.owner), np.asarray(st.army),
                             np.asarray(st.mountain))
    land = np.asarray([(owner == p).sum() for p in range(cfg.num_players)], np.int32)
    totals = np.asarray([army[owner == p].sum() for p in range(cfg.num_players)], np.int32)
    return BotState(cfg.height, cfg.width, cfg.num_players, int(st.step), owner.copy(),
        army.copy(), mountain.copy(), np.asarray(st.castle).copy(),
        np.asarray(st.general).copy(), np.asarray(st.general_idx).copy(),
        np.asarray(st.alive).copy(), land, totals,
        {"maxTicks": cfg.max_steps, "generalGrowthInterval": cfg.general_growth_interval,
         "landGrowthInterval": cfg.land_growth_interval, "minSpawnDist": cfg.min_spawn_dist})


def _positions(cfg, count, seed):
    """Collect opening, expanding, and long-wait producer states."""
    out = []
    for i in range(count):
        st = genmod.reset(jax.random.PRNGKey(seed + i), cfg)
        wait_side = i % 2
        for tick in range(101):
            if tick in (0, 15, 30, 50, 75, 100):
                out.append((_bot_state(st, cfg), i))
            actions = [0, 0]
            # One side demonstrates producer camping; the other expands.
            actions[1-wait_side] = int(expander_bot(
                jax.random.PRNGKey(seed * 1000 + i * 101 + tick), st, cfg,
                player=1-wait_side))
            st, _, done, _ = genmod.step(st, jnp.asarray(actions, jnp.int32), cfg)
            if bool(done):
                break
    return out


def _labels_for_state(state, cfg):
    xs, ys = [], []
    for side in range(2):
        core = list(dict.fromkeys(int(a) for a in teacher._live_action_set(state, side)))
        extras = [int(a) for a in buckets._bucket_action_candidates(state, side, limit=8)
                  if int(a) not in set(core)]
        if not extras:
            continue
        mine = list(dict.fromkeys(core + extras))
        theirs = list(dict.fromkeys(teacher._live_action_set(state, 1-side)))
        if not theirs:
            theirs = [0]
        gs = state.to_genmod_state(cfg)
        pairs = []
        for a in mine:
            for b in theirs:
                joint = [0, 0]
                joint[side], joint[1-side] = a, b
                pairs.append(joint)
        children = teacher._joint_successors(gs, jnp.asarray(pairs, jnp.int32), cfg)
        scores = np.asarray([teacher._score_successor(
            jax.tree.map(lambda x, k=k: x[k], children), side, cfg)
            for k in range(len(pairs))], np.float64).reshape(len(mine), len(theirs))
        shallow_values=np.asarray([cheap_score(
            jax.tree.map(lambda x,k=k:x[k],children),side,cfg.max_steps)
            for k in range(len(pairs))],np.float64).reshape(len(mine),len(theirs))
        security = scores.min(axis=1)
        best = float(security.max())
        pass_col = theirs.index(0) if 0 in theirs else int(np.argmin(np.abs(theirs)))
        shallow_scores = shallow_values[:, pass_col]
        pass_row = mine.index(0) if 0 in mine else 0
        shallow_baseline = float(shallow_values[pass_row, pass_col])
        shallow_best = float(shallow_scores.max())
        # Search labels: a branch is irrelevant only when its worst-case value
        # is clearly behind the best branch. Core branches never enter data.
        margin = max(2.0, abs(best) * 0.025)
        for index, action in enumerate(mine):
            if action not in extras:
                continue
            xs.append(branch_features(state, side, action, core,
                float(shallow_scores[index] - shallow_baseline),
                float(shallow_scores[index] - shallow_best)))
            ys.append(float(security[index] < best - margin))
    return xs, ys


def _fit_mlp(x, y, seed, steps=1800):
    rng = np.random.default_rng(seed)
    d, hidden = x.shape[1], 24
    w1 = rng.normal(0, 0.18, (d, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, np.float32)
    w2 = rng.normal(0, 0.12, (hidden, 1)).astype(np.float32)
    b2 = np.zeros(1, np.float32)
    params = [w1, b1, w2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    batch = min(256, len(y))
    for step in range(1, steps+1):
        ix = rng.integers(0, len(y), batch)
        xb, yb = x[ix], y[ix, None]
        z = xb @ w1 + b1
        h = np.tanh(z)
        logits = h @ w2 + b2
        prob = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        dz2 = (prob-yb) / batch
        gw2 = h.T @ dz2
        gb2 = dz2.sum(axis=0)
        dh = dz2 @ w2.T
        dz1 = dh * (1-h*h)
        gw1, gb1 = xb.T @ dz1, dz1.sum(axis=0)
        grads = [gw1, gb1, gw2, gb2]
        for i, grad in enumerate(grads):
            m[i] = .9*m[i] + .1*grad
            v[i] = .999*v[i] + .001*grad*grad
            mh = m[i]/(1-.9**step)
            vh = v[i]/(1-.999**step)
            params[i] -= .002*mh/(np.sqrt(vh)+1e-8)
    return params


def _prob(params, x):
    w1,b1,w2,b2=params
    z=np.tanh(x@w1+b1)@w2+b2
    return 1/(1+np.exp(-np.clip(z.reshape(-1),-30,30)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--maps',type=int,default=24)
    ap.add_argument('--seed',type=int,default=41000)
    ap.add_argument('--steps',type=int,default=1800)
    args=ap.parse_args()
    cfg=EnvConfig(height=23,width=23,max_steps=400,bfs_iters=92)
    print('JAX devices:',jax.devices(),flush=True)
    states=_positions(cfg,args.maps,args.seed)
    features=[]; labels=[]; groups=[]
    for i,(state,map_id) in enumerate(states):
        try:
            x,y=_labels_for_state(state,cfg)
            features.extend(x); labels.extend(y); groups.extend([map_id]*len(y))
        except (ValueError,IndexError,TypeError) as exc:
            print(f'skip state {i}: {exc}',flush=True)
        if (i+1)%10==0: print(f'labelled positions {i+1}/{len(states)} samples={len(labels)}',flush=True)
    x=np.asarray(features,np.float32); y=np.asarray(labels,np.float32)
    if len(y)<100 or y.min()==y.max():
        raise SystemExit(f'insufficient label variety: samples={len(y)}, positives={int(y.sum())}')
    # Split by position order so nearby snapshots from one map do not leak.
    groups=np.asarray(groups)
    train_mask=groups < int(.8*args.maps)
    valid_mask=~train_mask
    if not train_mask.any() or not valid_mask.any():
        raise SystemExit('map-group split is empty')
    mean=x[train_mask].mean(0); scale=np.maximum(x[train_mask].std(0),1e-3)
    xn=(x-mean)/scale
    params=_fit_mlp(xn[train_mask],y[train_mask],args.seed+9,args.steps)
    pv=_prob(params,xn[valid_mask]); yv=y[valid_mask]
    chosen_threshold=1.01; precision=0.; recall=0.
    for threshold in np.linspace(.90,.9999,100):
        selected=pv>=threshold
        if selected.sum() >= 30:
            pr=float(yv[selected].mean())
            if pr>=.995 and selected.sum()>0:
                chosen_threshold=float(threshold); precision=pr
                recall=float((selected & (yv>0.5)).sum()/max(1,(yv>0.5).sum()))
    enabled=chosen_threshold<=1.0 and precision>=.995
    model={'version':1,'enabled':enabled,'threshold':chosen_threshold,
        'mean':mean.tolist(),'scale':scale.tolist(),
        'w1':params[0].tolist(),'b1':params[1].tolist(),
        'w2':params[2].tolist(),'b2':params[3].tolist(),
        'features':FEATURES,'samples':len(y),'prunable_fraction':float(y.mean()),
        'validation_precision_at_threshold':precision,'validation_recall':recall,
        'validation_prunes':int((pv>=chosen_threshold).sum()) if enabled else 0,
        'training_device':str(jax.devices()[0]),'label':'one-ply matrix dominated extra branch'}
    out=pathlib.Path(__file__).with_name('learned_pruner_v32.json')
    out.write_text(json.dumps(model,separators=(',',':')))
    np.savez_compressed(pathlib.Path(__file__).with_name('learned_pruner_v32_data.npz'),
                        x=x,y=y,groups=groups)
    print(f'samples={len(y)} prunable={y.mean():.3f} validation_precision={precision:.3f} '
          f'recall={recall:.3f} prunes={model["validation_prunes"]} enabled={enabled}')
    print('saved',out,flush=True)

if __name__=='__main__': main()
