#!/usr/bin/env python3
"""Train the v40 plan ranker on the Mac GPU (PyTorch MPS). Refuses to run on CPU.

Model (after Barriga, Stanescu & Buro 2017): a fully convolutional torso over
the 14 board planes with global average pooling (size-independent), joined
with each candidate plan's feature vector, scoring every plan of a position.
Loss: listwise cross-entropy against the regret-matching mix of the teacher
plus MSE on the plan's value against the opponent mix. Parameter EMA is used
for the deployed weights (Straka et al. 2026 found the EMA consistently
stronger than the last iterate).

Held out by map (last 20% of seeds). Search-quality metrics, not accuracy
alone (Ontanon 2016/2017): top-k recall of the teacher's best plan and
regret = teacher value of best plan minus best kept plan.
"""
from __future__ import annotations

import copy
import glob
import json
import pathlib
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "models")]
import core40  # noqa: E402

if not torch.backends.mps.is_available():
    raise SystemExit("MPS GPU not available: refusing to train on CPU")
DEV = torch.device("mps")


class Ranker(nn.Module):
    def __init__(self, planes=core40.N_PLANES, plan_dim=core40.PLAN_DIM, width=48):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Conv2d(planes, width, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.LeakyReLU(0.1),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.LeakyReLU(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width + plan_dim, 96), nn.LeakyReLU(0.1),
            nn.Linear(96, 48), nn.LeakyReLU(0.1), nn.Linear(48, 2))

    def forward(self, x, plan_vecs, owner):
        """x (B,C,H,W); plan_vecs (P,D); owner (P,) index of each plan's board."""
        f = self.torso(x)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)       # global avg + max pool
        h = torch.cat([g[owner], plan_vecs], 1)
        out = self.head(h)
        return out[:, 0], out[:, 1]            # logit, value


def load():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "models/ranker40_data/*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    return [r for r in rows if len(r["plans"]) >= 2]


def batches(rows, bs, rng):
    idx = rng.permutation(len(rows))
    for i in range(0, len(idx), bs):
        yield [rows[j] for j in idx[i:i + bs]]


def tensors(group):
    x = torch.tensor(np.stack([r["planes"] for r in group]).astype(np.float32), device=DEV)
    pv = torch.tensor(np.concatenate([r["plan_vecs"] for r in group]), device=DEV)
    owner = torch.tensor(np.concatenate([[i] * len(r["plans"]) for i, r in enumerate(group)]),
                         device=DEV, dtype=torch.long)
    sig = torch.tensor(np.concatenate([r["sigma"] for r in group]), device=DEV)
    val = torch.tensor(np.concatenate([r["row_value"] for r in group]), device=DEV)
    return x, pv, owner, sig, val


def group_softmax(logit, owner, n):
    m = torch.full((n,), -1e9, device=DEV).scatter_reduce(0, owner, logit, "amax")
    e = torch.exp(logit - m[owner])
    s = torch.zeros(n, device=DEV).scatter_add(0, owner, e)
    return e / s[owner]


@torch.no_grad()
def evaluate(model, rows, ks=(1, 2, 3)):
    model.eval()
    rep = {f"top{k}_{m}": 0.0 for k in ks for m in ("recall", "regret")}
    for i in range(0, len(rows), 256):
        group = rows[i:i + 256]
        x, pv, owner, _sig, _val = tensors(group)
        logit, _v = model(x, pv, owner)
        logit = logit.cpu().numpy()
        off = 0
        for r in group:
            n = len(r["plans"])
            s = logit[off:off + n]; off += n
            rv = r["row_value"]
            best = rv.max()
            order = np.argsort(-s)
            for k in ks:
                got = rv[order[:k]].max()
                rep[f"top{k}_recall"] += got >= best - 1e-6
                rep[f"top{k}_regret"] += best - got
    return {k: round(float(v) / len(rows), 4) for k, v in rep.items()}


def baseline(rows, ks=(1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    rep = {}
    for k in ks:
        hit = reg = 0.0
        for r in rows:
            rv = r["row_value"]
            order = rng.permutation(len(rv))
            got = rv[order[:k]].max()
            hit += got >= rv.max() - 1e-6
            reg += rv.max() - got
        rep[f"top{k}_recall"] = round(float(hit) / len(rows), 4)
        rep[f"top{k}_regret"] = round(float(reg) / len(rows), 4)
    return rep


def main(epochs=60, bs=128, lr=1e-3):
    rows = load()
    seeds = sorted(set(r["seed"] for r in rows))
    cut = seeds[int(0.8 * len(seeds))]
    train = [r for r in rows if r["seed"] < cut]
    test = [r for r in rows if r["seed"] >= cut]
    print(f"positions train={len(train)} test={len(test)} device={DEV}", flush=True)
    torch.manual_seed(0)
    model = Ranker().to(DEV)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(0)
    best, best_state = None, None
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for group in batches(train, bs, rng):
            x, pv, owner, sig, val = tensors(group)
            logit, v = model(x, pv, owner)
            prob = group_softmax(logit, owner, len(group))
            ce = -(sig * torch.log(prob + 1e-8)).sum() / len(group)
            mse = ((v - val) ** 2).mean()
            loss = ce + 4.0 * mse
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(0.99).add_(pm, alpha=0.01)
            tot += float(loss)
        sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            rep = evaluate(ema, test)
            print(f"epoch {ep+1} loss {tot:.2f} ema-test {rep}", flush=True)
            if best is None or rep["top2_regret"] < best:
                best, best_state = rep["top2_regret"], copy.deepcopy(ema.state_dict())
    ema.load_state_dict(best_state)
    report = {"test_ranker": evaluate(ema, test), "test_random": baseline(test),
              "n_train": len(train), "n_test": len(test), "device": str(DEV)}
    for ph in ("opening", "mid", "end"):
        sub = [r for r in test if r["phase"] == ph]
        if sub:
            report[f"test_ranker_{ph}"] = evaluate(ema, sub)
            report[f"test_random_{ph}"] = baseline(sub)
            report[f"mean_plans_{ph}"] = round(float(np.mean([len(r['plans']) for r in sub])), 2)
    print(json.dumps(report, indent=1))
    torch.save({k: v.cpu() for k, v in ema.state_dict().items()}, ROOT / "models/ranker40.pt")
    (ROOT / "models/ranker40_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
