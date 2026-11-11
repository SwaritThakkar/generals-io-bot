"""Dark-theme charts, equations and a real plan table for the v51 talk."""
import json
import pathlib
import pickle
import sys
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
OUT = HERE / "art"
OUT.mkdir(exist_ok=True)
BG, PANEL, LINE = "#0B1012", "#121A1D", "#243036"
INK, INK2, INK3, BRASS = "#E2E9EA", "#9DB0B4", "#6B7D82", "#D7A244"
RED, BLUE = "#D1544A", "#4A86C6"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 12, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK3, "mathtext.fontset": "cm",
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
D = pickle.load(open(HERE / "game.pkl", "rb"))
F = {f["t"]: f for f in D["frames"]}
NICE = {"rusher": "rusher", "castle_rusher": "castle rusher", "expander": "land grabber", "turtle": "turtle",
        "gather_early": "builds up, attacks early", "gather_mid": "builds up, attacks mid-game",
        "gather_late": "builds up, then attacks", "spread": "spreads & collects", "econ": "economy bot",
        "planner": "planner bot"}


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def belief_bars(t, name):
    tp = sorted(F[t]["tp"].items(), key=lambda kv: -kv[1])[:4][::-1]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    cols = [BRASS if k == "gather_late" else INK3 for k, _ in tp]
    ax.barh([NICE[k] for k, _ in tp], [v for _, v in tp], color=cols, height=.6)
    for i, (_, v) in enumerate(tp):
        ax.text(v + .02, i, f"{v:.0%}", va="center", fontsize=11, color=INK)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    save(fig, name)


def belief_over_game():
    ts = sorted(F)
    keys = ["expander", "planner", "econ", "gather_late"]
    cols = [INK3, "#4B5C61", "#2F3B40", BRASS]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ys = [[F[t]["tp"][k] for t in ts] for k in keys]
    other = [1 - sum(F[t]["tp"][k] for k in keys) for t in ts]
    ax.stackplot(ts, ys + [other], colors=cols + ["#1B2427"], labels=[NICE[k] for k in keys] + ["other 6 styles"])
    ax.axvline(200, color=INK2, ls="--", lw=.8)
    ax.text(203, .05, "opponent starts collecting", fontsize=10, color=INK2)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tick")
    ax.set_ylabel("belief")
    ax.legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.0, .5), labelcolor=INK)
    clean(ax)
    save(fig, "belief_game.png")


def safety_chart():
    ts = [t for t in sorted(F) if 120 <= t <= 300 and F[t]["m0"] is not None]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.plot(ts, [F[t]["m0"] for t in ts], color=BRASS, lw=2)
    ax.axvspan(217, 236, color="#1B2427")
    ax.text(219, 58, "fast defence", fontsize=10, color=INK2)
    ax.annotate("enemy empties its\ngeneral (tick 213)", xy=(213, 12), xytext=(150, 52), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK3))
    ax.set_xlabel("tick")
    ax.set_ylabel("safety score")
    ax.set_ylim(0, 70)
    clean(ax)
    save(fig, "safety_game.png")


def army_chart():
    ts = sorted(F)
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.plot(ts, [F[t]["totals"][1][0] for t in ts], color=RED, lw=2, label="v51")
    ax.plot(ts, [F[t]["totals"][1][1] for t in ts], color=BLUE, lw=2, label="opponent")
    for x, lab in ((213, "sweep"), (237, "attack absorbed"), (298, "first castle"), (407, "capture")):
        ax.axvline(x, color=INK3, ls=":", lw=.9)
        ax.text(x + 3, ax.get_ylim()[1] * .92 if x != 237 else ax.get_ylim()[1] * .78, lab, fontsize=9.5, color=INK2)
    ax.set_xlabel("tick")
    ax.set_ylabel("total army")
    ax.legend(frameon=False, fontsize=10, labelcolor=INK, loc="upper left")
    clean(ax)
    save(fig, "army_game.png")


def results():
    rows = [("gather-rusher\n(sweep at 200)", 19, 12), ("gather-rusher\n(sweep at 120)", 20, 18),
            ("gather-rusher\n(sweep at 60)", 19, 16), ("spread &\ncollect", 13, 14), ("rushers\n(40 games)", 20, None)]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(rows))
    v51 = [r[1] / 20 for r in rows]
    v46 = [r[2] / 20 if r[2] is not None else 0 for r in rows]
    ax.bar(x - .2, v51, .38, color=RED, label="v51")
    ax.bar(x + .2, v46, .38, color=INK3, label="v46 (previous)")
    for i, r in enumerate(rows):
        ax.text(i - .2, v51[i] + .02, f"{v51[i]:.0%}", ha="center", fontsize=10, color=INK)
        if r[2] is not None:
            ax.text(i + .2, v46[i] + .02, f"{v46[i]:.0%}", ha="center", fontsize=10, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=10, labelcolor=INK, loc="upper left", ncol=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "results.png")


EQS = {
    "belief": r"$\ell_t(y)\;=\;0.95\,\ell_{t-1}(y)\;+\;0.4\,\log P(y\mid x_t)$",
    "safety": r"$m\;=\;\min_T\;\left[\,G+\mathrm{growth}(T)+D_{\mathrm{us}}(T-\mathrm{lead})\;-\;D_{\mathrm{them}}(T)\,\right]$",
    "incoming": r"$d_{t-5}(s)-d_t(s)\;\geq\;2\qquad\mathrm{and}\qquad \mathrm{army}(s)\;\geq\;\max(15,\;0.2\,G)$",
    "tree": r"$\mathrm{maximise}\quad \sum_{v\,\in\,\mathrm{tree}}(a_v-1)\;-\;\lambda\cdot\#\mathrm{moves}\qquad\mathrm{with}\quad \#\mathrm{moves}\leq\mathrm{ETA}-2$",
    "plan": r"$\mathrm{score}(a)=\frac{1}{2}\,\sum_b P(b)\,V(a,b)\;+\;\frac{1}{2}\,\min_b V(a,b)$",
}


def equations():
    for k, tex in EQS.items():
        fig = plt.figure(figsize=(.01, .01))
        fig.text(0, 0, tex, fontsize=24, color=INK)
        fig.savefig(OUT / f"eq_{k}.png", dpi=250, transparent=True, bbox_inches="tight", pad_inches=.05)
        plt.close(fig)


def plan_table():
    """Replay the same game to tick 240 and score v51's plans there, as the planner does."""
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(ROOT), str(ROOT / "models"), str(ROOT / "models/tools36")]
    s = importlib.util.spec_from_file_location("v51pt", ROOT / "models/v51_fast_defence.py")
    v51 = importlib.util.module_from_spec(s)
    s.loader.exec_module(v51)
    import adversaries38
    import core37
    from arena36 import start
    ctx, sim = start(93003, 600)
    me = v51.Planner51(ctx, 0)
    op = adversaries38.GatherRusher(ctx, 1, t0=200)
    while sim.step < 240:
        a = me.act(sim)
        b = op.act(sim)
        sim.step2(a or 0, b if b and b > 0 else 0)
    _v39 = v51._v42._v39
    _v37 = v51._v42._v37
    thr, m0 = me.econ.margin_of(sim)
    phase = core37.detect_phase(ctx, sim, 0, m0)
    cands = _v39.plan_set39(ctx, sim, 0, phase, m0, me.econ.pad)
    models = core37.opponent_models(ctx, 1)
    hz = core37.HORIZON[phase]
    tot = sum(me.weights.values())
    rows = []
    for name, fac in cands:
        vals = {o: _v37._VALUE(core37.rollout(sim, 0, fac, of, hz), 0, phase) for o, of in models.items()}
        mean = sum(me.weights[o] * v for o, v in vals.items()) / tot
        rows.append({"plan": name, "vals": vals, "mean": mean, "worst": min(vals.values()),
                     "score": .5 * mean + .5 * min(vals.values())})
    json.dump({"tick": 240, "phase": phase, "horizon": hz, "weights": {k: v / tot for k, v in me.weights.items()},
               "rows": rows}, open(OUT / "plan_table.json", "w"), indent=1)
    for r in rows:
        print(r["plan"], {k: round(v, 2) for k, v in r["vals"].items()}, round(r["score"], 3))
    print("weights", {k: round(v / tot, 2) for k, v in me.weights.items()})


for f in (lambda: belief_bars(205, "belief205.png"), lambda: belief_bars(215, "belief215.png"), belief_over_game,
          safety_chart, army_chart, results, equations, plan_table):
    f()
print("ok")
