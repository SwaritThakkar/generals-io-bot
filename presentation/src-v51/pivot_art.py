import pathlib
import json, os, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).parent
ART, SH = HERE / "art", HERE / "shots"
ROOT = pathlib.Path(__file__).resolve().parents[2]
BG, INK, INK2, INK3, BRASS, RED, BLUE = "#0B1012", "#E2E9EA", "#9DB0B4", "#6B7D82", "#D7A244", "#D1544A", "#4A86C6"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 12, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK3, "mathtext.fontset": "cm",
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
def save(fig, name):
    fig.savefig(ART / name, dpi=220, bbox_inches="tight", pad_inches=.08); plt.close(fig)
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# 1. ProbCut: shallow vs deep, real pairs
pairs = np.array(json.loads((ROOT / "models/probcut_charge_fit_v19.json").read_text())["pairs"])
x, y = pairs[:, 0], pairs[:, 1]
a, b = np.polyfit(x, y, 1)
r = np.corrcoef(x, y)[0, 1]
sd = np.std(y - (a * x + b))
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.scatter(x, y, s=14, color=BRASS, alpha=.8, edgecolor="none")
xx = np.linspace(x.min(), x.max(), 50)
ax.plot(xx, a * xx + b, color=INK, lw=1.4)
ax.fill_between(xx, a * xx + b - 2 * sd, a * xx + b + 2 * sd, color="#1B2427")
ax.set_xlabel("shallow search value  v'")
ax.set_ylabel("deep search value  v")
ax.text(.04, .92, f"r = {r:.3f}   ({len(x)} positions)", transform=ax.transAxes, fontsize=12, color=INK)
clean(ax); save(fig, "probcut_scatter.png")
print("probcut", len(x), round(r, 3), round(a, 3), round(b, 2), round(sd, 2))

# 2. paper-era bots vs the champion v11
rows = [("v12\nProbCut", .5), ("v14\nProbCut", .5), ("v16\nProbCut\n1.5σ", .5), ("v17\n4-ply", .5), ("v18\nqueue,\nforecasts", 0),
        ("v20\nlearned\nProbCut", 1 / 8), ("v21", .5), ("v23\nmacro\nscripts", .5), ("v36\nsafety\nrule", 1.0)]
fig, ax = plt.subplots(figsize=(8.4, 3.6))
cols = [INK3] * 8 + [RED]
ax.bar(range(len(rows)), [r_[1] for r_ in rows], color=cols, width=.62)
ax.axhline(.5, color=INK2, ls="--", lw=.9)
ax.text(len(rows) - .45, .52, "a coin flip", fontsize=10, color=INK2, ha="right")
for i, (_, v) in enumerate(rows):
    ax.text(i, v + .03, f"{v:.0%}", ha="center", fontsize=10, color=INK)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r_[0] for r_ in rows], fontsize=9.5)
ax.set_yticks([]); ax.set_ylim(0, 1.15)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
save(fig, "paper_era.png")

# 3. time scale: log bars
fig, ax = plt.subplots(figsize=(6.6, 2.6))
items = [("time allowed per move", 500, INK3), ("v51 full decision", 10, BRASS), ("one exact distance map", .042, RED)]
ax.barh([i[0] for i in items][::-1], [i[1] for i in items][::-1], color=[i[2] for i in items][::-1], height=.55)
ax.set_xscale("log"); ax.set_xlim(.01, 3000)
for i, (lab, v, c) in enumerate(items[::-1]):
    ax.text(v * 1.4, i, f"{v:g} ms", va="center", fontsize=11, color=INK)
ax.tick_params(axis="y", length=0, labelsize=11)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_xlabel("milliseconds (log scale)")
save(fig, "time_scale.png")

# 4. rusher chart
rows = [("v27\nlive\nmatrix", 4/6), ("v32\nlearned\npruning", 5/6), ("v33\ntime-\naware A*", .5), ("v35\npayback\nA*", .5), ("v33\nrollouts", .25), ("v34\nlonger\nrollouts", .25), ("v36\nsafety\nrule", 1.0)]
fig, ax = plt.subplots(figsize=(7.4, 3.5))
ax.bar(range(len(rows)), [v for _, v in rows], color=[INK3] * 6 + [RED], width=.6)
for i, (_, v) in enumerate(rows): ax.text(i, v + .03, f"{v:.0%}", ha="center", fontsize=10.5, color=INK)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r_[0] for r_ in rows], fontsize=9.5)
ax.set_yticks([]); ax.set_ylim(0, 1.15)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
save(fig, "rusher_dark.png")

# 5. v46 vs v51 trace pair, dark
d = json.loads((HERE.parent / "src" / "trace51.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for axx, name in zip(axes, ("v46", "v51")):
    rr = [r_ for r_ in d[name]["rows"] if 190 <= r_["t"] <= 245]
    t = [r_["t"] for r_ in rr]
    axx.plot(t, [r_["G"] for r_ in rr], color=RED, lw=2, label="our general")
    axx.plot(t, [r_["att"] for r_ in rr], color=BLUE, lw=2, label="their biggest stack")
    if name == "v51":
        inc = [r_["t"] for r_ in rr if r_.get("inc")]
        axx.axvspan(min(inc), max(inc), color="#1B2427", zorder=0)
    axx.set_title("v46: general captured at turn 237" if name == "v46" else "v51: attack absorbed, wins later", fontsize=11.5, color=INK, loc="left")
    axx.set_xlabel("turn"); axx.set_ylim(0, 190); clean(axx)
axes[0].legend(frameon=False, fontsize=10, labelcolor=INK, loc="upper left")
save(fig, "trace_pair_dark.png")

# 6. human game frames (human = red, seat 0; v37 = blue)
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, str(ROOT))
import pygame
from arena.board_draw import BoardPainter, BoardView, BG as GBG, BRASS as GBR
pygame.init()
h = np.load(ROOT / "models/tournaments/human_games/20261103_121122_seed538794140_human0.npz", allow_pickle=True)
own0 = h["owner"][0]
gens = [int(np.where(own0 == q)[0][0]) for q in (0, 1)]
for t in (520, 845, 884):
    owner, army = h["owner"][t], h["army"][t]
    gen = np.zeros(529, bool)
    for g in gens:
        if owner[g] >= 0: gen[g] = True
    mv = []
    for p, act in enumerate(h["moves"][t]):
        act = int(act)
        if act > 0:
            c, k = divmod(act, 9); mv.append((c, (k - 1) % 4, k >= 5, p))
    v = BoardView(23, 23, owner, army, h["mountain"].reshape(-1).astype(bool), h["castle"][t].astype(bool), gen, last_moves=mv)
    surf = pygame.Surface((1100, 1100)); surf.fill(GBG)
    bp = BoardPainter(23, 23); bp.layout(pygame.Rect(0, 0, 1100, 1100)); bp.draw(surf, v)
    big = max(range(529), key=lambda c: army[c] if owner[c] == 0 else -1)
    pygame.draw.rect(surf, GBR, bp.cell_rect(big).inflate(8, 8), 5)
    if owner[gens[1]] >= 0: pygame.draw.rect(surf, GBR, bp.cell_rect(gens[1]).inflate(8, 8), 5)
    pygame.image.save(surf, str(SH / f"human_{t}.png"))
    cas = [int(((owner == q) & h["castle"][t].astype(bool)).sum()) for q in (0, 1)]
    print("human tick", t, "human biggest", army[big], "v37 general", army[gens[1]], "castles", cas)
