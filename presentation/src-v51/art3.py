import pathlib
"""Charts for the value model and the classifier (dark theme)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).parent
ART = HERE / "art"
ROOT = pathlib.Path(__file__).resolve().parents[2]
BG, INK, INK2, INK3, BRASS, RED, BLUE = "#0B1012", "#E2E9EA", "#9DB0B4", "#6B7D82", "#D7A244", "#D1544A", "#4A86C6"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 12, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK3, "mathtext.fontset": "cm",
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})


def save(fig, name):
    fig.savefig(ART / name, dpi=220, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def clean(ax, left=True):
    for s in ("top", "right") + (() if left else ("left",)):
        ax.spines[s].set_visible(False)


# sigmoid with the worked example
z = np.linspace(-5, 5, 200)
fig, ax = plt.subplots(figsize=(5.2, 2.9))
ax.plot(z, 1 / (1 + np.exp(-z)), color=BRASS, lw=2.4)
ax.axhline(.5, color=INK3, ls=":", lw=.8)
ax.axvline(0, color=INK3, ls=":", lw=.8)
ax.scatter([1.29], [1 / (1 + np.exp(-1.29))], s=60, color=RED, zorder=3)
ax.annotate("z = 1.29 gives 78%", xy=(1.29, .784), xytext=(1.9, .45), fontsize=11, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK3))
ax.set_xlabel("score  z = w·φ(s)")
ax.set_ylabel("chance of winning")
ax.set_ylim(-.03, 1.03)
clean(ax)
save(fig, "sigmoid.png")

# value weights per phase
V = json.loads((ROOT / "models/value37.json").read_text())
names = ["land lead", "army lead", "producers", "income × turns left", "our general − their stack",
         "our stack − their general", "biggest stack lead"]
idx = [1, 2, 3, 4, 5, 6, 7]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
y = np.arange(len(idx))
for k, (ph, col) in enumerate((("opening", INK3), ("mid", BRASS), ("end", RED))):
    w = [V["weights"][ph][i] for i in idx]
    ax.barh(y + (k - 1) * .26, w, .25, color=col, label=ph)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=10.5)
ax.invert_yaxis()
ax.axvline(0, color=INK3, lw=.8)
ax.set_xlabel("weight (change in log-odds per unit)")
ax.legend(frameon=False, fontsize=10, labelcolor=INK, loc="lower right")
clean(ax)
ax.tick_params(axis="y", length=0)
save(fig, "value_weights.png")

# held-out results vs land-only baseline
R = V["report"]
fig, ax = plt.subplots(figsize=(6.2, 3.1))
x = np.arange(3)
m = [R[p]["test_logloss"] for p in ("opening", "mid", "end")]
b = [R[p]["baseline_logloss"] for p in ("opening", "mid", "end")]
ax.bar(x - .19, b, .36, color=INK3, label="land only")
ax.bar(x + .19, m, .36, color=BRASS, label="9 features")
for i in range(3):
    ax.text(i - .19, b[i] + .012, f"{b[i]:.3f}", ha="center", fontsize=10, color=INK2)
    ax.text(i + .19, m[i] + .012, f"{m[i]:.3f}", ha="center", fontsize=10, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"{p}\n({R[p]['n_test']:,} test positions)" for p in ("opening", "mid", "end")], fontsize=10)
ax.set_ylabel("log-loss (lower is better)")
ax.set_ylim(0, .6)
ax.legend(frameon=False, fontsize=10, labelcolor=INK)
clean(ax)
save(fig, "value_results.png")

# classifier accuracy over the game
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ranges = ["0–100", "100–200", "200–300", "300–500"]
acc = [.246, .533, .699, .672]
ax.bar(range(4), acc, .55, color=INK3, label="one board on its own")
ax.plot([0.5, 1.5, 2.5], [.388, .662, .702], "o-", color=BRASS, lw=2, label="belief built over the game")
for i, v in enumerate(acc):
    ax.text(i, v + .02, f"{v:.0%}", ha="center", fontsize=10, color=INK2)
for xx, v in zip([0.5, 1.5, 2.5], [.388, .662, .702]):
    ax.text(xx, v + .045, f"{v:.0%}", ha="center", fontsize=10.5, color=BRASS)
ax.axhline(.1, color=RED, ls="--", lw=.9, label="random guess (10%)")
ax.set_xticks(range(4))
ax.set_xticklabels([f"turns {r}" for r in ranges], fontsize=10)
ax.set_ylim(0, .9)
ax.set_yticks([])
ax.legend(frameon=False, fontsize=10, labelcolor=INK, loc="upper left")
clean(ax, left=False)
save(fig, "cls_acc.png")

# confusion matrix, rows normalised
types = ["rusher", "castle rusher", "land grabber", "turtle", "gather early", "gather mid", "gather late",
         "spread", "economy bot", "planner bot"]
C = np.array([[1586, 34, 0, 2052, 0, 0, 0, 0, 0, 48], [646, 975, 0, 446, 8, 0, 4, 27, 0, 52],
              [0, 0, 2524, 0, 337, 30, 114, 206, 332, 434], [578, 0, 0, 3247, 0, 0, 0, 0, 0, 48],
              [0, 0, 60, 0, 2888, 45, 0, 41, 7, 110], [0, 0, 171, 0, 496, 1737, 54, 177, 206, 267],
              [0, 0, 436, 0, 337, 267, 986, 362, 300, 270], [0, 0, 332, 0, 337, 37, 106, 1142, 266, 290],
              [0, 0, 437, 0, 325, 41, 52, 198, 1851, 959], [0, 0, 569, 0, 334, 73, 25, 139, 1712, 1148]], float)
P = C / C.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.6, 5.6))
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list("b", [BG, "#3A3220", BRASS])
ax.imshow(P, cmap=cm, vmin=0, vmax=.8)
for i in range(10):
    for j in range(10):
        if P[i, j] >= .05:
            ax.text(j, i, f"{P[i, j]*100:.0f}", ha="center", va="center", fontsize=9, color=INK if P[i, j] < .5 else BG)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(types, rotation=55, ha="right", fontsize=9.5)
ax.set_yticklabels(types, fontsize=9.5)
ax.set_xlabel("guessed")
ax.set_ylabel("true style")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
save(fig, "cls_confusion.png")

# MLP diagram
fig, ax = plt.subplots(figsize=(6.4, 3.2))
cols = [(0, 14, "14 measurements"), (1, 64, "64"), (2, 64, "64"), (3, 10, "10 styles")]
shown = {14: 7, 64: 8, 10: 6}
pos = {}
for c, n, lab in cols:
    k = shown[n]
    ys = np.linspace(.9, .1, k)
    pos[c] = ys
    for yy in ys:
        ax.add_patch(plt.Circle((c, yy), .045, color=BRASS if c in (0, 3) else INK3, zorder=3))
    ax.text(c, -.05, lab, ha="center", fontsize=10.5, color=INK2)
    ax.text(c, .97, "⋮" if False else "", ha="center")
for c in range(3):
    for y0 in pos[c]:
        for y1 in pos[c + 1]:
            ax.plot([c, c + 1], [y0, y1], color="#243036", lw=.6, zorder=1)
for c, lab in ((0.5, "ReLU"), (1.5, "ReLU"), (2.5, "softmax")):
    ax.text(c, 1.0, lab, ha="center", fontsize=10, color=BRASS)
ax.set_xlim(-.4, 3.4)
ax.set_ylim(-.12, 1.06)
ax.axis("off")
save(fig, "mlp.png")
print("ok")
