"""Equations (Computer Modern) and figures for the final deck, Graphite Paper theme."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Circle
import numpy as np

HERE = pathlib.Path(__file__).parent
FIG, EQ = HERE / "fig", HERE / "eq"
OLD = HERE.parent / "ppt"
FIG.mkdir(exist_ok=True)
EQ.mkdir(exist_ok=True)

INK, CHAR, SLATE, RULE, PANEL = "#222222", "#36454F", "#708090", "#D3D3D3", "#F4F4F2"
US, THEM = "#B03A2E", "#2C5F8A"
US_T, THEM_T, NEU, MTN = "#E8B4AE", "#AAC4D8", "#F0F0EE", "#C4C4C0"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 11, "axes.edgecolor": SLATE, "axes.linewidth": 0.8,
                     "xtick.color": SLATE, "ytick.color": SLATE, "axes.labelcolor": SLATE,
                     "mathtext.fontset": "cm"})


def save(fig, name):
    fig.savefig(FIG / name, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# equations
# --------------------------------------------------------------------------- #
EQS = {
    "game": r"$\mathcal{G}=(S,\;A_0,\;A_1,\;T,\;U),\qquad s_{t+1}=T(s_t,\,a^0_t,\,a^1_t),\qquad U\in\{-1,+1\}$",
    "branch": r"$b\;\approx\;8\,\left|\{c:\ \mathrm{owner}(c)=p,\ \mathrm{army}(c)\geq 2\}\right|+1,\qquad b_{\mathrm{joint}}=b_0\,b_1$",
    "wastar": r"$f(n)=g(n)+\varepsilon\,h(n),\qquad \mathrm{cost}(\pi_\varepsilon)\;\leq\;\varepsilon\,C^{*}$",
    "rtaa": r"$h(s)\;\leftarrow\;g(\bar s)+h(\bar s)-g(s)\qquad \forall s\in\mathrm{CLOSED}$",
    "depr": r"$\mathrm{prefer}\ s\ \mathrm{with}\ h(s)=h_0(s)\quad(\mathrm{not\ yet\ raised})$",
    "probcut": r"$v\approx a\,v'+b+e,\ \ e\sim\mathcal{N}(0,\sigma^2);\qquad \mathrm{cut\ if}\ \ v'\geq\frac{\beta+t\sigma-b}{a}$",
    "voc": r"$\mathrm{VOC}(c)=\mathbb{E}\left[U(\alpha_c)\right]-U(\alpha)-\mathrm{cost}(c)$",
    "maximin": r"$v^{*}=\max_{x\in\Delta_m}\ \min_{j}\ \sum_i x_i\,M_{ij}$",
    "abcd": r"$\max_a\min_b V(a,b)\;\leq\;v^{*}\;\leq\;\min_b\max_a V(a,b)$",
    "walk": r"$x_{\mathrm{arrive}}=x-d-\sum_{c\in\pi}\mathrm{garrison}(c),\qquad G(t+d)=G(t)+\lfloor d/2\rfloor$",
    "invariant": r"$m=\min_k\left[\,G+\mathrm{growth}(d_k)-x_k\,\right]\;>\;\mathrm{pad},\qquad \mathrm{pad}=3+0.08\,x_{\max}$",
    "dxt": r"$D_p(X,T)=\max_{W}\ \sum_{w\in W}\mathrm{force}(w)\quad\mathrm{s.t.}\quad \sum_{w\in W}|w|\leq T\quad(\mathrm{each\ tile\ once})$",
    "race": r"$m=\min_{T}\left[\,G+\mathrm{growth}(T)+D_{\mathrm{us}}\!\left(g,\,T-2-\lfloor T/10\rfloor\right)-D_{\mathrm{them}}(g,T)\,\right]$",
    "reserve": r"$G_{\mathrm{keep}}+\mathrm{growth}(H+d)\;\geq\;R\cdot x(H),\qquad H\in\{0,30,60\},\ \ R\in\{0.4,\,0.6\}$",
    "payback": r"$2\,(\mathrm{garrison}+d)\;\leq\;\min(L-d,\;400)$",
    "hurwicz": r"$S(a)=\frac{1}{2}\sum_b P(b)\,V(a,b)+\frac{1}{2}\min_b V(a,b)$",
    "logistic": r"$V(s)=\sigma\!\left(w_{\phi}^{\top}\,\phi(s)\right),\qquad \phi(s)\in\mathbb{R}^9$",
    "posterior": r"$w_b\;\leftarrow\;0.95\,w_b\,\frac{L_b}{0.5}+0.05,\qquad L_b\in\{0.8,\,0.2\}$",
    "exp3": r"$p_i=(1-\gamma)\frac{w_i}{\sum_j w_j}+\frac{\gamma}{K},\qquad w_i\leftarrow w_i\,e^{\gamma\,r/(p_i K)}$",
    "ses": r"$a^{\dagger}=\arg\max_a \sum_b P(b)V(a,b)\quad \mathrm{s.t.}\quad \min_b V(a,b)\geq v^{*}-k$",
    "income": r"$\Delta=(A_{\mathrm{us}}-A_{\mathrm{them}})+(\iota_{\mathrm{us}}-\iota_{\mathrm{them}})\,L,\qquad \iota=\frac{1}{2}\,\mathrm{producers}+\frac{\mathrm{land}}{50}$",
    "approach": r"$\mathrm{incoming}(s)\Leftrightarrow d_{t-k}(s)-d_t(s)\geq 2,\ \ k\leq 5,\ \ x_s\geq\max(15,\,0.2\,G)$",
    "need": r"$\mathrm{need}=\max_{T\leq \mathrm{ETA}+5}\left[D_{\mathrm{them}}(g,T)-G-\mathrm{growth}(T)\right]+0.15\,x_{\mathrm{att}}$",
    "pcst": r"$\max_{\mathcal{T}\ni g}\ \sum_{v\in\mathcal{T}}(a_v-1)\;-\;\lambda\,|E(\mathcal{T})|,\qquad \lambda^{*}=\min\{\lambda:\ |E(\mathcal{T}_\lambda)|\leq \mathrm{ETA}-2\}$",
    "mlp": r"$P(y\mid x)=\mathrm{softmax}\!\left(W_3\,\mathrm{ReLU}(W_2\,\mathrm{ReLU}(W_1\tilde x+b_1)+b_2)+b_3\right),\ \ \tilde x=\frac{x-\mu}{\sigma}$",
    "online": r"$\ell_t(y)=0.95\,\ell_{t-1}(y)+0.4\,\log P(y\mid x_t),\qquad \hat y_t=\arg\max_y\ \ell_t(y)$",
}


def equations():
    for name, tex in EQS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=22, color=CHAR)
        fig.savefig(EQ / f"{name}.png", dpi=250, transparent=True, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def tile(ax, x, y, kind, num=None, crown=None, fs=12):
    col = {"us": US_T, "them": THEM_T, "neu": NEU, "mtn": MTN}[kind]
    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=col, edgecolor="#8A8A8A", lw=0.6))
    if kind == "mtn":
        ax.add_patch(Polygon([(x + .28, y + .3), (x + .72, y + .3), (x + .5, y + .68)], closed=True, fill=False,
                             edgecolor="#555555", lw=0.9))
    if crown:
        c = US if crown == "us" else THEM
        cx, cy = x + .5, y + .74
        ax.add_patch(Polygon([(cx - .2, cy - .08), (cx + .2, cy - .08), (cx + .2, cy + .1), (cx + .1, cy + .02),
                              (cx, cy + .14), (cx - .1, cy + .02), (cx - .2, cy + .1)], closed=True, facecolor=c,
                             edgecolor="none"))
    if num is not None:
        ax.text(x + .5, y + (.36 if crown else .5), str(num), ha="center", va="center", fontsize=fs,
                family="Menlo", color=INK)


def timeline():
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0.2, 52.8)
    ax.set_ylim(-1.9, 1.9)
    ax.axis("off")
    eras = [(1, 11, "the five papers"), (12, 26, "deeper, learned search"), (27, 35, "more search"),
            (36, 41, "search from the rules"), (42, 46, "collection race, opponent"), (47, 52, "papers' algorithms, v51")]
    for i, (a, b, name) in enumerate(eras):
        ax.add_patch(Rectangle((a - .5, -.35), b - a + 1, .7, facecolor=PANEL if i % 2 == 0 else "#E9E9E6", edgecolor="none"))
        ax.text((a + b) / 2, 1.0 if i % 2 == 0 else -1.15, name, ha="center", fontsize=10.5, color=CHAR)
    prom = {11: "v11", 37: "v37", 42: "v42", 46: "v46", 51: "v51"}
    for v in range(1, 53):
        if v in prom:
            ax.scatter([v], [0], s=95, color=US, zorder=3)
            ax.text(v, .5 if v not in (37, 42) else -.7, prom[v], ha="center", fontsize=10, color=US, weight="bold")
        else:
            ax.scatter([v], [0], s=26, facecolor="white", edgecolor=SLATE, lw=0.8, zorder=3)
    save(fig, "timeline.png")


def invariant():
    fig, ax = plt.subplots(figsize=(9, 2.3))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(-0.7, 2.6)
    ax.axis("off")
    ax.set_aspect("equal")
    for i in range(10):
        kind = "them" if i == 0 else "us" if i == 9 else "neu"
        tile(ax, i + .2, .4, kind, 30 if i == 0 else 12 if i == 9 else None, crown="us" if i == 9 else None)
        if 1 <= i <= 8:
            ax.text(i + .7, .15, str(i), ha="center", va="top", fontsize=9, color=SLATE)
    ax.add_patch(FancyArrowPatch((.9, 1.75), (9.3, 1.75), arrowstyle="-|>", mutation_scale=14, color=THEM, lw=1.4))
    ax.text(5.1, 1.95, "d = 8 steps, one unit left on each tile", ha="center", fontsize=10, color=THEM)
    ax.text(10.7, 1.55, r"$30-8-1=21$", fontsize=15, color=THEM)
    ax.text(10.7, 0.75, r"$12+\lfloor 8/2\rfloor=16$", fontsize=15, color=US)
    ax.text(10.7, -0.05, "margin = 16 - 21 < pad", fontsize=10, color=CHAR)
    save(fig, "invariant.png")


def collect():
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.axis("off")
    ax.set_aspect("equal")
    W, H = 9, 6
    ax.set_xlim(-.1, W + .1)
    ax.set_ylim(-.1, H + .1)
    stacks = {(4, 5): 30, (3, 6): 25, (0, 7): 22}
    for r in range(H):
        for c in range(W):
            kind = "them" if c >= 5 else ("us" if (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)] else "neu")
            if (r, c) in [(1, 4), (3, 3), (2, 3)]:
                kind = "mtn"
            n = stacks.get((r, c), 1 if kind in ("us", "them") else None)
            if (r, c) == (0, 0):
                n = 40
            tile(ax, c, r, kind, n, crown="us" if (r, c) == (0, 0) else None, fs=11)
    for (r, c) in stacks:
        ax.add_patch(FancyArrowPatch((c + .5, r + .5), (0.9, 0.5), arrowstyle="-|>", mutation_scale=12, color=THEM,
                                     lw=1.3, alpha=.8, connectionstyle="arc3,rad=0.15"))
    save(fig, "collect.png")


def race():
    d = json.loads((OLD / "data46.json").read_text())["spread"]["profile"]
    T = [p["T"] for p in d]
    home = [p["G"] + p["us"] for p in d]
    them = [p["them"] for p in d]
    m = [p["m"] for p in d]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.step(T, home, where="post", color=US, lw=1.8, label="our general + growth + walks home")
    ax.step(T, them, where="post", color=THEM, lw=1.8, label="what they can collect onto it")
    ax.fill_between(T, them, home, step="post", color=PANEL)
    i = int(np.argmin(m))
    ax.annotate(f"min = {m[i]} at T = {T[i]}", xy=(T[i] + .5, (them[i] + home[i]) / 2), xytext=(T[i] - 16, 52),
                fontsize=10, color=CHAR, arrowprops=dict(arrowstyle="-", color=SLATE, lw=.8))
    ax.set_xlabel("horizon T (ticks)")
    ax.set_ylabel("army at our general")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    clean(ax)
    save(fig, "race.png")


def human():
    hp = pathlib.Path(__file__).resolve().parents[2] / "models" / "tournaments" / "human_games"
    d = np.load(hp / "20261103_121122_seed538794140_human0.npz", allow_pickle=True)
    o, a, cas = d["owner"], d["army"], d["castle"]
    t = np.arange(o.shape[0] - 2)
    bot = [a[i][o[i] == 1].max() for i in t]
    hum = [a[i][o[i] == 0].max() for i in t]
    hc = [((o[i] == 0) & cas[i].astype(bool)).sum() for i in t]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.6, 4.3), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax.plot(t, bot, color=THEM, lw=1.6, label="v37's general")
    ax.plot(t, hum, color=US, lw=1.6, label="human's largest stack")
    ax.set_ylabel("army")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax2.step(t, hc, color=US, lw=1.6, where="post")
    ax2.set_ylabel("human castles")
    ax2.set_xlabel("tick")
    for x in (ax, ax2):
        clean(x)
    save(fig, "human.png")


def posterior46():
    rows = json.loads((OLD / "data46.json").read_text())["gather120"]["rows"]
    t = [r["t"] for r in rows]
    names = ["expander", "hold", "rusher", "castle_rusher"]
    cols = ["#9A9A96", "#D6D6D2", THEM, "#86A6C4"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.stackplot(t, [[r["w"][n] for r in rows] for n in names], colors=cols, labels=[n.replace("_", " ") for n in names])
    ax.axvline(120, color=CHAR, ls="--", lw=.8)
    ax.text(124, .04, "sweep starts", fontsize=9.5, color=CHAR)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("tick")
    ax.set_ylabel("P(type)")
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, .5))
    clean(ax)
    save(fig, "posterior46.png")


def trace_pair():
    d = json.loads((HERE / "trace51.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, name in zip(axes, ("v46", "v51")):
        rows = [r for r in d[name]["rows"] if 190 <= r["t"] <= 245]
        t = [r["t"] for r in rows]
        ax.plot(t, [r["G"] for r in rows], color=US, lw=1.8, label="our general")
        ax.plot(t, [r["att"] for r in rows], color=THEM, lw=1.8, label="their largest stack")
        if name == "v51":
            inc = [r["t"] for r in rows if r.get("inc")]
            if inc:
                ax.axvspan(min(inc), max(inc), color=PANEL, zorder=0)
                ax.text(min(inc) + 1, 168, "approach detected:\nevery move gathers home", fontsize=9, color=CHAR)
        end = "general captured, tick 237" if name == "v46" else "attack absorbed; v51 wins at 408"
        ax.set_title(f"{name}: {end}", fontsize=11, color=CHAR, loc="left")
        ax.set_xlabel("tick")
        ax.set_ylim(0, 190)
        clean(ax)
    axes[0].set_ylabel("army")
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    save(fig, "trace_pair.png")


def typep():
    rows = json.loads((HERE / "trace51.json").read_text())["v51"]["rows"]
    t = [r["t"] for r in rows]
    show = ["expander", "planner", "econ", "gather_late", "spread", "gather_mid"]
    cols = ["#9A9A96", "#C4C4C0", "#E2E2DE", US, "#D98880", THEM]
    other = [1 - sum(r["tp"][k] for k in show) for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.stackplot(t, [[r["tp"][k] for r in rows] for k in show] + [other], colors=cols + ["#F1F1EE"],
                 labels=[k.replace("_", " ") for k in show] + ["other 4"])
    ax.axvline(200, color=CHAR, ls="--", lw=.8)
    ax.text(204, .04, "sweep at 200", fontsize=9.5, color=CHAR)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tick")
    ax.set_ylabel("posterior")
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, .5))
    clean(ax)
    save(fig, "typep.png")


def confusion():
    types = ["rusher", "castle rusher", "expander", "turtle", "gather early", "gather mid", "gather late", "spread",
             "econ", "planner"]
    C = np.array([[1586, 34, 0, 2052, 0, 0, 0, 0, 0, 48], [646, 975, 0, 446, 8, 0, 4, 27, 0, 52],
                  [0, 0, 2524, 0, 337, 30, 114, 206, 332, 434], [578, 0, 0, 3247, 0, 0, 0, 0, 0, 48],
                  [0, 0, 60, 0, 2888, 45, 0, 41, 7, 110], [0, 0, 171, 0, 496, 1737, 54, 177, 206, 267],
                  [0, 0, 436, 0, 337, 267, 986, 362, 300, 270], [0, 0, 332, 0, 337, 37, 106, 1142, 266, 290],
                  [0, 0, 437, 0, 325, 41, 52, 198, 1851, 959], [0, 0, 569, 0, 334, 73, 25, 139, 1712, 1148]], float)
    R = C / C.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(R, cmap="Greys", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if R[i, j] >= 0.05:
                ax.text(j, i, f"{R[i, j]:.2f}"[1:], ha="center", va="center", fontsize=8,
                        color="white" if R[i, j] > .5 else INK)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(types, rotation=55, ha="right", fontsize=9)
    ax.set_yticklabels(types, fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "confusion.png")


def pcst():
    """Illustration (not game data): a PCST tree rooted at the general, leaf-first order."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-.2, 9.2)
    ax.set_ylim(-1.2, 6.2)
    own = {(r, c) for r in range(6) for c in range(9)}
    armies = {(0, 0): 70, (0, 2): 6, (1, 4): 9, (2, 2): 12, (3, 5): 15, (4, 1): 8, (5, 6): 3, (2, 7): 2, (4, 7): 5}
    tree = {(0, 1): (0, 0), (0, 2): (0, 1), (1, 2): (0, 2), (2, 2): (1, 2), (1, 3): (1, 2), (1, 4): (1, 3),
            (2, 4): (1, 4), (3, 4): (2, 4), (3, 5): (3, 4), (1, 1): (0, 1), (2, 1): (1, 1), (3, 1): (2, 1), (4, 1): (3, 1)}
    for (r, c) in own:
        a = armies.get((r, c), 1)
        tile(ax, c, r, "us", a, crown="us" if (r, c) == (0, 0) else None, fs=10)
    for ch, par in tree.items():
        ax.add_patch(FancyArrowPatch((ch[1] + .5, ch[0] + .5), (par[1] + .5, par[0] + .5), arrowstyle="-|>",
                                     mutation_scale=10, color=CHAR, lw=1.3))
    ax.text(0, -0.55, "13 tree edges = 13 moves, played leaf-first into the general. The 3, 2 and 5 on the right are left out:",
            fontsize=9, color=SLATE)
    ax.text(0, -0.95, "at this lambda the moves to reach them cost more than the army they would add.", fontsize=9, color=SLATE)
    save(fig, "pcst.png")


def layers():
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    rows = [("simulator (sim36)", "exact replica of the game step, used for every rollout", "#EDEDEA", INK),
            ("safety (v36)", "invariant, survives(), verify(), swap-safe", "#EDEDEA", INK),
            ("plans (v37)", "6 macros, rollouts vs 4 scripts, Hurwicz score", "#EDEDEA", INK),
            ("collection race (v42, v44)", "D(X,T) margin, reserve, castles, counter", "#E3E3DF", INK),
            ("opponent gates (v45, v46)", "income race, posterior, Exp3 arm", "#D6D6D2", INK),
            ("fast defence + classifier (v51)", "approach detector, PCST gather, 10-type MLP", CHAR, "white")]
    for i, (h, s, fc, tc) in enumerate(rows):
        y = .1 + i * 1.18
        ax.add_patch(Rectangle((.1, y), 9.8, 1.0, facecolor=fc, edgecolor="none"))
        ax.text(.35, y + .64, h, fontsize=12, weight="bold", color=tc, va="center")
        ax.text(.35, y + .28, s, fontsize=10, color=tc if tc == "white" else SLATE, va="center")
    save(fig, "layers.png")


for f in (equations, timeline, invariant, collect, race, human, posterior46, trace_pair, typep, confusion, pcst, layers):
    f()
    print("ok", f.__name__)
