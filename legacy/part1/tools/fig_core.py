"""Figures 1-3: the fog comparison, the branching-factor comparison, and the
annotated game tree. Positions and move counts come from the real simulator."""

from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))  # project root (genmod)
sys.path.insert(0, str(ROOT / "tools"))

from figstyle import (ACCENT, BLUE, BLUE_LT, FILL_DARK, FILL_LIGHT, FILL_MID,  # noqa: E402
                      GREY, HIDDEN, INK, LINE, MONO, RED, RED_LT, SANS, SERIF,
                      Fig, crown, mountain)

from genmod import EnvConfig, reset, step, visibility  # noqa: E402
from genmod.bots import expander_bot, random_bot  # noqa: E402

MEDIA = ROOT / "media"
MEDIA.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
def real_position(size=15, ticks=150, seed=6):
    """Play a real game forward so the figures show a genuine mid-game board."""
    cfg = EnvConfig(height=size, width=size, min_spawn_dist=7, max_spawn_dist=10**9,
                    max_steps=900, bfs_iters=4 * size)
    st = reset(jax.random.PRNGKey(seed), cfg)
    k = jax.random.PRNGKey(seed + 1)
    for _ in range(ticks):
        k, a, b = jax.random.split(k, 3)
        acts = jnp.stack([expander_bot(a, st, cfg, 0), expander_bot(b, st, cfg, 1)])
        st, *_ = step(st, acts, cfg)
        if bool(st.done):
            break
    return cfg, st


def legal_move_count(st, cfg, player):
    from genmod import legal_action_mask
    m = np.asarray(legal_action_mask(st, cfg)[player])
    return int(m[..., 1:].sum())  # exclude the pass action


def draw_board(f: Fig, st, cfg, x0, y0, cell, vis=None, title=None, sub=None):
    """Draw a board. `vis` is an optional (H,W) boolean visibility mask."""
    owner = np.asarray(st.owner)
    army = np.asarray(st.army)
    mtn = np.asarray(st.mountain)
    cas = np.asarray(st.castle)
    gen = np.asarray(st.general)
    H, W = cfg.height, cfg.width

    if title:
        f.text(x0, y0 - 26, title, size=14, weight="600")
    if sub:
        f.text(x0, y0 - 10, sub, size=11.5, fill=GREY)

    for r in range(H):
        for c in range(W):
            x, y = x0 + c * cell, y0 + r * cell
            seen = True if vis is None else bool(vis[r, c])

            if not seen:
                f.rect(x, y, cell, cell, fill=HIDDEN, stroke="#5a5a5a", sw=0.5)
                continue

            if mtn[r, c]:
                f.rect(x, y, cell, cell, fill=FILL_DARK, stroke=LINE, sw=0.5)
                mountain(f, x + cell / 2, y + cell / 2, cell * 0.26, "#5a5a5a")
                continue

            o = int(owner[r, c])
            fill = RED_LT if o == 0 else BLUE_LT if o == 1 else FILL_LIGHT
            f.rect(x, y, cell, cell, fill=fill, stroke=LINE, sw=0.5)

            a = int(army[r, c])
            marked = bool(gen[r, c]) or bool(cas[r, c])

            if gen[r, c]:
                # crown sits in the upper half so the army count stays legible
                crown(f, x + cell / 2, y + cell * 0.30, cell * 0.21,
                      RED if o == 0 else BLUE if o == 1 else GREY)
            elif cas[r, c]:
                f.circle(x + cell / 2, y + cell * 0.32, cell * 0.19,
                         fill="none", stroke=INK if o < 0 else (RED if o == 0 else BLUE), sw=1.1)

            if a:
                ty = y + cell * 0.82 if marked else y + cell / 2 + cell * 0.19
                f.text(x + cell / 2, ty, str(a),
                       size=cell * (0.34 if marked else 0.40), anchor="middle",
                       family=MONO, fill=INK, weight="500")

    f.rect(x0, y0, W * cell, H * cell, fill="none", stroke=INK, sw=1.4)


def legend_row(f, x, y, kind, text, cell=15):
    if kind == "red":
        f.rect(x, y - cell + 3, cell, cell, fill=RED_LT, stroke=LINE, sw=0.6)
    elif kind == "blue":
        f.rect(x, y - cell + 3, cell, cell, fill=BLUE_LT, stroke=LINE, sw=0.6)
    elif kind == "neutral":
        f.rect(x, y - cell + 3, cell, cell, fill=FILL_LIGHT, stroke=LINE, sw=0.6)
    elif kind == "mtn":
        f.rect(x, y - cell + 3, cell, cell, fill=FILL_DARK, stroke=LINE, sw=0.6)
        mountain(f, x + cell / 2, y - cell / 2 + 3, cell * 0.26, "#5a5a5a")
    elif kind == "hidden":
        f.rect(x, y - cell + 3, cell, cell, fill=HIDDEN, stroke="#5a5a5a", sw=0.6)
    elif kind == "gen":
        f.rect(x, y - cell + 3, cell, cell, fill=RED_LT, stroke=LINE, sw=0.6)
        crown(f, x + cell / 2, y - cell / 2 + 3, cell * 0.30, RED)
    elif kind == "castle":
        f.rect(x, y - cell + 3, cell, cell, fill=FILL_LIGHT, stroke=LINE, sw=0.6)
        f.circle(x + cell / 2, y - cell / 2 + 3, cell * 0.28, stroke=INK, sw=1.1)
    f.text(x + cell + 7, y, text, size=11, fill=INK)


# --------------------------------------------------------------------------- #
def fig1_fog():
    cfg, st = real_position()
    cell = 24
    bw = cfg.width * cell
    f = Fig(2 * bw + 150, cfg.height * cell + 168)

    fog_cfg = EnvConfig(**{**cfg.__dict__, "visibility": "fog"})
    vis = np.asarray(visibility(st, fog_cfg))[0]

    x1, x2, y0 = 40, 40 + bw + 70, 78
    draw_board(f, st, cfg, x1, y0, cell,
               title="(a)  Our variant - no fog",
               sub="Red to play. The state is the board.")
    draw_board(f, st, cfg, x2, y0, cell, vis=vis,
               title="(b)  The original game - fog of war",
               sub="What Red can see: own tiles plus their neighbours.")

    hidden = int((~vis).sum())
    total = cfg.height * cfg.width
    f.text(x2, y0 + cfg.height * cell + 22,
           f"{hidden} of {total} tiles hidden ({100*hidden/total:.0f}%)",
           size=11.5, fill=GREY, family=MONO)
    f.text(x1, y0 + cfg.height * cell + 22,
           "0 tiles hidden - minimax is sound here",
           size=11.5, fill=ACCENT, family=MONO)

    ly = y0 + cfg.height * cell + 48
    for i, (k, t) in enumerate([("red", "Red"), ("blue", "Blue"), ("neutral", "neutral"),
                                ("mtn", "mountain"), ("gen", "general"),
                                ("castle", "castle"), ("hidden", "hidden by fog")]):
        legend_row(f, x1 + i * 118, ly, k, t)

    f.caption("1", "The same position with and without fog. Removing fog is not a "
                   "simplification: it replaces a belief state with a state.")
    return f.save(MEDIA / "fig1-fog-vs-nofog.svg")


# --------------------------------------------------------------------------- #
def fan(f, cx, cy, n, spread, depth_y, stroke=LINE, sw=0.6, cap=None, dot=2.0):
    """Draw a fan of n branches from (cx,cy). Returns the child x positions."""
    shown = min(n, cap or n)
    xs = []
    for i in range(shown):
        t = 0.5 if shown == 1 else i / (shown - 1)
        x = cx - spread / 2 + t * spread
        f.line(cx, cy, x, cy + depth_y, stroke=stroke, sw=sw)
        xs.append(x)
    return xs


def fig2_branching():
    cfg, st = real_position()
    b_gen = legal_move_count(st, cfg, 0)
    b_chess = 35

    f = Fig(1180, 706)
    f.text(40, 44, "Branching factor: why depth is expensive here", size=17, weight="600")
    f.text(40, 66, "Every line below is one legal move from the position shown.",
           size=12, fill=GREY)

    # ---- chess side -------------------------------------------------------
    cx = 250
    f.text(40, 106, "Chess", size=14, weight="600")
    f.text(40, 124, f"typical b ≈ {b_chess}", size=12, fill=GREY, family=MONO)

    # small 8x8 board with a few move arrows
    bs, c8 = 26, 8
    bx, by = 40, 140
    for r in range(c8):
        for c in range(c8):
            f.rect(bx + c * bs, by + r * bs, bs, bs,
                   fill=FILL_LIGHT if (r + c) % 2 == 0 else FILL_MID, stroke="none")
    f.rect(bx, by, c8 * bs, c8 * bs, fill="none", stroke=INK, sw=1.3)
    # a knight and its 8 moves, plus a queen ray, to suggest ~35
    kx, ky = bx + 3 * bs + bs / 2, by + 4 * bs + bs / 2
    f.circle(kx, ky, 7, fill=INK, stroke="none")
    for dx, dy in [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]:
        f.circle(kx + dx * bs, ky + dy * bs, 3.2, fill="none", stroke=LINE, sw=1)
    f.text(bx, by + c8 * bs + 18, "one piece shown; ~35 legal in total",
           size=10.5, fill=GREY)

    ty = by + c8 * bs + 44
    f.line(cx, ty, cx, ty + 14, stroke=INK, sw=1.2)
    xs = fan(f, cx, ty + 14, b_chess, 400, 66)
    for x in xs:
        f.circle(x, ty + 80, 1.9, fill=LINE, stroke="none")
    f.text(cx, ty + 106, f"{b_chess} branches", size=12, anchor="middle", family=MONO)
    f.text(cx, ty + 126, "3 ply ≈ 43,000 leaves", size=11.5, anchor="middle",
           family=MONO, fill=GREY)

    # ---- divider ----------------------------------------------------------
    f.line(590, 96, 590, 640, stroke="#cccccc", sw=1, dash="4 4")

    # ---- generals side ----------------------------------------------------
    gx = 880
    f.text(630, 106, "Generals.io (our variant)", size=14, weight="600")
    f.text(630, 124, f"measured b = {b_gen} in this position", size=12,
           fill=GREY, family=MONO)

    cell = 13
    draw_board(f, st, cfg, 630, 140, cell)
    f.text(630, 140 + cfg.height * cell + 18,
           "every owned tile with ≥2 army can move 4 ways, × 2 move types",
           size=10.5, fill=GREY)

    ty2 = 140 + cfg.height * cell + 44
    f.line(gx, ty2, gx, ty2 + 14, stroke=INK, sw=1.2)
    xs2 = fan(f, gx, ty2 + 14, b_gen, 470, 66, sw=0.35)
    f.text(gx, ty2 + 106, f"{b_gen} branches", size=12, anchor="middle", family=MONO)
    f.text(gx, ty2 + 126, f"3 ply ≈ {b_gen**3/1e6:.0f} million leaves",
           size=11.5, anchor="middle", family=MONO, fill=RED)

    # ---- the punchline: how the two scale with depth ---------------------
    bx0, by0 = 40, 566
    f.rect(bx0, by0, 1100, 96, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(bx0 + 16, by0 + 22,
           "Nodes searched by alpha-beta with perfect move ordering, "
           "b^(d/2):", size=12.5, weight="600")

    cols = [bx0 + 16, bx0 + 200, bx0 + 420, bx0 + 660]
    hdr = ["depth", f"chess  (b = {b_chess})", f"ours  (b = {b_gen})", "ratio"]
    for cx_, t in zip(cols, hdr):
        f.text(cx_, by0 + 44, t, size=11, fill=GREY, family=MONO)
    f.line(bx0 + 16, by0 + 50, bx0 + 800, by0 + 50, stroke="#cccccc", sw=0.8)

    for i, d in enumerate((3, 5, 7)):
        y = by0 + 68 + i * 0
        yy = by0 + 66 + i * 0
    rows = []
    for d in (3, 5, 7):
        a_ = b_chess ** (d / 2)
        b_ = b_gen ** (d / 2)
        rows.append((d, a_, b_, b_ / a_))
    for i, (d, a_, b_, rat) in enumerate(rows):
        y = by0 + 66 + i * 0
    # lay the three rows side by side to keep the figure short
    for i, (d, a_, b_, rat) in enumerate(rows):
        x = bx0 + 16 + i * 360
        f.text(x, by0 + 70, f"{d} ply", size=12, family=MONO, weight="600")
        f.text(x + 62, by0 + 70, f"{a_:,.0f}", size=12, family=MONO)
        f.text(x + 150, by0 + 70, f"{b_:,.0f}", size=12, family=MONO,
               fill=RED if d > 3 else INK)
        f.text(x + 262, by0 + 70, f"\u00d7{rat:,.0f}", size=11.5, family=MONO, fill=GREY)

    f.text(bx0 + 16, by0 + 89,
           "Three ply is affordable. Five is not - and each node needs a full "
           "game-state simulation, not just an evaluation.",
           size=12, fill=INK)

    f.caption("2", "Chess versus Generals.io. Sound pruning gets us to three ply "
                   "and no further; depth beyond that needs forward pruning.", y=690)
    return f.save(MEDIA / "fig2-branching-factor.svg")


# --------------------------------------------------------------------------- #
def fig3_tree_regions():
    """Tree centred; the depth axis pairs with ARA* on the left, the rest right."""
    f = Fig(1240, 780)
    f.text(40, 44, "Where each technique acts on the search", size=17, weight="600")
    f.text(40, 66, "One tick of deliberation, drawn as a game tree.", size=12, fill=GREY)

    CX = 640
    L, LW = 40, 250
    R, RW = 960, 250

    def box(x, y, w, h, num, title, lines, colour, bg):
        f.rect(x, y, w, h, fill=bg, stroke=colour, sw=1.3, rx=3)
        f.text(x + 14, y + 21, f"{num}.  {title}", size=11.5, weight="700", fill=colour)
        for i, ln in enumerate(lines):
            f.text(x + 14, y + 40 + i * 16, ln, size=11)

    # ---- the tree ---------------------------------------------------------
    ry = 140
    f.circle(CX, ry, 13, fill="#ffffff", stroke=INK, sw=1.6)
    f.text(CX, ry + 4.5, "s", size=12, anchor="middle", family=SERIF, style="italic")

    y1 = 282
    plans = [CX - 175 + i * 70 for i in range(6)]
    for i in range(260):
        x = CX - 205 + i / 259 * 410
        f.line(CX, ry + 13, x, y1 - 10, stroke="#e4e4e4", sw=0.3)
    f.text(CX - 232, y1 - 44, "~400 raw moves,", size=10.5, fill="#9a9a9a",
           anchor="end", family=MONO)
    f.text(CX - 232, y1 - 30, "discarded", size=10.5, fill="#9a9a9a",
           anchor="end", family=MONO)

    for x, t in zip(plans, ["exp", "gth", "def", "cst", "atk", "cut"]):
        f.line(CX, ry + 13, x, y1 - 10, stroke=INK, sw=1.2)
        f.circle(x, y1, 10, fill="#ffffff", stroke=INK, sw=1.3)
        f.text(x, y1 + 3.4, t, size=8, anchor="middle", family=MONO)
    f.text(CX + 200, y1 + 4, "6 plans kept", size=10.5, fill=INK, family=MONO)

    y2 = 386
    for x in plans:
        for j in range(4):
            xx = x - 21 + j * 14
            f.line(x, y1 + 10, xx, y2 - 6, stroke=LINE, sw=0.6)
            f.rect(xx - 4.5, y2 - 6, 9, 9, fill=FILL_MID, stroke=LINE, sw=0.6)
    f.text(CX, y2 + 22, "each plan \u00d7 4 abstracted opponent intents",
           size=10.5, fill=GREY, anchor="middle")

    # ---- ProbCut on one subtree -------------------------------------------
    y3 = 478
    px = plans[3]
    for j in range(4):
        xx = px - 21 + j * 14
        if j in (1, 2):
            f.line(xx, y2 + 4, xx, y3 - 20, stroke="#cfcfcf", sw=0.7, dash="3 3")
            f.line(xx - 6, y3 - 26, xx + 6, y3 - 14, stroke=RED, sw=1.7)
            f.line(xx - 6, y3 - 14, xx + 6, y3 - 26, stroke=RED, sw=1.7)
        else:
            f.line(xx, y2 + 4, xx, y3 - 6, stroke=LINE, sw=0.6)
            f.rect(xx - 4.5, y3 - 6, 9, 9, fill=FILL_LIGHT, stroke=LINE, sw=0.6)
    f.text(px, y3 + 12, "pruned", size=10, fill=RED, family=MONO, anchor="middle")

    y4 = 530
    f.line(CX - 230, y4, CX + 230, y4, stroke="#cccccc", sw=0.9, dash="4 4")
    f.text(CX + 236, y4 + 4, "evaluate", size=10.5, fill=GREY)

    # ---- depth axis, on the LEFT so it pairs with the ARA* box ------------
    ax = CX - 268
    f.line(ax, ry + 6, ax, y4, stroke="#b0b0b0", sw=1)
    f.arrow(ax, y4 - 26, ax, y4 + 4, stroke="#b0b0b0", sw=1, head=6)
    for yy, lab in ((y1 - 10, "\u03b5 = 3"), (y2 - 6, "\u03b5 = 2"), (y3 - 6, "\u03b5 = 1.5")):
        f.line(ax - 5, yy, ax + 5, yy, stroke="#b0b0b0", sw=1)
        f.text(ax - 10, yy + 4, lab, size=10.5, fill="#909090", family=MONO, anchor="end")
    f.text(ax - 10, y4 + 6, "more time", size=10, fill="#909090",
           style="italic", anchor="end")

    # ---- annotations ------------------------------------------------------
    box(R, 108, RW, 78, "1", "METAREASONING GATE",
        ["Score every move in one batch,", "0.6 ms. If the best is clearly",
         "ahead, play it and stop here."], ACCENT, "#fdf6e3")
    f.arrow(R, 147, CX + 18, ry, stroke=ACCENT, sw=1.2)

    box(R, 250, RW, 78, "2", "CSP NARROWS THE CHOICE",
        ["Which stack does what. Repair", "last tick's assignment.",
         "400 moves \u2192 ~6 coherent plans."], BLUE, "#eef4f8")
    f.arrow(R, 289, CX + 190, 285, stroke=BLUE, sw=1.2)

    box(R, 428, RW, 78, "3", "PROBCUT PRUNES INSIDE",
        ["A cheap shallow search decides", "whether the deep one is worth it.",
         "Unsound, with a risk dial T."], RED, "#fdf1f0")
    f.arrow(R, 467, px + 30, y3 - 18, stroke=RED, sw=1.2)

    box(L, 250, LW, 78, "4", "ARA* BOUNDS THE TIME",
        ["Search with \u03b5 decreasing,", "reusing the previous pass.",
         "Always an answer, always a bound."], "#6a5a8a", "#f2f0f6")
    f.arrow(L + LW, 289, ax - 62, 289, stroke="#6a5a8a", sw=1.2)

    box(CX - 250, 578, 500, 78, "5", "RTAA* IS THE MEMORY",
        ["h(tile) is updated every tick and only ever raised, so it only becomes",
         "better informed. Tiles whose value we had to raise are marked as traps",
         "and the evaluation avoids them."], "#3a6b45", "#eef6ef")
    f.arrow(CX, 574, CX, y4 + 6, stroke="#3a6b45", sw=1.2)

    # ---- the five questions, as a key -------------------------------------
    ky = 706
    for x, t, c in ((L, "whether to search", ACCENT), (L + 165, "what to search", BLUE),
                    (L + 310, "how long", "#6a5a8a"), (L + 415, "how deep", RED),
                    (L + 520, "what to remember", "#3a6b45")):
        f.rect(x, ky - 9, 9, 9, fill=c, stroke="none")
        f.text(x + 15, ky, t, size=11, fill=INK)

    f.caption("3", "The same tree, annotated with where each technique acts.", y=754)
    return f.save(MEDIA / "fig3-tree-regions.svg")


if __name__ == "__main__":
    for fn in (fig1_fog, fig2_branching, fig3_tree_regions):
        p = fn()
        print(f"  {pathlib.Path(p).name}")
