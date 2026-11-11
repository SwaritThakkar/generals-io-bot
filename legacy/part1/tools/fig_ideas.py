"""Figures 4-12: one per idea in ideas.txt. Data-driven where possible."""

from __future__ import annotations

import pathlib
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))  # project root (genmod); sys.path.insert(0, str(ROOT / "tools"))

from figstyle import (ACCENT, BLUE, BLUE_LT, FILL_DARK, FILL_LIGHT, FILL_MID,  # noqa: E402
                      GREY, INK, LINE, MONO, RED, RED_LT, SANS, SERIF, Fig, crown)
import genmod  # noqa: E402
from genmod import EnvConfig, legal_action_mask, reset, step  # noqa: E402
from genmod.bots import expander_bot  # noqa: E402
from genmod.state import army_counts, land_counts  # noqa: E402

MEDIA = ROOT / "media"; MEDIA.mkdir(exist_ok=True)
W, H = 760, 470          # one slide-friendly size for every idea figure


# --------------------------------------------------------------------------- #
# real 1-ply score distributions
# --------------------------------------------------------------------------- #
def _cfg(size=15):
    return EnvConfig(height=size, width=size, min_spawn_dist=7, max_spawn_dist=10**9,
                     max_steps=900, bfs_iters=4 * size)


def play_to(tick, seed=6, size=15):
    cfg = _cfg(size)
    st = reset(jax.random.PRNGKey(seed), cfg)
    k = jax.random.PRNGKey(seed + 1)
    for _ in range(tick):
        k, a, b = jax.random.split(k, 3)
        st, *_ = step(st, jnp.stack([expander_bot(a, st, cfg, 0),
                                     expander_bot(b, st, cfg, 1)]), cfg)
        if bool(st.done):
            break
    return cfg, st


def one_ply_scores(st, cfg, player=0):
    """Score every legal move by land+army swing after one tick. Returns sorted desc."""
    mask = np.asarray(legal_action_mask(st, cfg)[player]).reshape(-1)
    moves = np.flatnonzero(mask)
    moves = moves[moves % 9 != 0]
    if len(moves) == 0:
        return np.array([0.0])

    @partial(jax.jit, static_argnames=())
    def batch(s, acts):
        def one(a):
            pair = jnp.zeros((2,), jnp.int32).at[player].set(a)
            nxt = step(s, pair, cfg)[0]
            land = land_counts(nxt, 2).astype(jnp.float32)
            army = army_counts(nxt, 2).astype(jnp.float32)
            producers = (nxt.castle | nxt.general)
            mine = nxt.owner == player
            theirs = (nxt.owner >= 0) & ~mine
            prod = (mine & producers).sum() - (theirs & producers).sum()
            # frontier: our tiles touching something we do not own - expansion potential
            pad = jnp.pad(mine, 1)
            nbr = (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:])
            frontier = (nbr & ~mine & ~nxt.mountain).sum()
            return ((land[player] - land[1 - player])
                    + 0.25 * (army[player] - army[1 - player])
                    + 25.0 * prod
                    + 0.10 * frontier)
        return jax.vmap(one)(acts)

    sc = np.asarray(batch(st, jnp.asarray(moves, jnp.int32)))
    return np.sort(sc)[::-1]


# --------------------------------------------------------------------------- #
# small chart helpers
# --------------------------------------------------------------------------- #
def axes(f, x, y, w, h, xlab="", ylab=""):
    f.line(x, y + h, x + w, y + h, stroke=INK, sw=1.1)
    f.line(x, y, x, y + h, stroke=INK, sw=1.1)
    if xlab:
        f.text(x + w / 2, y + h + 30, xlab, size=10.5, fill=GREY, anchor="middle")
    if ylab:
        f.add(f'<text x="{x-30}" y="{y+h/2}" font-family="{SANS}" font-size="10.5" '
              f'fill="{GREY}" text-anchor="middle" transform="rotate(-90 {x-30} {y+h/2})">'
              f'{ylab}</text>')


def head(f, num, title, sub):
    f.text(28, 38, f"Idea {num}", size=11.5, family=MONO, fill=ACCENT, weight="600")
    f.text(28, 62, title, size=16, weight="600")
    f.text(28, 82, sub, size=11.5, fill=GREY)


# --------------------------------------------------------------------------- #
def fig4_value_of_computation():
    """Measured: the tie fraction is bimodal, so a simple threshold works."""
    import json
    data = json.load(open(MEDIA / "_gapdata.json"))
    fr = [d["frac"] for d in data]
    n = len(fr)
    f = Fig(W, H)
    head(f, 1, "Value of computation is measurable, not estimated",
         "Measured over a full game: how many legal moves tie at the best 1-ply score.")

    # --- histogram ---------------------------------------------------------
    x0, y0, bw, bh = 66, 130, 380, 180
    nb = 10
    counts = [0] * nb
    for v in fr:
        counts[min(int(v * nb), nb - 1)] += 1
    mx = max(counts)
    axes(f, x0, y0, bw, bh, "% of moves tied at the best score", "ticks")
    for i, c in enumerate(counts):
        bx = x0 + 4 + i * (bw - 8) / nb
        hgt = c / mx * (bh - 12)
        col = "#3a6b45" if i == 0 else (ACCENT if i >= 5 else FILL_MID)
        f.rect(bx, y0 + bh - hgt, (bw - 8) / nb - 3, hgt, fill=col, stroke=LINE, sw=0.6)
        if c:
            f.text(bx + ((bw - 8) / nb - 3) / 2, y0 + bh - hgt - 5, str(c),
                   size=9, anchor="middle", fill=GREY, family=MONO)
    for v, lab in ((0, "0%"), (0.5, "50%"), (1.0, "100%")):
        f.text(x0 + 4 + v * (bw - 8), y0 + bh + 15, lab, size=9.5,
               anchor="middle", fill=GREY, family=MONO)

    lo = sum(1 for v in fr if v < 0.10)
    hi = sum(1 for v in fr if v > 0.50)
    f.text(x0, y0 - 12, "The distribution is bimodal", size=12.5, weight="600")

    # --- what that licenses ------------------------------------------------
    px = 480
    f.rect(px, 130, 250, 84, fill="#eef6ef", stroke="#3a6b45", sw=1.2)
    f.text(px + 14, 154, f"{100*lo/n:.0f}% of ticks", size=17, weight="700",
           fill="#3a6b45", family=MONO)
    f.text(px + 14, 175, "almost nothing ties.", size=11.5)
    f.text(px + 14, 193, "1-ply already decided it -", size=11.5)
    f.text(px + 14, 208, "play now, ~3 ms total.", size=11.5, weight="600")

    f.rect(px, 226, 250, 84, fill="#fdf6e3", stroke=ACCENT, sw=1.2)
    f.text(px + 14, 250, f"{100*hi/n:.0f}% of ticks", size=17, weight="700",
           fill=ACCENT, family=MONO)
    f.text(px + 14, 271, "most moves tie. 1-ply", size=11.5)
    f.text(px + 14, 286, "cannot separate them -", size=11.5)
    f.text(px + 14, 304, "spend the budget here.", size=11.5, weight="600")

    f.rect(28, 348, W - 56, 84, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(44, 370, "Because the distribution is bimodal, a single threshold "
                    "separates the two cases cleanly.", size=11.5, weight="600")
    f.text(44, 391, "And the signal is not the gap to the second-best move - "
                    "ties are everywhere, so that gap", size=11.5)
    f.text(44, 408, "is almost always zero. It is how many moves are "
                    "indistinguishable.", size=11.5)
    f.text(44, 428, "Cost side: one tick of tempo \u2248 \u03c1 army, "
                    "\u03c1 = producers/2 + land/50, fixed by the rules.",
           size=11, fill=GREY)

    f.caption("4", "Deciding whether a tick deserves a search, from measured data.",
              y=H - 14)
    return f.save(MEDIA / "fig4-value-of-computation.svg")


def fig5_speculative():
    f = Fig(W, H)
    head(f, 2, "Speculative pre-computation",
         "Production is deterministic, so most of the next board is already known.")

    tw, th, y0 = 150, 34, 140
    for i, lab in enumerate(["tick t", "tick t+1", "tick t+2"]):
        x = 60 + i * (tw + 20)
        f.rect(x, y0, tw, th, fill=FILL_LIGHT, stroke=LINE, sw=1)
        f.text(x + tw / 2, y0 + 22, lab, size=11.5, anchor="middle", family=MONO)
        f.text(x + tw / 2, y0 - 8, "500 ms", size=10, anchor="middle", fill=GREY)

    f.text(28, y0 + 68, "now:", size=11.5, weight="600")
    for i in range(3):
        x = 60 + i * (tw + 20)
        f.rect(x, y0 + 54, 44, 22, fill=BLUE_LT, stroke=BLUE, sw=1)
        f.text(x + 22, y0 + 69, "think", size=9.5, anchor="middle", family=MONO)
        f.rect(x + 46, y0 + 54, tw - 46, 22, fill="#f4f4f4", stroke="#dddddd", sw=1)
        f.text(x + 46 + (tw - 46) / 2, y0 + 69, "idle", size=9.5, anchor="middle",
               fill="#aaaaaa", family=MONO)

    f.text(28, y0 + 132, "with pre-roll:", size=11.5, weight="600")
    for i in range(3):
        x = 60 + i * (tw + 20)
        f.rect(x, y0 + 118, 44, 22, fill=BLUE_LT, stroke=BLUE, sw=1)
        f.text(x + 22, y0 + 133, "think", size=9.5, anchor="middle", family=MONO)
        f.rect(x + 46, y0 + 118, tw - 46, 22, fill="#e8f0e8", stroke="#3a6b45", sw=1)
        f.text(x + 46 + (tw - 46) / 2, y0 + 133, f"search t+{i+1}", size=9.5,
               anchor="middle", family=MONO, fill="#3a6b45")
        if i < 2:
            f.arrow(x + tw - 8, y0 + 146, x + tw + 26, y0 + 158,
                    stroke="#3a6b45", sw=1.1, head=5)

    f.rect(28, 350, W - 56, 76, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(44, 372, "The board at t+1 equals the board at t plus deterministic "
                    "production,", size=11.5)
    f.text(44, 390, "minus at most two moves - each perturbing a handful of "
                    "tiles out of 529.", size=11.5)
    f.text(44, 412, "So the search started at t is still almost entirely valid at t+1. "
                    "Patch, don't restart.", size=11.5, weight="600", fill="#3a6b45")
    f.caption("5", "Only possible because the game is deterministic and fully observable.",
              y=H - 14)
    return f.save(MEDIA / "fig5-speculative-precomputation.svg")


# --------------------------------------------------------------------------- #
def fig6_queue_budget():
    f = Fig(W, H)
    head(f, 3, "The move queue is a compute budget",
         "Every queued move is a tick during which the agent owes the game nothing.")

    y0 = 132
    for i in range(12):
        x = 60 + i * 52
        f.rect(x, y0, 46, 30, fill=RED_LT, stroke=RED, sw=0.9)
        f.text(x + 23, y0 + 20, f"m{i+1}", size=10, anchor="middle", family=MONO)
    f.brace(60, y0 - 10, y0 - 10, "")
    f.line(60, y0 - 14, 60 + 11 * 52 + 46, y0 - 14, stroke=INK, sw=1)
    f.line(60, y0 - 20, 60, y0 - 8, stroke=INK, sw=1)
    f.line(60 + 11 * 52 + 46, y0 - 20, 60 + 11 * 52 + 46, y0 - 8, stroke=INK, sw=1)
    f.text((60 + 60 + 11 * 52 + 46) / 2, y0 - 24, "one queued path, 12 moves",
           size=11, anchor="middle", fill=GREY)

    f.rect(60, y0 + 54, 11 * 52 + 46, 34, fill="#e8f0e8", stroke="#3a6b45", sw=1.2)
    f.text((60 + 60 + 11 * 52 + 46) / 2, y0 + 76,
           "12 × 0.5 s  =  6 seconds of uninterrupted computation",
           size=12.5, anchor="middle", weight="600", fill="#3a6b45", family=MONO)

    f.text(28, y0 + 130, "The commit decision becomes a trade, not a confidence test:",
           size=12, weight="600")
    f.rect(60, y0 + 146, 300, 74, fill="#e8f0e8", stroke="#3a6b45", sw=1)
    f.text(76, y0 + 168, "path inside our own territory,", size=11)
    f.text(76, y0 + 185, "no enemy stack in reach", size=11)
    f.text(76, y0 + 206, "→ queue it all, buy the time", size=11.5, weight="600",
           fill="#3a6b45")
    f.rect(384, y0 + 146, 300, 74, fill="#fdf1f0", stroke=RED, sw=1)
    f.text(400, y0 + 168, "path approaching contact", size=11)
    f.text(400, y0 + 185, "- the position may change", size=11)
    f.text(400, y0 + 206, "→ commit one move, re-plan", size=11.5, weight="600",
           fill=RED)

    f.caption("6", "Prefix commitment, with a mechanic the game supplies natively.",
              y=H - 14)
    return f.save(MEDIA / "fig6-queue-as-budget.svg")


if __name__ == "__main__":
    for fn in (fig4_value_of_computation, fig5_speculative, fig6_queue_budget):
        print(" ", pathlib.Path(fn()).name)


# --------------------------------------------------------------------------- #
def fig7_phases():
    import json
    data = json.load(open(MEDIA / "_gapdata.json"))
    f = Fig(W, H)
    head(f, 4, "Let the game phase emerge, don't hardcode it",
         "Chess-style early/mid/end splits are arbitrary here. The signal defines them.")
    x0, y0, bw, bh = 66, 128, 640, 176
    axes(f, x0, y0, bw, bh, "tick", "% of moves tied at best")
    ts = [d["t"] for d in data]; t0, t1 = min(ts), max(ts)
    X = lambda t: x0 + (t - t0) / (t1 - t0) * bw
    Y = lambda fr: y0 + bh - fr * bh
    pts = " ".join(f"{X(d['t']):.1f},{Y(d['frac']):.1f}" for d in data)
    f.add(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="1.4"/>')
    # regimes read off the series
    for a, b, lab, col in ((t0, 130, "opening\nfew moves, all equivalent", "#eef4f8"),
                           (130, 215, "contact\nmany moves, mostly tied", "#fdf6e3"),
                           (215, t1, "conversion\none move stands out", "#eef6ef")):
        f.rect(X(a), y0, X(b) - X(a), bh, fill=col, stroke="none")
    f.add(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="1.4"/>')
    for a, b, lab in ((t0, 130, "opening"), (130, 215, "contact"), (215, t1, "conversion")):
        f.line(X(b), y0, X(b), y0 + bh, stroke="#bbbbbb", sw=0.8, dash="3 3")
        f.text((X(a) + X(b)) / 2, y0 - 10, lab, size=11.5, anchor="middle", weight="600")
    f.rect(28, 344, W - 56, 88, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(44, 366, "The phases are visible in a statistic we compute anyway.", size=12,
           weight="600")
    f.text(44, 386, "No tick thresholds to tune, and the boundaries move with the game", size=11.5)
    f.text(44, 403, "rather than sitting at a number someone guessed.", size=11.5)
    f.text(44, 423, "Strategy switching then falls out of budget allocation, not a "
                    "separate mechanism.", size=11.5, fill=GREY)
    f.caption("7", "Phase as a consequence of the metareasoning signal.", y=H - 14)
    return f.save(MEDIA / "fig7-emergent-phases.svg")


def fig8_derived_weights():
    f = Fig(W, H)
    head(f, 5, "Evaluation weights derived from the rules, not tuned",
         "Production is fixed by the rulebook, so the exchange rate is arithmetic.")
    y = 128
    f.rect(60, y, 300, 92, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.circle(96, y + 34, 13, stroke=INK, sw=1.4)
    f.text(120, y + 39, "one castle", size=12.5, weight="600")
    f.text(76, y + 66, "+1 army every 2 ticks", size=11.5, family=MONO)
    f.text(76, y + 84, "= 0.50 army / tick", size=12, family=MONO, weight="600", fill=RED)

    f.rect(400, y, 300, 92, fill=FILL_LIGHT, stroke=LINE, sw=1)
    for i in range(5):
        f.rect(418 + i * 15, y + 26, 12, 12, fill=RED_LT, stroke=LINE, sw=0.6)
    f.text(504, y + 39, "one land tile", size=12.5, weight="600")
    f.text(416, y + 66, "+1 army every 50 ticks", size=11.5, family=MONO)
    f.text(416, y + 84, "= 0.02 army / tick", size=12, family=MONO, weight="600", fill=BLUE)

    f.text(W / 2, y + 126, "one castle  =  25 land tiles of production",
           size=14, anchor="middle", weight="600")

    y2 = y + 150
    f.line(60, y2, W - 60, y2, stroke="#dddddd", sw=1)
    f.text(60, y2 + 26, "Now price them. A tile costs 1 army permanently "
                        "(one unit stays behind).", size=11.5)
    f.text(60, y2 + 44, "A neutral castle costs its garrison: 40-50.", size=11.5)

    by, bw2 = y2 + 60, 260
    f.text(60, by - 6, "spend 45 army on:", size=11.5, weight="600")
    f.rect(60, by, bw2 * 0.9, 24, fill=RED_LT, stroke=RED, sw=1)
    f.text(66, by + 17, "45 tiles → 0.90 army/tick", size=11.5, family=MONO)
    f.rect(60, by + 32, bw2 * 0.5, 24, fill=BLUE_LT, stroke=BLUE, sw=1)
    f.text(66, by + 49, "1 castle → 0.50 army/tick", size=11.5, family=MONO)
    f.text(60 + bw2 + 20, by + 34, "land wins by ~1.8×", size=13, weight="600",
           fill=RED)
    f.text(60 + bw2 + 20, by + 54, "whenever that much free land is reachable",
           size=10.5, fill=GREY)
    f.caption("8", "A falsifiable claim about a game most people play by feel.", y=H - 14)
    return f.save(MEDIA / "fig8-derived-weights.svg")


def fig9_cycle():
    f = Fig(W, H)
    head(f, 6, "The value of a tile depends on where you are in the 50-tick cycle",
         "Every owned tile pays +1 army on a global schedule, every 50 ticks.")
    x0, y0, bw, bh = 66, 140, 620, 150
    axes(f, x0, y0, bw, bh, "tick within the cycle  (t mod 50)",
         "value of capturing a tile now")
    for i in range(101):
        t = i / 100 * 50
        pass
    pts = []
    for i in range(101):
        t = i / 100 * 50
        v = 1.0 / (1.0 + (50 - t) / 10.0)
        pts.append(f"{x0 + t/50*bw:.1f},{y0 + bh - v/1.0*bh*0.92:.1f}")
    f.add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{INK}" stroke-width="1.6"/>')
    f.line(x0 + bw, y0, x0 + bw, y0 + bh, stroke=RED, sw=1.4)
    f.text(x0 + bw - 6, y0 - 8, "land tick: every owned tile pays +1", size=10.5,
           anchor="end", fill=RED, weight="600")
    for t, lab in ((0, "0"), (25, "25"), (49, "49")):
        f.text(x0 + t / 50 * bw, y0 + bh + 15, lab, size=9.5, anchor="middle",
               fill=GREY, family=MONO)
    f.text(x0 + 10, y0 + bh - 14, "a tile taken here waits 49 ticks to pay",
           size=10.5, fill=GREY)
    f.text(x0 + bw - 10, y0 + 24, "a tile taken here pays next tick",
           size=10.5, fill=RED, anchor="end")
    f.rect(28, 332, W - 56, 96, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(44, 354, "Predicted behaviour: expand hard just before the tick, "
                    "consolidate just after.", size=12, weight="600")
    f.text(44, 376, "Strong human players do this by instinct. We would do it because "
                    "the evaluation", size=11.5)
    f.text(44, 393, "says so - the weight is derived, not tuned, and it costs "
                    "nothing to compute.", size=11.5)
    f.text(44, 416, "Test: add the term, run 200 games, and watch the viewer for a "
                    "visible rhythm.", size=11.5, fill=GREY)
    f.caption("9", "A time-varying weight that falls out of the production rules.",
              y=H - 14)
    return f.save(MEDIA / "fig9-fifty-tick-cycle.svg")


def fig10_zones():
    """Conflict-zone decomposition of the simultaneous-move matrix."""
    from fig_core import real_position, draw_board
    cfg, st = real_position()
    f = Fig(W, 500)
    head(f, 7, "Decompose the simultaneous-move matrix by spatial independence",
         "My move on one edge and theirs on the other affect disjoint tiles.")

    cell = 15
    bx, by = 42, 116
    draw_board(f, st, cfg, bx, by, cell)
    for cx, cy, r in ((bx + 4.2 * cell, by + 2.4 * cell, 2.6 * cell),
                      (bx + 9.5 * cell, by + 5.0 * cell, 2.2 * cell),
                      (bx + 6.0 * cell, by + 10.5 * cell, 2.0 * cell)):
        f.circle(cx, cy, r, fill="none", stroke=RED, sw=1.6)
    f.text(bx, by + cfg.height * cell + 18, "three contested zones", size=11, fill=RED)

    mx = 330
    f.text(mx, by + 6, "one 400 × 400 matrix game", size=12, weight="600")
    f.rect(mx, by + 18, 120, 120, fill=FILL_MID, stroke=INK, sw=1.2)
    f.text(mx + 60, by + 84, "unmanageable", size=11, anchor="middle", fill=GREY)
    f.arrow(mx + 140, by + 78, mx + 186, by + 78, stroke=INK, sw=1.4)

    f.text(mx + 200, by + 6, "three small ones", size=12, weight="600")
    for i in range(3):
        f.rect(mx + 200 + i * 46, by + 18 + i * 20, 38, 38,
               fill=RED_LT, stroke=RED, sw=1.1)
    f.text(mx + 200, by + 132, "solve each, add the values", size=11, fill=GREY)

    f.rect(28, 352, W - 56, 96, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(44, 374, "n zones with k moves each side:   n · k²   "
                    "instead of   kⁿ · kⁿ", size=13,
           family=MONO, weight="600")
    f.text(44, 398, "This is the independent-subproblems idea from the CSP chapter, "
                    "applied to a game", size=11.5)
    f.text(44, 415, "matrix rather than a constraint graph - the link between "
                    "chapters 5 and 6.", size=11.5)
    f.text(44, 436, "Risk: by the endgame the board may be one connected blob. "
                    "Measure the zone count first.", size=11, fill=RED)
    f.caption("10", "Independent subproblems, borrowed from constraint satisfaction.",
              y=486)
    return f.save(MEDIA / "fig10-conflict-zones.svg")


def fig11_safety_proof():
    """AND-OR search proving the general cannot be captured within k ticks."""
    f = Fig(W, 500)
    head(f, 8, "Prove safety, then stop defending",
         "With no fog and deterministic rules, this question has a yes-or-no answer.")

    cx, y = W / 2, 150
    f.circle(cx, y, 13, fill="#ffffff", stroke=INK, sw=1.5)
    f.text(cx, y + 4, "s", size=11, anchor="middle", family=SERIF, style="italic")
    f.text(cx + 22, y + 4, "my position", size=10.5, fill=GREY)
    f.text(60, y + 4, "OR  (my move)", size=11, fill=BLUE, weight="600")

    kids = [cx - 150, cx, cx + 150]
    y2 = y + 80
    for x in kids:
        f.line(cx, y + 13, x, y2 - 13, stroke=INK, sw=1.1)
        f.rect(x - 13, y2 - 13, 26, 26, fill=FILL_LIGHT, stroke=INK, sw=1.2)
    f.text(60, y2 + 4, "AND  (their reply)", size=11, fill=RED, weight="600")
    f.path(f"M {cx-150+16} {y2+13} A 16 16 0 0 0 {cx-150+16+0.1} {y2+13}",
           stroke="none")

    y3 = y2 + 84
    outcomes = [("safe", "#3a6b45"), ("safe", "#3a6b45"), ("safe", "#3a6b45"),
                ("safe", "#3a6b45"), ("CAPTURED", RED), ("safe", "#3a6b45"),
                ("safe", "#3a6b45"), ("safe", "#3a6b45"), ("safe", "#3a6b45")]
    i = 0
    for x in kids:
        for j in range(3):
            xx = x - 44 + j * 44
            lab, col = outcomes[i]; i += 1
            f.line(x, y2 + 13, xx, y3 - 10, stroke=LINE, sw=0.8)
            f.circle(xx, y3, 9, fill="#ffffff", stroke=col, sw=1.4)
            if lab == "CAPTURED":
                f.line(xx - 4, y3 - 4, xx + 4, y3 + 4, stroke=col, sw=1.6)
                f.line(xx - 4, y3 + 4, xx + 4, y3 - 4, stroke=col, sw=1.6)
            else:
                f.path(f"M {xx-4} {y3} L {xx-1} {y3+3.5} L {xx+4.5} {y3-4}",
                       stroke=col, sw=1.6)
    f.text(kids[0], y3 + 32, "all replies survive", size=10.5,
           anchor="middle", fill="#3a6b45", weight="600")
    f.text(kids[0], y3 + 47, "→ this move is safe", size=10.5,
           anchor="middle", fill="#3a6b45")
    f.text(kids[1], y3 + 32, "one reply captures", size=10.5,
           anchor="middle", fill=RED, weight="600")
    f.text(kids[1], y3 + 47, "→ this move is not safe", size=10.5,
           anchor="middle", fill=RED)

    f.rect(28, 378, W - 56, 82, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(44, 400, "A move is safe only if it survives EVERY reply. "
                    "That is an AND node (AIMA 4.3).", size=11.5)
    f.text(44, 420, "If some move is provably safe for k ticks: "
                    "spend the whole budget on attack.", size=11.5, weight="600",
           fill="#3a6b45")
    f.text(44, 440, "If none is: override whatever the evaluation recommended. "
                    "A proof beats an opinion.", size=11.5, weight="600", fill=RED)
    f.caption("11", "A small AND-OR search used as a safety gate, not as the "
                    "main search.", y=486)
    return f.save(MEDIA / "fig11-safety-proof.svg")


def fig12_beam():
    """Adaptive beam width taken from the shape of the score distribution."""
    import json
    f = Fig(W, 500)
    head(f, 9, "Adaptive beam width from the score distribution",
         "Beam search keeps the best k. We can read k off the data instead of guessing.")

    cfgA, stA = play_to(200)
    cfgB, stB = play_to(175)
    for panel, (st, cfg, lab) in enumerate([(stA, cfgA, "a few clear leaders"),
                                            (stB, cfgB, "a long tie at the top")]):
        sc = one_ply_scores(st, cfg)[:26]
        x0, y0, bw, bh = 60 + panel * 360, 138, 260, 150
        axes(f, x0, y0, bw, bh, "moves, best first", "1-ply score" if panel == 0 else "")
        lo, hi = float(sc.min()), float(sc.max())
        rng = max(hi - lo, 1e-6)
        n = len(sc); wid = (bw - 10) / n
        # the largest step in the top of the distribution is where to cut
        steps = [(sc[i] - sc[i + 1], i) for i in range(min(n - 1, 20))]
        cut = max(steps)[1] + 1 if steps and max(steps)[0] > 1e-9 else n
        for i, v in enumerate(sc):
            bx = x0 + 5 + i * wid
            hgt = 6 + (v - lo) / rng * (bh - 18)
            f.rect(bx, y0 + bh - hgt, wid - 2, hgt,
                   fill=RED_LT if i < cut else FILL_MID, stroke=LINE, sw=0.4)
        cx_ = x0 + 5 + cut * wid - 1
        f.line(cx_, y0 - 6, cx_, y0 + bh + 4, stroke=RED, sw=1.4, dash="4 3")
        f.text(x0, y0 - 12, lab, size=12, weight="600")
        f.text(cx_ + 6, y0 + 6, f"k = {cut}", size=11.5, fill=RED, family=MONO,
               weight="600")

    f.rect(28, 350, W - 56, 106, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(44, 372, "A fixed k is wrong in both cases: too wide on the left, "
                    "too narrow on the right.", size=12, weight="600")
    f.text(44, 394, "Since the full distribution costs 0.6 ms, cut at the largest "
                    "step instead - the", size=11.5)
    f.text(44, 411, "beam width becomes a property of the position rather than a "
                    "constant in the code.", size=11.5)
    f.text(44, 434, "Small idea, nearly free, and it composes with everything else.",
           size=11, fill=GREY)
    f.caption("12", "Beam search (AIMA 4.1) with the width read from the data.", y=486)
    return f.save(MEDIA / "fig12-adaptive-beam.svg")
