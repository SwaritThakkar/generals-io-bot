"""Render media/d10-weighted-astar.png from position.json.

Every board cell, move, expansion count and cost is read from position.json,
which find_position.py produced by running the real engine and the real search.
Nothing here invents a number.
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # project root

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle

W, H = 1960, 1000
TEXT, MUTED, GOLD = "#1A1A1A", "#808080", "#8F6410"
RED, BLUE = "#B4342C", "#2C5F86"
T_OURS, T_THEIRS, T_FREE, T_MTN = "#F0C4BE", "#BFD4E6", "#F0F0EE", "#C9C9C4"
GLYPH = "#43433E"   # army numbers, a shade back so the move reads first
MOVE_COLOURS = ["#7BA05B", "#4F8A3D", "#2E6B34"]
CARD_BG, CARD_EDGE = "#F1F7F1", "#2E6B34"
BOX_BG, BOX_EDGE = "#EFEFEF", "#9A9A9A"

PANEL_X = [60, 700, 1340]
PANEL_W = 560
BOARD_TOP = 225
BOARD_MAX_H = 380
MONO = {"family": "DejaVu Sans Mono"}
SERIF = {"family": "DejaVu Serif"}


# --------------------------------------------------------------------------- #
# board drawing
# --------------------------------------------------------------------------- #
def crop_window(d, pad=1):
    """Smallest window holding every mover, destination, target and both generals."""
    h, w = d["h"], d["w"]
    pts = []
    for p in d["panels"]:
        pts += [tuple(p["mover"]), tuple(p["dest"]), tuple(p["target"])]
    pts += [divmod(int(g), w) for g in d["generalIdx"] if g >= 0]
    r0 = max(0, min(p[0] for p in pts) - pad)
    r1 = min(h - 1, max(p[0] for p in pts) + pad)
    c0 = max(0, min(p[1] for p in pts) - pad)
    c1 = min(w - 1, max(p[1] for p in pts) + pad)
    return r0, r1, c0, c1


def star(cx, cy, r, colour):
    a = np.linspace(-np.pi / 2, 3 * np.pi / 2, 11)[:-1]
    rad = np.where(np.arange(10) % 2 == 0, r, r * 0.45)
    return Polygon(np.c_[cx + rad * np.cos(a), cy + rad * np.sin(a)],
                   closed=True, facecolor=colour, edgecolor="none", zorder=5)


def draw_board(ax, d, win, x0, y0, size):
    """Draw the cropped board with its top-left corner at (x0, y0)."""
    h, w = d["h"], d["w"]
    r0, r1, c0, c1 = win
    owner = np.array(d["owner"]).reshape(h, w)
    army = np.array(d["army"]).reshape(h, w)
    mtn = np.array(d["mountain"]).reshape(h, w).astype(bool)
    city = np.array(d["castle"]).reshape(h, w).astype(bool)
    gen = np.array(d["general"]).reshape(h, w).astype(bool)

    fs = max(6.2, size * 0.38)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            x = x0 + (c - c0) * size
            y = y0 + (r - r0) * size
            if mtn[r, c]:
                fill = T_MTN
            elif owner[r, c] == 0:
                fill = T_OURS
            elif owner[r, c] == 1:
                fill = T_THEIRS
            else:
                fill = T_FREE
            ax.add_patch(Rectangle((x, y), size, size, facecolor=fill,
                                   edgecolor="white", linewidth=1.4, zorder=2))
            cx, cy = x + size / 2, y + size / 2
            if mtn[r, c]:
                t = size * 0.26
                ax.add_patch(Polygon([(cx, cy - t), (cx - t, cy + t * 0.8),
                                      (cx + t, cy + t * 0.8)], closed=True,
                                     facecolor="#8E8E88", edgecolor="none", zorder=4))
                continue
            if gen[r, c]:
                col = RED if owner[r, c] == 0 else BLUE
                ax.add_patch(star(cx, cy - size * 0.13, size * 0.25, col))
                ax.text(cx, cy + size * 0.34, str(int(army[r, c])), ha="center",
                        va="center", fontsize=fs, color=GLYPH, zorder=6, **MONO)
            elif city[r, c]:
                ax.add_patch(Circle((cx, cy - size * 0.13), size * 0.19,
                                    facecolor="white", edgecolor="#6E6E68",
                                    linewidth=1.1, zorder=4))
                ax.text(cx, cy + size * 0.34, str(int(army[r, c])), ha="center",
                        va="center", fontsize=fs, color=GLYPH, zorder=6, **MONO)
            elif owner[r, c] >= 0 and army[r, c]:
                ax.text(cx, cy, str(int(army[r, c])), ha="center", va="center",
                        fontsize=fs, color=GLYPH, zorder=6, **MONO)

    bw, bh = (c1 - c0 + 1) * size, (r1 - r0 + 1) * size
    ax.add_patch(Rectangle((x0, y0), bw, bh, facecolor="none",
                           edgecolor="#5A5A54", linewidth=1.6, zorder=7))
    return bw, bh


def draw_move(ax, panel, win, x0, y0, size, colour):
    """One arrow, mover -> adjacent destination, and the destination outlined."""
    r0, _, c0, _ = win
    (mr, mc), (dr_, dc_) = panel["mover"], panel["dest"]
    mx = x0 + (mc - c0 + 0.5) * size
    my = y0 + (mr - r0 + 0.5) * size
    dx = x0 + (dc_ - c0 + 0.5) * size
    dy = y0 + (dr_ - r0 + 0.5) * size
    dxr, dyr = x0 + (dc_ - c0) * size, y0 + (dr_ - r0) * size
    ring = Rectangle((dxr, dyr), size, size, facecolor="none", edgecolor=colour,
                     linewidth=3.0, zorder=11)
    ring.set_path_effects([pe.withStroke(linewidth=5.6, foreground="white")])
    ax.add_patch(ring)
    vx, vy = dx - mx, dy - my
    n = float(np.hypot(vx, vy)) or 1.0
    ux, uy = vx / n, vy / n
    # Run the arrow from the back of the mover tile to the far side of the
    # destination so a single-tile step is still legible at this board scale.
    back, over, head = size * 0.34, size * 0.30, size * 0.50
    sx, sy = mx - ux * back, my - uy * back
    span = n + back + over
    arrow = FancyArrow(sx, sy, (span - head) * ux, (span - head) * uy,
                       width=size * 0.20, head_width=size * 0.58,
                       head_length=head, length_includes_head=True,
                       facecolor=colour, edgecolor="none", zorder=12)
    arrow.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
    ax.add_patch(arrow)
    origin = Circle((mx, my), size * 0.36, facecolor="none", edgecolor=colour,
                    linewidth=2.4, zorder=13)
    origin.set_path_effects([pe.withStroke(linewidth=4.4, foreground="white")])
    ax.add_patch(origin)


def swatch(ax, x, y, fill, label, edge="#BDBDB8", glyph=None):
    ax.add_patch(Rectangle((x, y), 22, 22, facecolor=fill, edgecolor=edge,
                           linewidth=1.0, zorder=3))
    if glyph == "city":
        ax.add_patch(Circle((x + 11, y + 11), 4.6, facecolor="white",
                            edgecolor="#6E6E68", linewidth=1.0, zorder=4))
    elif glyph == "mtn":
        ax.add_patch(Polygon([(x + 11, y + 6), (x + 5, y + 16), (x + 17, y + 16)],
                             closed=True, facecolor="#8E8E88", edgecolor="none", zorder=4))
    ax.text(x + 30, y + 12, label, ha="left", va="center", fontsize=11, color=MUTED)
    return x + 30 + len(label) * 6.6 + 26


# --------------------------------------------------------------------------- #
def build_panels(d):
    """Everything panel-specific, so a different position.json just works."""
    h, w = d["h"], d["w"]
    army = np.array(d["army"]).reshape(h, w)
    city = np.array(d["castle"]).reshape(h, w).astype(bool)
    verdicts = ["Good", "Better", "Best"]
    out = []
    for i, p in enumerate(d["panels"]):
        tr, tc = p["target"]
        kind = p["target_kind"]
        if kind == "neutral city":
            what = f"takes the neutral city, garrison {int(army[tr, tc])}"
            payoff = "a 25-tile city"
        elif kind == "enemy general":
            what = "walks onto the enemy general"
            payoff = "the game"
        else:
            what = f"takes an enemy tile holding {int(army[tr, tc])}"
            payoff = "one enemy tile"
        top = max(q["expanded"] for q in d["panels"])
        effort = ("the full search" if p["expanded"] == top
                  else f"{100 * p['expanded'] / top:.0f}% of the nodes ε = 1 needed")
        out.append(dict(
            eps=p["epsilon"], colour=MOVE_COLOURS[i], mover=p["mover"], dest=p["dest"],
            move_name=f"{p['dir_name']} toward {'the' if kind == 'neutral city' else 'an'} {kind}",
            expanded=p["expanded"], ticks=p["ticks"], cost=p["cost"], effort=effort,
            what=what, worth=f"{p['ticks']} ticks for {payoff}",
            verdict=verdicts[i], target=p["target"], kind=kind,
        ))
    return out


def main():
    d = json.loads(Path(__file__).with_name("position.json").read_text())
    panels = build_panels(d)
    win = crop_window(d)
    rows, cols = win[1] - win[0] + 1, win[3] - win[2] + 1
    size = min(PANEL_W / cols, BOARD_MAX_H / rows)

    fig = plt.figure(figsize=(19.6, 10.0), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.set_facecolor("white")

    # ---- the third panel sits on a card -----------------------------------
    ax.add_patch(Rectangle((PANEL_X[2] - 22, 140), PANEL_W + 44, 640,
                           facecolor=CARD_BG, edgecolor=CARD_EDGE,
                           linewidth=1.4, zorder=0))

    # ---- titles ------------------------------------------------------------
    ax.text(60, 52, "One position, three good moves, three values of ε",
            ha="left", va="center", fontsize=21, fontweight="bold", color=TEXT)
    ax.text(60, 94, "The board is identical in all three panels; only ε changes. "
                    "A tighter ε searches further before it commits to a move.",
            ha="left", va="center", fontsize=12, color=MUTED)

    # ---- inline legend -----------------------------------------------------
    lx = 1170
    for fill, label, glyph in ((T_OURS, "us", None), (T_THEIRS, "them", None),
                               (T_MTN, "mountain", "mtn"),
                               (T_FREE, "neutral city", "city")):
        lx = swatch(ax, lx, 40, fill, label, glyph=glyph)

    # ---- panels ------------------------------------------------------------
    bw = cols * size
    for i, (px, pn) in enumerate(zip(PANEL_X, panels)):
        colour = pn["colour"]
        head = CARD_EDGE if i == 2 else GOLD
        ax.text(px, 172, f"ε = {pn['eps']:g}", ha="left", va="center",
                fontsize=14, fontweight="bold", color=head, **MONO)
        ax.text(px, 202, pn["move_name"], ha="left", va="center",
                fontsize=13, fontweight="bold", color=TEXT)

        x0 = px + (PANEL_W - bw) / 2
        draw_board(ax, d, win, x0, BOARD_TOP, size)
        draw_move(ax, pn, win, x0, BOARD_TOP, size, colour)

        # search effort bar, normalised so the largest count fills the track
        ax.text(px, 625, "search effort", ha="left", va="center",
                fontsize=10, color=MUTED)
        tx, tw = px + 140, 360
        top = max(q["expanded"] for q in panels)
        ax.add_patch(Rectangle((tx, 618), tw, 14, facecolor="#E4E4E0",
                               edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((tx, 618), tw * pn["expanded"] / top, 14,
                               facecolor=colour, edgecolor="none", zorder=4))

        ax.text(px, 665, f"{pn['expanded']} nodes expanded", ha="left", va="center",
                fontsize=11.5, color=TEXT)
        ax.text(px, 692, pn["what"], ha="left", va="center", fontsize=11.5, color=TEXT)
        ax.text(px, 719, pn["effort"], ha="left", va="center",
                fontsize=11.5, color=MUTED)
        ax.text(px, 758, f"{pn['verdict']} — {pn['worth']}".replace("—", "-"),
                ha="left", va="center", fontsize=12, fontweight="bold", color=colour)

    # ---- takeaway box ------------------------------------------------------
    ax.add_patch(Rectangle((60, 820), W - 120, 86, facecolor=BOX_BG,
                           edgecolor=BOX_EDGE, linewidth=1.2, zorder=1))
    ax.text(84, 850, "All three are moves we would be happy to play: a large ε returns a "
                     "shallower move, not a bad one.", ha="left", va="center",
            fontsize=12, color=TEXT, zorder=2)
    ax.text(84, 880, "So we take the ε = 3 move if the tick ends now, and keep tightening "
                     "ε for as long as time remains.", ha="left", va="center",
            fontsize=12, fontweight="bold", color=CARD_EDGE, zorder=2)

    # ---- caption -----------------------------------------------------------
    r0, r1, c0, c1 = win
    ax.text(60, 955, "Figure J", ha="left", va="center", fontsize=12,
            fontweight="bold", color=TEXT, **SERIF)
    ax.text(178, 955, f"Weighted A* on one real position: seed {d['seed']}, tick {d['tick']}. "
                      f"Board cropped to rows {r0}-{r1}, columns {c0}-{c1}; the crop is the "
                      f"same in all three panels. Every count is measured.",
            ha="left", va="center", fontsize=12, color=TEXT, **SERIF)

    out = Path(__file__).parent / "part1" / "media" / "d10-weighted-astar.png"
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  {W}x{H}")


if __name__ == "__main__":
    main()
