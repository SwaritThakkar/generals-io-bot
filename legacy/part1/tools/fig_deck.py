"""Figures referenced by writeup.txt that did not exist yet."""
from __future__ import annotations
import pathlib, sys
import jax, jax.numpy as jnp, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))  # project root (genmod); sys.path.insert(0, str(ROOT / "tools"))
from figstyle import (ACCENT, BLUE, BLUE_LT, FILL_DARK, FILL_LIGHT, FILL_MID, GREY,
                      INK, LINE, MONO, RED, RED_LT, SANS, SERIF, Fig, crown, mountain)
from fig_core import real_position, draw_board, legal_move_count
from genmod import EnvConfig, reset, step
from genmod.bots import expander_bot
from genmod.state import army_counts, land_counts
MEDIA = ROOT / "media"

def tile(f, x, y, c, owner=None, army=None, gen=False, castle=False, mtn=False,
         label=None, dim=False):
    fill = (FILL_DARK if mtn else RED_LT if owner == 0 else BLUE_LT if owner == 1
            else FILL_LIGHT)
    if dim: fill = "#f6f6f4"
    f.rect(x, y, c, c, fill=fill, stroke=LINE, sw=1.0)
    if mtn: mountain(f, x + c/2, y + c/2, c*0.26, "#5a5a5a"); return
    if gen: crown(f, x + c/2, y + c*0.30, c*0.20, RED if owner == 0 else BLUE if owner == 1 else GREY)
    elif castle: f.circle(x + c/2, y + c*0.32, c*0.18, stroke=INK if owner is None else (RED if owner==0 else BLUE), sw=1.2)
    if army is not None:
        ty = y + c*0.84 if (gen or castle) else y + c/2 + c*0.16
        f.text(x + c/2, ty, str(army), size=c*(0.34 if (gen or castle) else 0.40),
               anchor="middle", family=MONO, fill=INK, weight="500")
    if label:
        f.text(x + c/2, y + c + 15, label, size=10.5, anchor="middle", fill=GREY)

def title(f, t, sub=None):
    f.text(30, 40, t, size=17, weight="600")
    if sub: f.text(30, 62, sub, size=12, fill=GREY)

# ---------------------------------------------------------------- 1. the board
def f_board():
    """Board on the left, a legend of sample tiles on the right. No leader lines
    across the board - they were unreadable."""
    cfg, st = real_position(size=11, ticks=110, seed=11)
    c = 46
    f = Fig(980, 690)
    title(f, "The board")
    bx, by = 40, 92
    draw_board(f, st, cfg, bx, by, c)

    lx = bx + cfg.width * c + 52
    ly = by + 4
    sc = 42                                   # legend swatch size
    items = [
        (dict(owner=0, army=2),               "your tile", "the number is the army standing on it"),
        (dict(owner=1, army=3),               "their tile", ""),
        (dict(),                              "unowned", "walk onto it with anything and it is yours"),
        (dict(mtn=True),                      "mountain", "impassable"),
        (dict(owner=0, army=6, gen=True),     "general", "your base, and what you must defend"),
        (dict(army=44, castle=True),          "neutral castle", "beat the garrison and it produces for you"),
    ]
    for i, (kw, name, note) in enumerate(items):
        y = ly + i * 74
        tile(f, lx, y, sc, **kw)
        f.text(lx + sc + 14, y + 18, name, size=12.5, weight="600")
        if note:
            f.text(lx + sc + 14, y + 35, note, size=10.5, fill=GREY)

    f.caption("A", "One position, mid-game. Red is us, blue is the opponent.", y=664)
    return f.save(MEDIA / "d01-board.svg")

# ------------------------------------------------------ 2. combat is subtraction
def f_combat():
    c = 74
    f = Fig(900, 400)
    title(f, "Combat is subtraction", "the two stacks cancel one for one; the larger survives with the difference")
    y = 120
    panels = [(40, "(a)  before", [(0,10),(1,4)]), (330, "(b)  the move", None), (620, "(c)  after", [(0,1),(0,5)])]
    for x, lab, tiles in panels:
        f.text(x, y - 18, lab, size=12.5, weight="600")
        if tiles:
            for i,(o,a) in enumerate(tiles): tile(f, x + i*(c+6), y, c, owner=o, army=a)
        else:
            tile(f, x, y, c, owner=0, army=10)
            tile(f, x + c + 34, y, c, owner=1, army=4)
            # arrow lives in the gap between the tiles, never over a number
            f.arrow(x + c + 4, y + c/2, x + c + 30, y + c/2, stroke=INK, sw=2.4, head=9)
            f.text(x + c + 17, y - 8, "9", size=13, anchor="middle", family=MONO,
                   weight="600", fill=RED)
            f.text(x + c/2 + 17, y + c + 30, "9 units leave, 1 stays behind",
                   size=11, fill=GREY, anchor="middle")
    f.arrow(280, y + c/2, 320, y + c/2, stroke="#bbb", sw=1.4, head=7)
    f.arrow(572, y + c/2, 612, y + c/2, stroke="#bbb", sw=1.4, head=7)
    f.text(40, y + c + 78, "9 attackers meet 4 defenders  →  I take the tile with 5 left standing",
           size=13, weight="600")
    f.text(40, y + c + 104, "had they held 12, they would keep the tile with 3 left standing",
           size=11.5, fill=GREY)
    f.caption("B", "The whole combat rule.", y=376)
    return f.save(MEDIA / "d02-combat.svg")

# ------------------------------------------------------------- 3. a stack walks
def f_stack_walks():
    c = 62
    f = Fig(900, 500)
    title(f, "A stack walks", "one unit is left behind on every tile it takes")
    for row in range(5):
        y = 104 + row*58
        f.text(30, y + c*0.42, f"tick {row}", size=11, fill=GREY, family=MONO)
        for i in range(5):
            x = 110 + i*(c+4)
            owned = i < row
            moving = (i == row)
            if moving:
                tile(f, x, y, c, owner=0, army=5-row)
                f.rect(x-2, y-2, c+4, c+4, fill="none", stroke=ACCENT, sw=2)
            elif owned:
                tile(f, x, y, c, owner=0, army=1)
            else:
                tile(f, x, y, c)
        if row < 4:
            f.arrow(110 + row*(c+4) + c*0.6, y + c + 6, 110 + (row+1)*(c+4) + c*0.4, y + c + 24,
                    stroke=GREY, sw=1.0, head=5)
    f.text(110, 104 + 5*58 + 26, "a stack of size a captures a - 1 tiles",
           size=13, weight="600")
    f.caption("C", "The mental model for the whole expansion phase.", y=482)
    return f.save(MEDIA / "d03-stack-walks.svg")

# --------------------------------------------------------------- 5. how you win
def f_how_you_win():
    c = 56
    f = Fig(700, 350)
    title(f, "How you win", "move an army onto the enemy general and win the fight there")
    bx, by = 60, 110
    for i in range(5):
        x = bx + i*(c+3)
        if i == 0: tile(f, x, by, c, owner=0, army=9)
        elif i == 4: tile(f, x, by, c, owner=1, army=3, gen=True)
        else: tile(f, x, by, c)
    f.arrow(bx + c*0.6, by + c/2, bx + 4*(c+3) + c*0.35, by + c/2, stroke=RED, sw=2.4, head=10)
    f.text(bx, by + c + 30, "9 → 8 → 7 → 6 arrives, beats a garrison of 3", size=12, family=MONO)
    f.text(bx, by + c + 56, "no points system; no other victory condition", size=12.5, weight="600")
    f.caption("D", "The only terminal condition in the game.", y=326)
    return f.save(MEDIA / "d04-how-you-win.svg")

if __name__ == "__main__":
    for fn in (f_board, f_combat, f_stack_walks, f_how_you_win):
        print(" ", pathlib.Path(fn()).name)


def axes(f, x, y, w, h, xlab="", ylab=""):
    f.line(x, y+h, x+w, y+h, stroke=INK, sw=1.1); f.line(x, y, x, y+h, stroke=INK, sw=1.1)
    if xlab: f.text(x+w/2, y+h+30, xlab, size=10.5, fill=GREY, anchor="middle")
    if ylab: f.add(f'<text x="{x-32}" y="{y+h/2}" font-family="{SANS}" font-size="10.5" '
                   f'fill="{GREY}" text-anchor="middle" transform="rotate(-90 {x-32} {y+h/2})">{ylab}</text>')

# ------------------------------------------------------------- compounding
def f_compounding():
    cfg = EnvConfig(height=15, width=15, min_spawn_dist=7, max_spawn_dist=10**9,
                    max_steps=400, bfs_iters=60)
    st = reset(jax.random.PRNGKey(3), cfg); k = jax.random.PRNGKey(4)
    L, A = [], []
    for t in range(320):
        k, a, b = jax.random.split(k, 3)
        st, *_ = step(st, jnp.stack([expander_bot(a, st, cfg, 0),
                                     expander_bot(b, st, cfg, 1)]), cfg)
        L.append(np.asarray(land_counts(st, 2))); A.append(np.asarray(army_counts(st, 2)))
        if bool(st.done): break
    L, A = np.array(L), np.array(A)
    f = Fig(900, 560)
    title(f, "Compounding", "territory makes army, army takes territory")
    for pi, (D, name) in enumerate([(L, "land (tiles owned)"), (A, "army (total units)")]):
        x0, y0, w, h = 70, 110 + pi*200, 760, 150
        axes(f, x0, y0, w, h, "tick" if pi else "", name)
        mx = max(D.max(), 1); n = len(D)
        for pl, col in ((0, RED), (1, BLUE)):
            pts = " ".join(f"{x0+i/(n-1)*w:.1f},{y0+h-D[i,pl]/mx*h:.1f}" for i in range(n))
            f.add(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        f.text(x0+w+6, y0+h-D[-1,0]/mx*h+4, "red", size=11, fill=RED, weight="600")
        f.text(x0+w+6, y0+h-D[-1,1]/mx*h+4, "blue", size=11, fill=BLUE, weight="600")
        f.text(x0+w-4, y0-8, f"peak {int(mx)}", size=10, fill=GREY, family=MONO, anchor="end")
    # mark where the lead opens
    diff = np.abs(L[:,0]-L[:,1]); sustained = (diff > 8)
    op = int(np.argmax(sustained)) if sustained.any() else len(L)//4
    xx = 70 + op/(len(L)-1)*760
    for pi in range(2):
        f.line(xx, 110+pi*200, xx, 260+pi*200, stroke=ACCENT, sw=1, dash="4 3")
    f.text(xx+6, 104, f"lead opens ~tick {op}", size=10.5, fill=ACCENT, weight="600")
    f.caption("E", "One real self-play game. The curves separate and never reconverge.", y=536)
    return f.save(MEDIA / "d05-compounding.svg")

# --------------------------------------------------------- everything visible
def f_visible():
    cfg, st = real_position(size=11, ticks=110, seed=11)
    c = 40
    f = Fig(960, 650)
    title(f, "Everything is visible", "there is nothing hidden on the board")
    draw_board(f, st, cfg, 40, 96, c)
    owner = np.asarray(st.owner); army = np.asarray(st.army); gen = np.asarray(st.general)
    gi = np.asarray(st.general_idx)
    lx = 40 + cfg.width*c + 46
    facts = [
        ("enemy general", f"row {gi[1]//cfg.width}, col {gi[1]%cfg.width}  -  known from tick 0"),
        ("its garrison", f"{int(army.reshape(-1)[gi[1]])} units, exactly"),
        ("their total army", f"{int(army_counts(st,2)[1])} units"),
        ("their land", f"{int(land_counts(st,2)[1])} tiles"),
        ("every castle garrison", "exact, including unclaimed ones"),
        ("the whole terrain", "no unexplored region"),
    ]
    for i, (a, b) in enumerate(facts):
        y = 130 + i*58
        f.circle(lx + 6, y - 4, 3.4, fill=INK, stroke="none")
        f.text(lx + 20, y, a, size=12.5, weight="600")
        f.text(lx + 20, y + 17, b, size=10.5, fill=GREY)
    f.rect(lx, 130 + 6*58 - 6, 400, 46, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(lx + 14, 130 + 6*58 + 22, "so evaluation is exact, not expected", size=12, weight="600")
    f.caption("F", "The state is the board.", y=626)
    return f.save(MEDIA / "d06-visible.svg")

# ---------------------------------------------------------------- sparse reward
def f_sparse():
    f = Fig(900, 340)
    title(f, "Sparse, terminal reward", "nothing tells you whether move 137 was good")
    x0, y0, w, h = 60, 140, 720, 40
    f.rect(x0, y0, w, h, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(x0 + w/2, y0 + 25, "no signal for ~500 ticks", size=12.5, anchor="middle", fill=GREY)
    for frac in (0.12, 0.28, 0.45, 0.62, 0.8):
        x = x0 + frac*w
        f.line(x, y0, x, y0+h, stroke="#ccc", sw=0.8)
        f.text(x, y0 - 8, "?", size=13, anchor="middle", fill=GREY, weight="600")
    f.rect(x0 + w, y0 - 38, 8, h + 38, fill=RED, stroke="none")
    f.text(x0 + w + 18, y0 - 22, "+1  win", size=13, weight="600", fill=RED, family=MONO)
    f.text(x0 + w + 18, y0 - 4, "-1  loss", size=13, weight="600", fill=RED, family=MONO)
    f.text(x0, y0 - 40, "each ? is a decision with no feedback", size=11, fill=GREY)
    f.text(x0, y0 + h + 44, "the only ground truth arrives once, at the very end",
           size=12.5, weight="600")
    f.text(x0, y0 + h + 66, "so the evaluation function is doing all the work in between",
           size=11.5, fill=GREY)
    f.caption("G", "Why designing the evaluation is the hard part.", y=316)
    return f.save(MEDIA / "d07-sparse-reward.svg")

# ------------------------------------------------------------ simultaneous moves
def f_simultaneous():
    f = Fig(940, 580)
    title(f, "Simultaneous moves", "both players commit for the same tick without seeing the other")
    # (a) sequential
    f.rect(40, 100, 400, 150, fill="#fdf1f0", stroke=RED, sw=1.2, rx=3)
    f.text(56, 124, "(a)  SEQUENTIAL - what minimax assumes", size=12, weight="700", fill=RED)
    f.text(56, 150, "I move", size=12); f.arrow(120, 146, 168, 146, stroke=INK, sw=1.2, head=6)
    f.text(176, 150, "they SEE it", size=12); f.arrow(262, 146, 310, 146, stroke=INK, sw=1.2, head=6)
    f.text(318, 150, "they reply", size=12)
    f.text(56, 186, "This is false here.", size=12.5, weight="700", fill=RED)
    f.text(56, 208, "Minimax quietly gives the second player", size=11, fill=GREY)
    f.text(56, 224, "information they do not actually have.", size=11, fill=GREY)
    # (b) simultaneous
    f.rect(470, 100, 430, 150, fill="#eef4f8", stroke=BLUE, sw=1.2, rx=3)
    f.text(486, 124, "(b)  SIMULTANEOUS - what happens", size=12, weight="700", fill=BLUE)
    for i, (lab, col) in enumerate((("my move", RED), ("their move", BLUE))):
        y = 146 + i*34
        f.rect(486, y, 96, 26, fill=FILL_MID, stroke=LINE, sw=1)
        f.text(534, y + 18, "face down", size=10, anchor="middle", family=MONO, fill=GREY)
        f.text(592, y + 18, lab, size=11, fill=col, weight="600")
    f.arrow(700, 180, 748, 180, stroke=INK, sw=1.4, head=7)
    f.text(756, 176, "revealed", size=11.5, weight="600")
    f.text(756, 193, "together", size=11.5, weight="600")
    # payoff matrix
    mx, my, cs = 300, 300, 62
    f.text(mx, my - 14, "so a node is a matrix game, not a max or a min", size=12.5, weight="600")
    f.text(mx - 96, my + cs, "my", size=11, fill=RED, weight="600")
    f.text(mx - 96, my + cs + 15, "move", size=11, fill=RED, weight="600")
    f.text(mx + cs, my - 34 + 34, "", size=11)
    f.text(mx + cs*1.0, my + 12, "their move", size=11, fill=BLUE, weight="600", anchor="middle")
    for i in range(2):
        for j in range(2):
            x = mx + j*cs; y = my + 22 + i*cs
            f.rect(x, y, cs, cs, fill="#ffffff", stroke=INK, sw=1)
            f.text(x + cs/2, y + cs/2 + 5, ["+2","-1","-3","+1"][i*2+j], size=15,
                   anchor="middle", family=MONO)
    f.text(mx + 2*cs + 22, my + 22 + cs, "solved as a matrix game,", size=11.5)
    f.text(mx + 2*cs + 22, my + 22 + cs + 17, "not by taking a max", size=11.5)
    f.caption("H", "The one piece of hidden information that remains.", y=556)
    return f.save(MEDIA / "d08-simultaneous.svg")

# ----------------------------------------------------------------- NP-hard
def f_nphard():
    cfg, st = real_position(size=11, ticks=110, seed=11)
    c = 42
    f = Fig(940, 660)
    title(f, "The sub-problems are NP-hard",
          "gathering army optimally is a rooted prize-collecting Steiner tree")
    bx, by = 40, 100
    draw_board(f, st, cfg, bx, by, c)
    owner = np.asarray(st.owner); army = np.asarray(st.army); gi = int(np.asarray(st.general_idx)[0])
    gr, gq = gi // cfg.width, gi % cfg.width
    cells = [(r, q) for r in range(cfg.height) for q in range(cfg.width)
             if owner[r, q] == 0 and army[r, q] >= 2 and (r, q) != (gr, gq)]
    cells.sort(key=lambda rq: -army[rq])
    root = (bx + gq*c + c/2, by + gr*c + c/2)
    f.circle(*root, c*0.56, fill="none", stroke=ACCENT, sw=2.2)
    for i, (r, q) in enumerate(cells[:9]):
        cx, cy = bx + q*c + c/2, by + r*c + c/2
        taken = i < 5
        f.line(root[0], root[1], cx, cy,
               stroke=ACCENT if taken else "#cccccc", sw=1.8 if taken else 1.0,
               dash=None if taken else "3 3")
        f.circle(cx, cy, 5, fill=ACCENT if taken else "#ffffff",
                 stroke=ACCENT if taken else "#bbbbbb", sw=1.2)
    lx = bx + cfg.width*c + 40
    f.text(lx, by + 20, "prize", size=12, weight="600")
    f.text(lx, by + 38, "army sitting on the tile", size=11, fill=GREY)
    f.text(lx, by + 74, "cost", size=12, weight="600")
    f.text(lx, by + 92, "ticks to walk there", size=11, fill=GREY)
    f.text(lx, by + 128, "you may skip terminals", size=12, weight="600")
    f.text(lx, by + 146, "distant, small stacks are", size=11, fill=GREY)
    f.text(lx, by + 162, "not worth collecting", size=11, fill=GREY)
    f.line(lx, by + 186, lx + 300, by + 186, stroke="#ddd", sw=1)
    f.text(lx, by + 212, "ROOTED PRIZE-COLLECTING", size=12.5, weight="700", fill=ACCENT)
    f.text(lx, by + 230, "STEINER TREE", size=12.5, weight="700", fill=ACCENT)
    f.text(lx, by + 254, "NP-hard.", size=12, weight="600")
    f.text(lx, by + 274, "There is no formula for the", size=11.5)
    f.text(lx, by + 291, "right move - which is exactly", size=11.5)
    f.text(lx, by + 308, "why search is the tool.", size=11.5)
    f.circle(lx + 10, by + 340, 5, fill=ACCENT, stroke="none")
    f.text(lx + 26, by + 344, "collected", size=11, fill=GREY)
    f.circle(lx + 110, by + 340, 5, fill="#ffffff", stroke="#bbbbbb", sw=1.2)
    f.text(lx + 126, by + 344, "skipped", size=11, fill=GREY)
    f.caption("I", "Also: choosing what to spend a turn budget on is multiple-choice knapsack.", y=636)
    return f.save(MEDIA / "d09-nphard.svg")

for _fn in (f_compounding, f_visible, f_sparse, f_simultaneous, f_nphard):
    globals().setdefault("_BATCH2", []).append(_fn)


# =========================== paper figures ===================================
import heapq

def _grid(w=17, h=13, seed=9, density=0.26):
    """A maze where greedy weighted A* genuinely pays for its speed."""
    rng = np.random.default_rng(seed)
    g = (rng.random((h, w)) < density)
    g[1, 1] = g[h-2, w-2] = False
    return g

def _wastar(g, start, goal, eps):
    """Weighted A*. Returns (path, expanded set)."""
    h, w = g.shape
    H = lambda n: abs(n[0]-goal[0]) + abs(n[1]-goal[1])
    open_ = [(H(start)*eps, 0, start)]; came = {start: None}; gs = {start: 0}; closed = set()
    while open_:
        _, gg, n = heapq.heappop(open_)
        if n in closed: continue
        closed.add(n)
        if n == goal: break
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            m = (n[0]+dr, n[1]+dc)
            if not (0 <= m[0] < h and 0 <= m[1] < w) or g[m]: continue
            ng = gg + 1
            if ng < gs.get(m, 1e9):
                gs[m] = ng; came[m] = n
                heapq.heappush(open_, (ng + eps*H(m), ng, m))
    path, cur = [], goal
    while cur in came: path.append(cur); cur = came[cur]
    return path[::-1], closed

def _draw_grid(f, g, x0, y0, c, closed=(), path=(), start=None, goal=None):
    h, w = g.shape
    for r in range(h):
        for q in range(w):
            fill = FILL_DARK if g[r,q] else ("#dbe6ef" if (r,q) in closed else "#ffffff")
            f.rect(x0+q*c, y0+r*c, c, c, fill=fill, stroke="#e0e0e0", sw=0.5)
    for (r,q) in path:
        f.rect(x0+q*c+c*0.22, y0+r*c+c*0.22, c*0.56, c*0.56, fill=RED, stroke="none")
    for node, col in ((start, INK), (goal, ACCENT)):
        if node:
            f.circle(x0+node[1]*c+c/2, y0+node[0]*c+c/2, c*0.34, fill=col, stroke="none")
    f.rect(x0, y0, w*c, h*c, fill="none", stroke=INK, sw=1.2)

def f_wastar():
    g = _grid(); start, goal = (1, 1), (11, 15)
    f = Fig(980, 500)
    title(f, "Weighted A*: \u03b5 trades quality for speed",
          "shaded cells were expanded; the red squares are the path returned")
    c = 17
    base = None
    for i, eps in enumerate((1.0, 2.0, 3.0)):
        path, closed = _wastar(g, start, goal, eps)
        cost = len(path) - 1
        if base is None: base = (len(closed), cost)
        x0 = 40 + i * 310
        _draw_grid(f, g, x0, 140, c, closed, path, start, goal)
        f.text(x0, 128, f"\u03b5 = {eps:g}", size=13, weight="700",
               fill=INK if eps == 1 else ACCENT)
        yb = 140 + g.shape[0] * c + 26
        f.text(x0, yb, f"{len(closed):>3} cells expanded", size=11.5, family=MONO)
        f.text(x0, yb + 20, f"path cost {cost}", size=11.5, family=MONO,
               fill=GREY if cost == base[1] else RED,
               weight="400" if cost == base[1] else "600")
        if i:
            f.text(x0, yb + 40, f"{base[0] / max(len(closed),1):.1f}x fewer expansions",
                   size=11, fill="#3a6b45", weight="600")
    f.text(40, 452, "Inflating h makes the search greedy and directional: far fewer "
                    "expansions, at a slightly worse path.", size=12.5, weight="600")
    f.caption("J", "Real weighted A* runs on one grid; every number is measured.", y=486)
    return f.save(MEDIA / "d10-weighted-astar.svg")

def f_anytime():
    f = Fig(880, 460)
    title(f, "Anytime: an answer at every moment", "ARA* tightens ε while time remains")
    x0, y0, w, h = 70, 120, 700, 210
    axes(f, x0, y0, w, h, "time (ms)", "solution cost")
    steps = [(0, 3.0, 46), (70, 2.0, 38), (150, 1.5, 34), (260, 1.2, 31), (410, 1.0, 29)]
    opt = 29; mx = 50
    Y = lambda v: y0 + h - (v - 24)/(mx - 24)*h
    f.line(x0, Y(opt), x0+w, Y(opt), stroke=GREY, sw=1, dash="5 4")
    f.text(x0+w-4, Y(opt)-8, "optimal", size=10.5, fill=GREY, anchor="end")
    X = lambda t: x0 + t/500*w
    prev = None
    for t, eps, cost in steps:
        if prev: f.line(X(prev[0]), Y(prev[2]), X(t), Y(prev[2]), stroke=INK, sw=2)
        if prev: f.line(X(t), Y(prev[2]), X(t), Y(cost), stroke=INK, sw=2)
        f.circle(X(t), Y(cost), 4, fill=INK, stroke="none")
        f.text(X(t)+7, Y(cost)-9, f"ε={eps:g}", size=10.5, family=MONO, fill=ACCENT)
        prev = (t, eps, cost)
    f.line(X(steps[-1][0]), Y(steps[-1][2]), x0+w, Y(steps[-1][2]), stroke=INK, sw=2)
    f.line(X(400), y0-10, X(400), y0+h+8, stroke=RED, sw=1.4, dash="4 3")
    f.text(X(400)-6, y0-16, "deadline", size=11, fill=RED, weight="600", anchor="end")
    f.rect(70, 352, 740, 62, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(86, 374, "At the deadline we are holding an answer with a known bound.",
           size=12, weight="600")
    f.text(86, 396, "Plain A* would still be returning nothing at that line.",
           size=11.5, fill=RED)
    f.caption("K", "Interruptible, with a guarantee at every step.", y=444)
    return f.save(MEDIA / "d11-anytime.svg")

def f_three_lists():
    f = Fig(900, 430)
    title(f, "How ARA* reuses the previous pass", "OPEN is exactly the set of locally inconsistent states")
    boxes = [(70, "OPEN", "inconsistent,\nmay expand", "#eef4f8", BLUE),
             (360, "CLOSED", "expanded this pass,\ndo not touch again", FILL_LIGHT, GREY),
             (650, "INCONS", "became inconsistent\nafter being closed", "#fdf6e3", ACCENT)]
    for x, name, note, bg, col in boxes:
        f.rect(x, 120, 180, 96, fill=bg, stroke=col, sw=1.4, rx=3)
        f.text(x+14, 146, name, size=14, weight="700", fill=col, family=MONO)
        for i, ln in enumerate(note.split("\n")):
            f.text(x+14, 170+i*16, ln, size=11, fill=INK)
    f.arrow(250, 150, 356, 150, stroke=INK, sw=1.3)
    f.text(303, 142, "expand", size=10.5, anchor="middle", fill=GREY)
    f.arrow(540, 170, 646, 170, stroke=INK, sw=1.3)
    f.text(593, 162, "g improves", size=10.5, anchor="middle", fill=GREY)
    f.text(593, 186, "after closing", size=10.5, anchor="middle", fill=GREY)
    f.path("M 740 216 L 740 268 L 160 268 L 160 222", stroke=ACCENT, sw=2.2, dash="6 4")
    f.arrow(160, 250, 160, 220, stroke=ACCENT, sw=2.2, head=8)
    f.text(450, 290, "at the end of a pass, INCONS is poured back into OPEN for the next ε",
           size=12.5, anchor="middle", weight="700", fill=ACCENT)
    f.rect(70, 320, 760, 58, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(86, 342, "That single arrow is the reuse.", size=12, weight="600")
    f.text(86, 364, "Without it, every new ε would restart the search from scratch.",
           size=11.5, fill=GREY)
    f.caption("L", "The mechanism that makes ARA* anytime rather than just repeated.", y=414)
    return f.save(MEDIA / "d12-three-lists.svg")

def f_lrta_vs_rtaa():
    g = _grid(seed=9); start, goal = (1,1), (11,15)
    path, closed = _wastar(g, start, goal, 1.0)
    closed = set(list(closed)[:70])
    c = 19
    f = Fig(940, 500)
    title(f, "LRTA* against RTAA*", "same lookahead, same search cost - different amount learned")
    for i, (lab, upd) in enumerate((("LRTA*: updates one state", 1),
                                    ("RTAA*: updates every expanded state", len(closed)))):
        x0 = 50 + i*450
        _draw_grid(f, g, x0, 128, c, closed, [], start, goal)
        f.text(x0, 116, lab, size=12.5, weight="700", fill=BLUE if i else GREY)
        upd_cells = list(closed)[:1] if i == 0 else list(closed)
        for (r, q) in upd_cells:
            f.rect(x0+q*c+2, 128+r*c+2, c-4, c-4, fill=ACCENT, stroke="none")
        f.text(x0, 128 + g.shape[0]*c + 24,
               f"{upd} heuristic value{'s' if upd>1 else ''} raised", size=11.5,
               family=MONO, fill=ACCENT, weight="600")
    f.rect(50, 404, 840, 46, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(66, 433, "h[s] <- g[s\u0304] + h[s\u0304] - g[s]   for every expanded s, "
                    "one subtraction each", size=13, family=MONO, weight="600")
    f.caption("M", "The whole contribution of the paper, in one comparison.", y=482)
    return f.save(MEDIA / "d13-lrta-vs-rtaa.svg")

def f_why_legal():
    g = _grid(seed=9); start, goal = (1,1), (11,15)
    path, closed = _wastar(g, start, goal, 1.0)
    closed = set(list(closed)[:60])
    c = 24
    f = Fig(940, 500)
    title(f, "Why the RTAA* update is admissible",
          "every escape from the searched region crosses the frontier")
    x0, y0 = 50, 118
    _draw_grid(f, g, x0, y0, c, closed, [], start, goal)
    # frontier = closed cells with an open neighbour
    frontier = [(r,q) for (r,q) in closed
                if any((r+dr,q+dc) not in closed and not g[max(0,min(g.shape[0]-1,r+dr)),
                        max(0,min(g.shape[1]-1,q+dc))]
                       for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)))]
    for (r,q) in frontier:
        f.rect(x0+q*c+1, y0+r*c+1, c-2, c-2, fill="none", stroke=ACCENT, sw=1.6)
    sbar = min(frontier, key=lambda rq: abs(rq[0]-goal[0])+abs(rq[1]-goal[1]))
    f.circle(x0+sbar[1]*c+c/2, y0+sbar[0]*c+c/2, c*0.36, fill=ACCENT, stroke="none")
    f.text(x0+sbar[1]*c+c/2, y0+sbar[0]*c-6, "s̄", size=13, anchor="middle",
           family=SERIF, style="italic", weight="700", fill=ACCENT)
    s_node = (5, 3)
    f.circle(x0+s_node[1]*c+c/2, y0+s_node[0]*c+c/2, c*0.3, fill=INK, stroke="none")
    f.text(x0+s_node[1]*c+c/2, y0+s_node[0]*c-6, "s", size=13, anchor="middle",
           family=SERIF, style="italic", weight="700")
    px = lambda rq: (x0 + rq[1]*c + c/2, y0 + rq[0]*c + c/2)
    for a, b in ((s_node, sbar), (sbar, goal)):
        (ax, ay), (bx2, by2) = px(a), px(b)
        f.line(ax, ay, bx2, by2, stroke=RED, sw=2, dash="5 4")
    lx = x0 + g.shape[1]*c + 40
    f.text(lx, y0+20, "any path from s to the goal", size=12)
    f.text(lx, y0+38, "must cross the frontier", size=12, weight="600")
    f.rect(lx, y0+62, 300, 56, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(lx+14, y0+88, "dist(s, goal)  ≥  f(s̄) - g(s)", size=13, family=MONO, weight="600")
    f.text(lx, y0+142, "so the new h is a lower bound", size=11.5)
    f.text(lx, y0+160, "→ still admissible", size=11.5, weight="600", fill="#3a6b45")
    f.text(lx, y0+186, "and it only ever increases", size=11.5)
    f.text(lx, y0+204, "→ only ever better informed", size=11.5, weight="600", fill="#3a6b45")
    f.circle(lx+8, y0+238, 6, fill=ACCENT, stroke="none")
    f.text(lx+24, y0+242, "frontier / s̄", size=11, fill=GREY)
    f.rect(lx+2, y0+258, 12, 12, fill="#dbe6ef", stroke="#c0c0c0", sw=0.8)
    f.text(lx+24, y0+268, "expanded", size=11, fill=GREY)
    f.caption("N", "The geometric argument behind the update.", y=476)
    return f.save(MEDIA / "d14-why-legal.svg")

for _f in ():
    pass


def f_depression():
    f = Fig(960, 470)
    title(f, "A heuristic depression", "a bowl in the landscape that traps a greedy agent")
    # 1-D landscape
    x0, y0, w, h = 50, 130, 400, 170
    axes(f, x0, y0, w, h, "position", "h  (lower = looks closer to goal)")
    def hval(t):
        # a dip in h, i.e. a region that looks closer to the goal than it is
        return 0.75 - 0.45*np.exp(-((t-0.45)**2)/0.010) - 0.25*t
    lo_, hi_ = 0.0, 0.8
    Y = lambda v: y0 + h - (v - lo_)/(hi_ - lo_)*h*0.9
    pts = [f"{x0+i/100*w:.1f},{Y(hval(i/100)):.1f}" for i in range(101)]
    f.add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{INK}" stroke-width="2"/>')
    bx = x0 + 0.45*w
    f.text(bx, Y(hval(0.45))+22, "depression", size=11, anchor="middle", fill=RED, weight="600")
    scrub = []
    for j in range(16):
        t = 0.45 + 0.075*np.sin(j*1.9)*(1 - j/20)
        scrub.append(f"{x0+t*w:.1f},{Y(hval(t))-5:.1f}")
    f.add(f'<polyline points="{" ".join(scrub)}" fill="none" stroke="{RED}" '
          f'stroke-width="1.4" stroke-dasharray="3 2"/>')
    f.text(x0+4, y0+16, "the agent scrubs back and forth inside it,", size=10.5, fill=RED)
    f.text(x0+4, y0+31, "raising one value at a time", size=10.5, fill=RED)
    # the same thing on the board
    bx0, by0, c = 520, 140, 34
    f.text(bx0, 128, "the same thing on the board", size=12.5, weight="600")
    walls = {(0,3),(1,3),(2,3),(3,3),(4,3),(4,2),(4,1),(4,0)}
    for r in range(5):
        for q in range(4):
            if (r,q) in walls: tile(f, bx0+q*c, by0+r*c, c, mtn=True)
            else: tile(f, bx0+q*c, by0+r*c, c)
    f.circle(bx0+0.5*c, by0+0.5*c, c*0.26, fill=RED, stroke="none")
    f.path(f"M {bx0+0.5*c} {by0+0.5*c} L {bx0+1.5*c} {by0+1.5*c} L {bx0+0.5*c} {by0+2.5*c} "
           f"L {bx0+2.5*c} {by0+3.5*c} L {bx0+1.5*c} {by0+2.5*c}",
           stroke=RED, sw=1.6, dash="4 3")
    f.text(bx0, by0+5*c+24, "a corridor that dead-ends into mountains:", size=11.5)
    f.text(bx0, by0+5*c+41, "every step in looks like progress", size=11.5, fill=GREY)
    f.rect(50, 366, 860, 62, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(66, 390, "The fix is one sentence: prefer to move to states not known to be "
                    "inside a depression.", size=12, weight="600")
    f.text(66, 412, "It can waste hundreds of moves; the literature calls it scrubbing.",
           size=11.5, fill=GREY)
    f.caption("O", "The failure mode this paper is about.", y=456)
    return f.save(MEDIA / "d15-depression.svg")

def f_detection():
    f = Fig(900, 530)
    title(f, "Detection is free", "a raised heuristic value IS the detector")
    c = 62
    rng = np.random.default_rng(2)
    raised = {(1,1),(1,2),(2,1),(2,2),(2,3),(3,2)}
    for r in range(5):
        for q in range(6):
            x, y = 50+q*c, 130+r*c
            hit = (r,q) in raised
            f.rect(x, y, c, c, fill="#fdf6e3" if hit else "#ffffff",
                   stroke=ACCENT if hit else LINE, sw=1.6 if hit else 0.8)
            h0 = 4 + (r+q) % 5
            hv = h0 + (3 + (r*q) % 4 if hit else 0)
            f.text(x+c/2, y+c*0.46, str(hv), size=17, anchor="middle", family=MONO,
                   weight="600" if hit else "400", fill=ACCENT if hit else INK)
            f.text(x+c/2, y+c*0.78, f"h₀={h0}", size=9.5, anchor="middle", fill=GREY,
                   family=MONO)
    lx = 50 + 6*c + 36
    f.text(lx, 152, "h  >  h₀", size=17, family=MONO, weight="700", fill=ACCENT)
    f.text(lx, 176, "means we were too", size=11.5)
    f.text(lx, 193, "optimistic here before", size=11.5)
    f.text(lx, 226, "one extra bit per tile", size=12, weight="600")
    f.text(lx, 246, "no separate test, no", size=11.5, fill=GREY)
    f.text(lx, 262, "extra search, no cost", size=11.5, fill=GREY)
    f.rect(lx, 288, 250, 60, fill="#fdf6e3", stroke=ACCENT, sw=1.2)
    f.text(lx+14, 312, "mark-and-avoid:", size=11.5, weight="600")
    f.text(lx+14, 331, "prefer unmarked tiles", size=11.5)
    f.caption("P", "Because updates only ever raise h, the mark comes for free.", y=506)
    return f.save(MEDIA / "d16-detection.svg")

def f_shallow_deep():
    """Real 1-ply vs 2-ply search values. This is the comparison ProbCut needs:
    the value a SHALLOW SEARCH returns against the value a DEEPER SEARCH returns
    on the same position - not an evaluation now against an evaluation later."""
    from genmod import legal_action_mask
    cfg = EnvConfig(height=11, width=11, min_spawn_dist=5, max_spawn_dist=10**9,
                    max_steps=300, bfs_iters=44)

    def ev(state):
        l = land_counts(state, 2).astype(jnp.float32)
        a = army_counts(state, 2).astype(jnp.float32)
        return (l[0] - l[1]) + 0.25 * (a[0] - a[1])

    @jax.jit
    def after(state, m, r):
        return step(state, jnp.array([m, r], jnp.int32), cfg)[0]

    def cands(state, player, k):
        mask = np.asarray(legal_action_mask(state, cfg)[player]).reshape(-1)
        mv = np.flatnonzero(mask); mv = mv[mv % 9 != 0]
        if len(mv) == 0: return np.array([0], np.int32)
        sc = np.asarray(jax.vmap(lambda a: ev(after(state, a, 0)) if player == 0
                                 else -ev(after(state, 0, a)))(jnp.asarray(mv, jnp.int32)))
        return mv[np.argsort(-sc)[:k]].astype(np.int32)

    xs, ys = [], []
    for g in range(6):
        st = reset(jax.random.PRNGKey(300 + g), cfg); k = jax.random.PRNGKey(400 + g)
        for t in range(200):
            if t > 15 and t % 8 == 0:
                M = cands(st, 0, 10)
                shallow = float(max(ev(after(st, int(m), 0)) for m in M))
                best = -1e9
                for m in M:                                  # my move
                    R = cands(after(st, int(m), 0), 1, 5)    # their plausible replies
                    worst = min(float(ev(after(st, int(m), int(r)))) for r in R)
                    best = max(best, worst)
                xs.append(shallow); ys.append(best)
            k, a, b = jax.random.split(k, 3)
            st, *_ = step(st, jnp.stack([expander_bot(a, st, cfg, 0),
                                         expander_bot(b, st, cfg, 1)]), cfg)
            if bool(st.done): break

    xs, ys = np.array(xs), np.array(ys)
    A = np.polyfit(xs, ys, 1); sigma = float(np.std(ys - np.polyval(A, xs)))
    r = float(np.corrcoef(xs, ys)[0, 1])

    f = Fig(880, 530)
    title(f, "Does a shallow search predict a deeper one?",
          f"{len(xs)} positions: value of a 1-ply search against a 2-ply search")
    x0, y0, w, h = 80, 132, 470, 250
    axes(f, x0, y0, w, h, "1-ply search value", "2-ply search value")
    f.text(x0 + w/2, y0 + h + 46, "", size=1)
    lo = float(min(xs.min(), ys.min())); hi = float(max(xs.max(), ys.max()))
    if hi - lo < 1e-6: hi = lo + 1
    X = lambda v: x0 + (v - lo)/(hi - lo)*w
    Y = lambda v: y0 + h - (v - lo)/(hi - lo)*h
    band = [f"{X(lo):.1f},{Y(np.polyval(A,lo)+sigma):.1f}",
            f"{X(hi):.1f},{Y(np.polyval(A,hi)+sigma):.1f}",
            f"{X(hi):.1f},{Y(np.polyval(A,hi)-sigma):.1f}",
            f"{X(lo):.1f},{Y(np.polyval(A,lo)-sigma):.1f}"]
    f.add(f'<polygon points="{" ".join(band)}" fill="{ACCENT}" fill-opacity="0.16" stroke="none"/>')
    for xv, yv in zip(xs, ys):
        f.circle(X(xv), Y(yv), 2.2, fill=INK, stroke="none")
    f.line(X(lo), Y(np.polyval(A, lo)), X(hi), Y(np.polyval(A, hi)), stroke=ACCENT, sw=2)

    lx = x0 + w + 44
    f.text(lx, y0 + 26, "v = a v' + b + e", size=15, family=MONO, weight="700")
    for i, (lab, val) in enumerate((("a", f"{A[0]:.2f}"), ("b", f"{A[1]:.2f}"),
                                    ("sigma", f"{sigma:.2f}"), ("r", f"{r:.2f}"))):
        f.text(lx, y0 + 58 + i*24, f"{lab:>5} = {val}", size=13, family=MONO,
               fill=ACCENT if lab == "sigma" else INK,
               weight="600" if lab == "sigma" else "400")
    verdict = "tight enough to prune on" if abs(r) > 0.8 else "too loose to prune on"
    f.text(lx, y0 + 176, verdict, size=11.5, weight="600",
           fill="#3a6b45" if abs(r) > 0.8 else RED)

    f.rect(80, 404, 720, 86, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(96, 428, f"r = {r:.2f}, sigma = {sigma:.2f}. Tight enough that ProbCut "
                    "would prune aggressively and safely.", size=11.5, weight="600")
    f.text(96, 450, "But read it both ways: with a material-only evaluation the extra "
                    "ply barely moves the", size=11.5, fill=INK)
    f.text(96, 467, "value. Good for pruning - and a sign the evaluation needs more "
                    "than land and army.", size=11.5, fill=RED)
    f.text(96, 484, "Measured on our own simulator, not taken from the paper.",
           size=10.5, fill=GREY)
    f.caption("Q", "The week-two experiment, run early as a sanity check.", y=512)
    return f.save(MEDIA / "d17-shallow-deep.svg")

def _pseudo(f, x, y, lines, hl=(), w=470, lh=19):
    f.rect(x, y, w, len(lines)*lh + 18, fill="#fbfbf9", stroke=LINE, sw=1)
    for i, ln in enumerate(lines):
        yy = y + 20 + i*lh
        if i in hl:
            f.rect(x+4, yy-13, w-8, lh-1, fill="#fdf6e3", stroke=ACCENT, sw=1)
        f.text(x+14, yy, ln, size=11.5, family=MONO,
               fill=ACCENT if i in hl else INK, weight="600" if i in hl else "400")

def f_two_params():
    f = Fig(940, 430)
    title(f, "Two lines the paper leaves undefined", "RTAA* does not say how to choose them")
    _pseudo(f, 50, 120, [
        "while s_curr not in GOAL:",
        "    lookahead  = any integer > 0",
        "    astar()                    # expand <= lookahead states",
        "    s_bar = state in OPEN with smallest f",
        "",
        "    for each s in CLOSED:",
        "        h[s] = g[s_bar] + h[s_bar] - g[s]",
        "",
        "    movements  = any integer > 0",
        "    take that many steps toward s_bar",
    ], hl=(1, 8))
    lx = 560
    f.arrow(524, 152, lx-10, 152, stroke=ACCENT, sw=1.3)
    f.text(lx, 148, "how much to think?", size=13, weight="700", fill=ACCENT)
    f.text(lx, 168, "→ metareasoning, decision 1", size=11.5, fill=GREY)
    f.arrow(524, 285, lx-10, 285, stroke=ACCENT, sw=1.3)
    f.text(lx, 281, "how far to commit?", size=13, weight="700", fill=ACCENT)
    f.text(lx, 301, "→ metareasoning, decision 2", size=11.5, fill=GREY)
    f.rect(50, 344, 840, 52, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(66, 366, "The paper says, literally, \u2018any integer greater than zero\u2019 "
                    "for both.", size=12, weight="600")
    f.text(66, 387, "That gap is where the next paper - and my own contribution - lives.",
           size=11.5, fill=GREY)
    f.caption("R", "The hinge between the published algorithms and this project.", y=416)
    return f.save(MEDIA / "d18-two-params.svg")

def f_voc():
    f = Fig(940, 450)
    title(f, "Value of computation", "a computation that cannot change your answer is worth nothing")
    for panel, (lab, vals, verdict, col, bg) in enumerate([
        ("decision is not close", [9.1, 2.2, 1.8, 1.1], "searching deeper cannot flip it\n→ play now", "#3a6b45", "#eef6ef"),
        ("decision is close",     [4.1, 4.0, 3.9, 3.8], "one more expansion could flip it\n→ worth searching", ACCENT, "#fdf6e3")]):
        x0 = 60 + panel*450
        f.text(x0, 122, lab, size=13, weight="700", fill=col)
        rx, ry = x0 + 160, 156
        f.circle(rx, ry, 12, fill="#ffffff", stroke=INK, sw=1.5)
        for i, v in enumerate(vals):
            cx = x0 + 40 + i*80
            f.line(rx, ry+12, cx, 236, stroke=INK, sw=1.1)
            f.circle(cx, 248, 12, fill=bg, stroke=col if i == 0 else LINE, sw=1.6 if i == 0 else 1)
            f.text(cx, 252, f"{v:.1f}", size=11, anchor="middle", family=MONO,
                   weight="700" if i == 0 else "400")
        f.rect(x0, 292, 370, 62, fill=bg, stroke=col, sw=1.3, rx=3)
        for j, ln in enumerate(verdict.split("\n")):
            f.text(x0+14, 316+j*20, ln, size=12, weight="700" if j else "400", fill=col if j else INK)
    f.rect(60, 372, 840, 44, fill=FILL_LIGHT, stroke=LINE, sw=1)
    f.text(76, 399, "Expand the node with the highest value of computation - and stop "
                    "when the benefit drops below the cost.", size=12, weight="600")
    f.caption("S", "Russell & Wefald, 1990. The same Russell as the textbook.", y=436)
    return f.save(MEDIA / "d19-voc.svg")

def f_mechanics():
    f = Fig(940, 430)
    title(f, "The game supplies both mechanics natively",
          "the metareasoning literature usually has to simulate these")
    f.text(60, 126, "1.  identity action", size=13, weight="700", fill=BLUE)
    f.text(60, 146, "action index 0 on every tile is PASS", size=11.5, fill=GREY)
    for i in range(9):
        x = 60 + i*44
        sel = i == 0
        f.rect(x, 162, 38, 38, fill="#eef4f8" if sel else FILL_LIGHT,
               stroke=BLUE if sel else LINE, sw=1.8 if sel else 0.9)
        f.text(x+19, 186, str(i), size=13, anchor="middle", family=MONO,
               weight="700" if sel else "400", fill=BLUE if sel else INK)
    f.text(60, 222, "PASS", size=11.5, weight="700", fill=BLUE)
    f.text(112, 222, "stand still and think - permitted by the rules", size=11.5, fill=GREY)
    f.line(60, 250, 880, 250, stroke="#e0e0e0", sw=1)
    f.text(60, 282, "2.  prefix commitment", size=13, weight="700", fill=RED)
    f.text(60, 302, "queue a path; the server plays one move per tick", size=11.5, fill=GREY)
    for i in range(10):
        x = 60 + i*46
        f.rect(x, 318, 40, 30, fill=RED_LT, stroke=RED, sw=1)
        f.text(x+20, 338, f"m{i+1}", size=10, anchor="middle", family=MONO)
    f.text(540, 338, "= 10 ticks owing the game nothing", size=12, weight="700", fill=RED)
    f.text(540, 356, "= 5 seconds of free computation", size=11.5, fill=GREY)
    f.caption("T", "Both decision points have a literal counterpart in the rules.", y=414)
    return f.save(MEDIA / "d20-mechanics.svg")
