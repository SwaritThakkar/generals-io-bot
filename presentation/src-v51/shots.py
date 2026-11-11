"""Game screenshots for the v51 talk, drawn with the arena's own board painter."""
import os
import pickle
import sys
import pathlib

os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pygame  # noqa: E402
from arena.board_draw import BRASS, BG, BoardPainter, BoardView, PLAYER, lighten  # noqa: E402

pygame.init()
HERE = pathlib.Path(__file__).parent
OUT = HERE / "shots"
OUT.mkdir(exist_ok=True)
D = pickle.load(open(HERE / "game.pkl", "rb"))
F = {f["t"]: f for f in D["frames"]}
M = D["meta"]
H, W = M["h"], M["w"]
mountain = np.array(M["mountain"], bool)
castle = np.array(M["castle"], bool)
SIZE = 1100


def decode(a):
    if not a:
        return None
    cell, k = divmod(a, 9)
    return cell, (k - 1) % 4, k >= 5


def path_cells(cells, src, dst):
    cells, order, seen = set(cells), [src], {src}
    while order[-1] != dst:
        r, c = divmod(order[-1], W)
        best = None
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            v = rr * W + cc
            if 0 <= rr < H and 0 <= cc < W and v in cells and v not in seen:
                score = abs(rr - dst // W) + abs(cc - dst % W)
                if best is None or score < best[0]:
                    best = (score, v)
        if best is None:
            break
        order.append(best[1])
        seen.add(best[1])
    return order


def shot(t, name, moves=True, walks=0, plan=False, rings=(), crop=None, us_walks=0):
    f = F[t]
    owner = np.array(f["owner"])
    army = np.array(f["army"])
    gen = np.zeros(H * W, bool)
    for g in f["gen"]:
        if g >= 0:
            gen[g] = True
    lm = []
    if moves:
        for a, p in ((f["a"], 0), (f["b"], 1)):
            d = decode(a)
            if d:
                lm.append((d[0], d[1], d[2], p))
    view = BoardView(H, W, owner, army, mountain, castle, gen, last_moves=lm)
    surf = pygame.Surface((SIZE, SIZE))
    surf.fill(BG)
    painter = BoardPainter(H, W)
    painter.layout(pygame.Rect(0, 0, SIZE, SIZE))
    painter.draw(surf, view)
    over = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    g = f["gen"][0]
    if walks:
        for d, x, src, ps in sorted(f["them"], key=lambda w: -w[1])[:walks]:
            pts = [painter.centre(c) for c in path_cells(ps, src, g)]
            if len(pts) > 1:
                pygame.draw.lines(over, (*lighten(PLAYER[1], .45), 235), False, pts, 7)
                pygame.draw.circle(over, (*lighten(PLAYER[1], .45), 255), pts[0], 9)
    if us_walks:
        for d, x, src, ps in sorted(f["us"], key=lambda w: -w[1])[:us_walks]:
            pts = [painter.centre(c) for c in path_cells(ps, src, g)]
            if len(pts) > 1:
                pygame.draw.lines(over, (*lighten(PLAYER[0], .45), 220), False, pts, 6)
    if plan:
        for child, par in f["def_moves"]:
            x0, y0 = painter.centre(child)
            x1, y1 = painter.centre(par)
            pygame.draw.line(over, (*BRASS, 245), (x0, y0), (x0 + (x1 - x0) * .78, y0 + (y1 - y0) * .78), 6)
            ang = np.arctan2(y1 - y0, x1 - x0)
            tip = (x0 + (x1 - x0) * .92, y0 + (y1 - y0) * .92)
            hl = painter.cell * .32
            wings = [(tip[0] - hl * np.cos(ang - .55), tip[1] - hl * np.sin(ang - .55)),
                     (tip[0] - hl * np.cos(ang + .55), tip[1] - hl * np.sin(ang + .55))]
            pygame.draw.polygon(over, (*BRASS, 255), [tip, *wings])
    surf.blit(over, (0, 0))
    for cell, col in rings:
        pygame.draw.rect(surf, col, painter.cell_rect(cell).inflate(8, 8), 5)
    if crop:
        r0, c0, r1, c1 = crop
        cs = painter.cell
        rect = pygame.Rect(painter.origin[0] + c0 * cs, painter.origin[1] + r0 * cs, (c1 - c0) * cs, (r1 - r0) * cs)
        surf = surf.subsurface(rect).copy()
    pygame.image.save(surf, str(OUT / f"{name}.png"))
    return f


def biggest_enemy(f):
    return max(((f["army"][c], c) for c in range(H * W) if f["owner"][c] == 1 and c != f["gen"][1]), default=(0, -1))[1]


BR = BRASS
g0 = F[219]["gen"][0]
shot(120, "t120")
shot(40, "t40")
shot(205, "t205")
shot(213, "t213", rings=[(biggest_enemy(F[213]), BR), (F[213]["gen"][1], BR)])
shot(215, "t215", rings=[(biggest_enemy(F[215]), BR)])
shot(219, "t219_walks", walks=3, rings=[(g0, BR)])
shot(219, "t219_plan", plan=True, rings=[(g0, BR), (biggest_enemy(F[219]), BR)])
shot(224, "t224", plan=True, rings=[(biggest_enemy(F[224]), BR)])
shot(230, "t230", plan=True, rings=[(biggest_enemy(F[230]), BR)])
shot(235, "t235", rings=[(g0, BR), (biggest_enemy(F[235]), BR)])
shot(237, "t237", rings=[(g0, BR)])
shot(298, "t298")
shot(405, "t405", rings=[(F[405]["gen"][1], BR)])
shot(407, "t407", rings=[(F[407]["gen"][1], BR)])
shot(150, "t150_us", us_walks=3, walks=2, rings=[(F[150]["gen"][0], BR)])


# a 1 x 10 strip for "walking costs army"
def strip(name, step):
    h, w = 1, 10
    owner = np.full(10, -1)
    army = np.zeros(10, int)
    owner[9], army[9] = 0, 12 + step // 2
    if step == 0:
        owner[0], army[0] = 1, 30
    else:
        owner[:step], army[:step] = 1, 1
        owner[step], army[step] = 1, 30 - step
    gen = np.zeros(10, bool)
    gen[9] = True
    view = BoardView(h, w, owner, army, np.zeros(10, bool), np.zeros(10, bool), gen)
    surf = pygame.Surface((1100, 110))
    surf.fill(BG)
    p = BoardPainter(h, w)
    p.layout(pygame.Rect(0, 0, 1100, 110))
    p.draw(surf, view)
    pygame.image.save(surf, str(OUT / f"{name}.png"))


strip("strip0", 0)
strip("strip8", 8)
print("shots:", sorted(x.name for x in OUT.iterdir()))
