import pathlib
"""Screenshots of v11 losing to a castle rusher, and v36 holding the same map. Our bot drawn red."""
import os, pickle, sys, pathlib
os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pygame
from arena.board_draw import BRASS, BG, BoardPainter, BoardView
pygame.init()
HERE = pathlib.Path(__file__).parent
OUT = HERE / "shots"
H = W = 23


def decode(a):
    if not a:
        return None
    cell, k = divmod(a, 9)
    return cell, (k - 1) % 4, k >= 5


def shot(name, d, t, rings=()):
    f = next(x for x in d["frames"] if x["t"] == t)
    seat = d["seat"]
    owner = np.array(f["owner"])
    sw = owner.copy()
    sw[owner == seat], sw[owner == 1 - seat] = 0, 1           # our bot red
    gen = np.zeros(H * W, bool)
    for g in f["gen"]:
        if g >= 0:
            gen[g] = True
    lm = []
    for a, p in ((f["a"], 0), (f["b"], 1)):
        dd = decode(a)
        if dd:
            lm.append((dd[0], dd[1], dd[2], 0 if p == seat else 1))
    view = BoardView(H, W, sw, np.array(f["army"]), np.array(d["mountain"], bool), np.array(d["castle"], bool), gen, last_moves=lm)
    surf = pygame.Surface((1100, 1100)); surf.fill(BG)
    p = BoardPainter(H, W); p.layout(pygame.Rect(0, 0, 1100, 1100)); p.draw(surf, view)
    for c in rings:
        pygame.draw.rect(surf, BRASS, p.cell_rect(c).inflate(8, 8), 5)
    pygame.image.save(surf, str(OUT / f"{name}.png"))
    o = 1 - seat
    big = max(((f["army"][c], c) for c in range(529) if f["owner"][c] == o and c != f["gen"][o]), default=(0, -1))
    mybig = max((f["army"][c], c) for c in range(529) if f["owner"][c] == seat)
    print(name, t, "gen", f["army"][f["gen"][seat]], "their big", big, "my big", mybig)
    return f


v11 = pickle.load(open(HERE / "v11loss_v11.pkl", "rb"))
v36 = pickle.load(open(HERE / "v11loss_v36.pkl", "rb"))
for t in (76, 78, 80, 82, 84):
    f = next(x for x in v11["frames"] if x["t"] == t)
    print(t, "v11 gen", f["army"][f["gen"][1]])
g11 = v11["frames"][0]["gen"][1]
og = v11["frames"][0]["gen"][0]
shot("v11_80", v11, 80, rings=[g11])
shot("v11_82", v11, 82, rings=[g11, 87])
f = next(x for x in v11["frames"] if x["t"] == 85)
big = max((f["army"][c], c) for c in range(529) if f["owner"][c] == 0 and c != f["gen"][0])[1]
shot("v11_85", v11, 85, rings=[g11, big])
f = next(x for x in v11["frames"] if x["t"] == 98)
big = max((f["army"][c], c) for c in range(529) if f["owner"][c] == 0 and c != f["gen"][0])[1]
shot("v11_98", v11, 98, rings=[g11, big])
f = next(x for x in v36["frames"] if x["t"] == 98)
g36 = f["gen"][1]
big = max((f["army"][c], c) for c in range(529) if f["owner"][c] == 0 and c != f["gen"][0])[1]
shot("v36_98", v36, 98, rings=[g36, big])


def trail(name, d, t0, t1):
    """Board at t1 with the path of each side's biggest stack from t0 to t1."""
    from arena.board_draw import PLAYER, lighten
    shot(name, d, t1, rings=[d["frames"][0]["gen"][d["seat"]]])
    surf = pygame.image.load(str(OUT / f"{name}.png"))
    p = BoardPainter(H, W); p.layout(pygame.Rect(0, 0, 1100, 1100))
    over = pygame.Surface((1100, 1100), pygame.SRCALPHA)
    seat = d["seat"]
    for side, col in ((seat, PLAYER[0]), (1 - seat, PLAYER[1])):
        g = d["frames"][0]["gen"][side]
        pts = []
        for f in d["frames"]:
            if t0 <= f["t"] <= t1:
                c = max((f["army"][c], c) for c in range(529) if f["owner"][c] == side and (c != g or f["t"] == t0 - 1))[1]
                pts.append(p.centre(c))
        pygame.draw.lines(over, (*lighten(col, .5), 235), False, pts, 8)
        pygame.draw.circle(over, (*lighten(col, .5), 255), pts[-1], 11)
    surf.blit(over, (0, 0))
    pygame.image.save(surf, str(OUT / f"{name}.png"))


trail("v11_trail", v11, 82, 98)
