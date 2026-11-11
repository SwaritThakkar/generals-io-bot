import pathlib
"""Human vs v37 frames with the bot drawn red (as in the rest of the deck) and the human blue."""
import os, sys, pathlib
import numpy as np
os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pygame
from arena.board_draw import BoardPainter, BoardView, BG, BRASS
pygame.init()
SH = pathlib.Path(__file__).parent / "shots"
h = np.load(ROOT / "models/tournaments/human_games/20261103_121122_seed538794140_human0.npz", allow_pickle=True)
gens = [int(np.where(h["owner"][0] == q)[0][0]) for q in (0, 1)]
for t in (520, 845, 884):
    owner, army = h["owner"][t].copy(), h["army"][t]
    sw = owner.copy(); sw[owner == 0], sw[owner == 1] = 1, 0
    gen = np.zeros(529, bool)
    for g in gens:
        if owner[g] >= 0: gen[g] = True
    mv = []
    for p, act in enumerate(h["moves"][t]):
        act = int(act)
        if act > 0:
            c, k = divmod(act, 9); mv.append((c, (k - 1) % 4, k >= 5, 1 - p))
    v = BoardView(23, 23, sw, army, h["mountain"].reshape(-1).astype(bool), h["castle"][t].astype(bool), gen, last_moves=mv)
    surf = pygame.Surface((1100, 1100)); surf.fill(BG)
    bp = BoardPainter(23, 23); bp.layout(pygame.Rect(0, 0, 1100, 1100)); bp.draw(surf, v)
    big = max(range(529), key=lambda c: army[c] if owner[c] == 0 else -1)
    pygame.draw.rect(surf, BRASS, bp.cell_rect(big).inflate(8, 8), 5)
    if owner[gens[1]] >= 0: pygame.draw.rect(surf, BRASS, bp.cell_rect(gens[1]).inflate(8, 8), 5)
    pygame.image.save(surf, str(SH / f"human_{t}.png"))
    print(t, "human biggest", army[big], "bot general", army[gens[1]], "land", [(owner == q).sum() for q in (0, 1)])
