"""Board painter shared by the arena app and the tutorial animation.

Same palette and glyphs as viewer.py (and the web client), so what players see
in the tutorial is exactly what they see in a game.

    painter = BoardPainter(h=23, w=23)
    painter.layout(pygame.Rect(x, y, width, height))     # fits square cells
    painter.draw(surface, board)                          # board: BoardView
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pygame

# --------------------------------------------------------------------------- #
# palette (identical to viewer.py)
# --------------------------------------------------------------------------- #
BG = (11, 16, 18)
SURFACE = (18, 26, 29)
LINE = (36, 48, 54)
INK = (226, 233, 234)
INK2 = (157, 176, 180)
INK3 = (107, 125, 130)
BRASS = (215, 162, 68)
PLAYER = [(209, 84, 74), (74, 134, 198), (99, 169, 106), (154, 122, 200)]
NEUTRAL_LAND = (43, 58, 63)
NEUTRAL_CASTLE = (72, 91, 97)
MOUNTAIN = (22, 32, 36)
GRID = (30, 41, 46)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))     # up, down, left, right


def lighten(colour, amount: float) -> tuple[int, int, int]:
    return tuple(int(v + (255 - v) * amount) for v in colour)


def pick_font(size: int, mono: bool = False, bold: bool = False):
    name = ("menlo,dejavusansmono,couriernew,monospace" if mono
            else "helveticaneue,helvetica,arial,dejavusans,sans")
    path = pygame.font.match_font(name, bold=bold)
    return pygame.font.Font(path, size) if path else pygame.font.SysFont(None, size, bold=bold)


@dataclass
class BoardView:
    """Everything needed to paint one tick. Arrays are flat, length h*w."""
    h: int
    w: int
    owner: np.ndarray                  # -1 neutral, 0/1 players
    army: np.ndarray
    mountain: np.ndarray               # bool
    castle: np.ndarray                 # bool (castle cells, owned or not)
    general: np.ndarray                # bool (cells holding a living general)
    #: moves just played: list of (src_cell, dir 0..3, half, player)
    last_moves: list = field(default_factory=list)
    #: the human's queued moves: list of (src_cell, dir, half)
    queued: list = field(default_factory=list)
    selection: int = -1
    #: extra outlines for the tutorial: list of (cell, rgb)
    highlights: list = field(default_factory=list)


class BoardPainter:
    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.cell = 20
        self.origin = (0, 0)
        self.f_army = pick_font(9, mono=True, bold=True)

    # -- geometry ------------------------------------------------------------
    def layout(self, rect: pygame.Rect) -> None:
        """Fit square cells into rect, centred."""
        self.cell = max(8, min(rect.width // self.w, rect.height // self.h))
        bw, bh = self.cell * self.w, self.cell * self.h
        self.origin = (rect.x + (rect.width - bw) // 2, rect.y + (rect.height - bh) // 2)
        self.f_army = pick_font(max(8, int(self.cell * 0.44)), mono=True, bold=True)

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.origin[0], self.origin[1], self.cell * self.w, self.cell * self.h)

    def cell_rect(self, i: int) -> pygame.Rect:
        r, c = divmod(i, self.w)
        return pygame.Rect(self.origin[0] + c * self.cell, self.origin[1] + r * self.cell,
                           self.cell, self.cell)

    def centre(self, i: int) -> tuple[int, int]:
        rc = self.cell_rect(i)
        return rc.centerx, rc.centery

    def cell_at(self, pos) -> int:
        x, y = pos
        c = (x - self.origin[0]) // self.cell
        r = (y - self.origin[1]) // self.cell
        if 0 <= r < self.h and 0 <= c < self.w:
            return int(r) * self.w + int(c)
        return -1

    def neighbour(self, i: int, d: int) -> int:
        r, c = divmod(i, self.w)
        rr, cc = r + DIRS[d][0], c + DIRS[d][1]
        if 0 <= rr < self.h and 0 <= cc < self.w:
            return rr * self.w + cc
        return -1

    # -- glyphs ----------------------------------------------------------------
    def _overlay(self, surface, i, draw) -> None:
        surf = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
        draw(surf)
        surface.blit(surf, self.cell_rect(i).topleft)

    def crown(self, surface, i, rgba) -> None:
        c = self.cell / 2
        s = self.cell * 0.28
        pts = [(c - s, c + s * 0.55), (c - s, c - s * 0.35), (c - s * 0.42, c + s * 0.1),
               (c, c - s * 0.75), (c + s * 0.42, c + s * 0.1), (c + s, c - s * 0.35),
               (c + s, c + s * 0.55)]
        self._overlay(surface, i, lambda surf: pygame.draw.polygon(surf, rgba, pts))

    def castle_ring(self, surface, i, rgba) -> None:
        c = int(self.cell / 2)
        self._overlay(surface, i, lambda surf: pygame.draw.circle(
            surf, rgba, (c, c), int(self.cell * 0.27), max(1, int(self.cell * 0.09))))

    def arrow(self, surface, src, direction, half, colour, scale=1.0, lift=0.55, progress=1.0) -> None:
        """A move arrow; `progress` < 1 draws it partway (for animation)."""
        dst = self.neighbour(src, direction)
        if dst < 0:
            return
        x0, y0 = self.centre(src)
        x1, y1 = self.centre(dst)
        x1 = x0 + (x1 - x0) * progress
        y1 = y0 + (y1 - y0) * progress
        bright = lighten(colour, lift)
        shaft = max(3, int(self.cell * 0.14 * scale))
        head = self.cell * 0.34 * scale
        mx, my = x0 + (x1 - x0) * 0.62, y0 + (y1 - y0) * 0.62
        tip = (x0 + (x1 - x0) * 0.94, y0 + (y1 - y0) * 0.94)
        ang = np.arctan2(y1 - y0, x1 - x0) if (x1, y1) != (x0, y0) else 0.0
        wings = [(mx - head * np.cos(ang - 0.6), my - head * np.sin(ang - 0.6)),
                 (mx - head * np.cos(ang + 0.6), my - head * np.sin(ang + 0.6))]
        pygame.draw.line(surface, (0, 0, 0), (x0, y0), (mx, my), shaft + 4)
        pygame.draw.line(surface, bright, (x0, y0), (mx, my), shaft)
        pygame.draw.polygon(surface, bright, [tip, *wings])
        pygame.draw.polygon(surface, (0, 0, 0), [tip, *wings], 2)
        if half:
            pygame.draw.circle(surface, (0, 0, 0), (int(x0), int(y0)), max(3, int(self.cell * 0.13)))
            pygame.draw.circle(surface, (255, 255, 255), (int(x0), int(y0)), max(2, int(self.cell * 0.10)))

    # -- frame -----------------------------------------------------------------
    def draw(self, surface, b: BoardView) -> None:
        for i in range(b.h * b.w):
            rc = self.cell_rect(i)
            owner = int(b.owner[i])
            mountain = bool(b.mountain[i])
            is_castle = bool(b.castle[i])
            is_general = bool(b.general[i])
            if mountain:
                colour = MOUNTAIN
            elif owner >= 0:
                colour = PLAYER[owner % len(PLAYER)]
            elif is_castle or is_general:
                colour = NEUTRAL_CASTLE
            else:
                colour = NEUTRAL_LAND
            pygame.draw.rect(surface, colour, rc)
            pygame.draw.rect(surface, GRID, rc, 1)
            if mountain:
                continue
            if is_general:
                self.crown(surface, i, (255, 255, 255, 90) if owner >= 0 else (0, 0, 0, 70))
            elif is_castle:
                self.castle_ring(surface, i, (255, 255, 255, 82) if owner >= 0 else (*INK3, 170))
            army = int(b.army[i])
            if army:
                label = self.f_army.render(str(army), True, (255, 255, 255) if owner >= 0 else INK)
                shadow = self.f_army.render(str(army), True, (0, 0, 0))
                pos = label.get_rect(center=self.centre(i))
                surface.blit(shadow, pos.move(1, 1))
                surface.blit(label, pos)
        for src, d, half, p in b.last_moves:
            self.arrow(surface, src, d, half, PLAYER[p % len(PLAYER)])
        for src, d, half in b.queued:
            self.arrow(surface, src, d, half, BRASS, 0.8, lift=0.12)
        if b.selection >= 0:
            pygame.draw.rect(surface, BRASS, self.cell_rect(b.selection), 3)
        for cell, rgb in b.highlights:
            pygame.draw.rect(surface, rgb, self.cell_rect(cell).inflate(2, 2), 3)
