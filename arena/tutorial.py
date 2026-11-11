"""Animated rules + controls tutorial for the arena.

Every board in here is drawn with `arena.board_draw.BoardPainter`, and every tick
is produced by `models.sim36.Sim.step2` on a small hand-built map with scripted
actions, so what the tutorial shows is exactly what the rules engine does.
Like the arena (arena/session.py), a general capture ends the game on the board
sim36 leaves: the fallen general loses its crown and nothing else changes.
(`genmod/env.py` also hands the loser's land to the captor at half strength;
set APPLY_ENV_AFTERMATH to show that instead. The game is over either way.)

    tut = Tutorial(pygame.Rect(0, 0, 1280, 800), chapter="rules")   # or "controls"
    tut.handle(event); tut.update(dt_ms); tut.draw(surface); tut.done

    python -m arena.tutorial                  # window: rules, then controls
    python -m arena.tutorial --export DIR     # DIR/rules.mp4, DIR/controls.mp4
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
import pygame

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena.board_draw import (BG, BRASS, INK, INK2, INK3, LINE, PLAYER, SURFACE,  # noqa: E402
                              BoardPainter, BoardView, pick_font)
from models.sim36 import Board, Sim  # noqa: E402

TICK_MS = 333            # ~3 sim ticks per second
TICK_LIMIT = 600         # arena default game length (arena/app.py LENGTHS = 300/600/1000)
LOOK = (240, 240, 240)   # "look here" outline
DIR_OF = {"U": 0, "D": 1, "L": 2, "R": 3}
NAMES = ("Red", "Blue")
#: arena/session.py skips an illegal queued move and tries the next one in the
#: same tick. (viewer.py / server/local_server.py drop the rest of the queue.)
DROP_REST_ON_ILLEGAL = False
#: see module docstring
APPLY_ENV_AFTERMATH = False


def act(cell: int, d: int, half: bool = False) -> int:
    return cell * 9 + 1 + d + (4 if half else 0)


@dataclass
class Frame:
    owner: np.ndarray
    army: np.ndarray
    castle: np.ndarray
    general: np.ndarray
    step: int


@dataclass
class Beat:
    kind: str                      # "hold" or "tick"
    ms: float
    caption: str
    pre: Frame
    post: Frame | None = None
    moves: list = field(default_factory=list)      # (src, dir, half, player)
    q_pre: list = field(default_factory=list)      # queued (src, dir, half)
    q_post: list = field(default_factory=list)
    sel_pre: int = -1
    sel_post: int = -1
    hl: list = field(default_factory=list)          # (cell, rgb)
    cursor_from: object = None     # cell, "off" or None (no cursor)
    cursor_to: object = None
    click: bool = False
    keys: tuple = ()               # keycaps shown, e.g. ("shift", "R")
    key_note: str = ""
    result: str = ""
    result_colour: tuple = INK
    legend: int = -1               # legend row to emphasise
    active_control: int = -1       # control row to emphasise
    actions: tuple = (0, 0)
    score_hl: bool = False


# --------------------------------------------------------------------------- #
# scene scripting
# --------------------------------------------------------------------------- #
class Script:
    """A scene: a small map plus a list of beats produced by real sim steps.

    Map tokens (whitespace separated, one string per row):
        .  empty neutral land     #  mountain      c40  neutral castle, garrison 40
        R3 / B3  red / blue tile  RG5 / BG5  general   RC5 / BC5  owned castle
    """

    def __init__(self, title: str, rows: list[str], step: int = 0, limit: int = TICK_LIMIT,
                 panel: str = "", show_limit: bool = False):
        grid = [r.split() for r in rows]
        self.h, self.w = len(grid), len(grid[0])
        n = self.h * self.w
        owner = np.full(n, -1, int)
        army = np.zeros(n, int)
        mountain = np.zeros(n, bool)
        castle = np.zeros(n, bool)
        gens = [-1, -1]
        for r, row in enumerate(grid):
            assert len(row) == self.w, (title, r)
            for c, tok in enumerate(row):
                i = r * self.w + c
                if tok == ".":
                    continue
                if tok == "#":
                    mountain[i] = True
                elif tok[0] == "c":
                    castle[i] = True
                    army[i] = int(tok[1:])
                else:
                    p = "RB".index(tok[0])
                    owner[i] = p
                    rest = tok[1:]
                    if rest[0] == "G":
                        gens[p] = i
                        rest = rest[1:]
                    elif rest[0] == "C":
                        castle[i] = True
                        rest = rest[1:]
                    army[i] = int(rest)
        assert min(gens) >= 0, f"{title}: both generals needed"
        self.title = title
        self.mountain = mountain
        self.castle = castle.copy()
        self.alive = [True, True]
        self.sim = Sim(Board(mountain, castle, self.h, self.w), owner, army, gens, step, limit)
        self.limit = limit
        self.panel = panel            # "", "legend", "score", "controls"
        self.show_limit = show_limit
        self.caption = ""
        self.beats: list[Beat] = []
        # human input model (mirrors viewer.py + server queue)
        self.queue: list[int] = []
        self.sel = -1
        self.trail: list[int] = []
        self.cursor = None

    # -- helpers ---------------------------------------------------------------
    def cell(self, r: int, c: int) -> int:
        return r * self.w + c

    def frame(self) -> Frame:
        gen = np.zeros(self.h * self.w, bool)
        for p in (0, 1):
            if self.alive[p]:
                gen[self.sim.gen[p]] = True
        return Frame(np.array(self.sim.owner), np.array(self.sim.army), self.castle.copy(), gen,
                     self.sim.step)

    def _hl(self, hl) -> list:
        out = []
        for item in hl or ():
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
                out.append((self.cell(*item), LOOK))
            else:
                (r, c), rgb = item
                out.append((self.cell(r, c), rgb))
        return out

    def _queued(self) -> list:
        return [(a // 9, (a % 9 - 1) % 4, (a % 9 - 1) >= 4) for a in self.queue]

    def _action(self, spec) -> int:
        if spec is None:
            return 0
        if isinstance(spec, int):
            return spec
        r, c, d = spec[:3]
        half = len(spec) > 3 and spec[3] == "h"
        return act(self.cell(r, c), DIR_OF[d], half)

    def say(self, text: str) -> "Script":
        self.caption = text
        return self

    def totals(self, frame: Frame | None = None):
        f = frame or self.frame()
        land = [int((f.owner == p).sum()) for p in (0, 1)]
        army = [int(f.army[f.owner == p].sum()) for p in (0, 1)]
        return land, army

    # -- beats -----------------------------------------------------------------
    def hold(self, ms: float, hl=None, **kw) -> "Script":
        f = self.frame()
        cur = kw.pop("cursor", None)
        b = Beat("hold", ms, self.caption, f, q_pre=self._queued(), q_post=self._queued(),
                 sel_pre=self.sel, sel_post=self.sel, hl=self._hl(hl),
                 cursor_from=self.cursor, cursor_to=cur if cur is not None else self.cursor, **kw)
        if cur is not None:
            self.cursor = cur
        self.beats.append(b)
        return self

    def tick(self, red=None, blue=None, ms: float = TICK_MS, hl=None, **kw) -> "Script":
        sim = self.sim
        pre = self.frame()
        q_pre, sel_pre = self._queued(), self.sel
        a0 = self._action(red)
        if red is None and self.queue:
            while self.queue:
                a = self.queue.pop(0)
                if sim.legal(0, a):
                    a0 = a
                    break
                if DROP_REST_ON_ILLEGAL:
                    self.queue.clear()
        a1 = self._action(blue)
        moves = []
        for p, a in ((0, a0), (1, a1)):
            if a and sim.legal(p, a):
                k = a % 9 - 1
                moves.append((a // 9, k % 4, k >= 4, p))
        before_done = sim.done
        sim.step2(a0, a1)
        result, colour = "", INK
        if sim.done and not before_done:
            result, colour = self._finish()
        self._reconcile()
        b = Beat("tick", ms, self.caption, pre, self.frame(), moves, q_pre, self._queued(),
                 sel_pre, self.sel, self._hl(hl), cursor_from=self.cursor, cursor_to=self.cursor,
                 actions=(a0, a1), result=kw.pop("result", result), result_colour=colour, **kw)
        self.beats.append(b)
        return self

    def ticks(self, n: int, ms: float = TICK_MS, hl=None, **kw) -> "Script":
        for _ in range(n):
            self.tick(ms=ms, hl=hl, **kw)
        return self

    def _finish(self):
        """Game over. Apply genmod.env's capture aftermath that sim36 skips."""
        sim = self.sim
        t = sim.step
        owner, army = sim.owner, sim.army
        captured = [owner[sim.gen[p]] != p for p in (0, 1)]
        if any(captured) and not APPLY_ENV_AFTERMATH:
            for p in (0, 1):
                if captured[p]:
                    self.alive[p] = False
        elif any(captured):
            for p in (0, 1):
                if not captured[p]:
                    continue
                captor = owner[sim.gen[p]]
                for i in range(len(owner)):
                    if owner[i] == p:
                        owner[i] = captor
                        army[i] = (army[i] + 1) // 2
                self.castle[sim.gen[p]] = True
                self.alive[p] = False
            producers = list(np.flatnonzero(self.castle))
            producers += [sim.gen[p] for p in (0, 1) if self.alive[p]]
            if t % sim.gil == 0:
                for c in set(producers):
                    if owner[c] >= 0:
                        army[c] += 1
            if t % sim.land_int == 0:
                for c in range(len(owner)):
                    if owner[c] >= 0:
                        army[c] += 1
        if any(captured):
            if sim.winner < 0:
                return "Draw: both generals fell", INK
            return f"{NAMES[sim.winner]} wins: general captured", PLAYER[sim.winner]
        land, tot = sim.totals()
        if sim.winner < 0:
            return "Draw: equal land and army", INK
        how = "land" if land[0] != land[1] else "army"
        return f"{NAMES[sim.winner]} wins on {how}", PLAYER[sim.winner]

    # -- the human's input (same logic as viewer.py) ---------------------------
    def _reconcile(self) -> None:
        if self.sel < 0 or self.queue:
            return
        owner = self.sim.owner
        if owner[self.sel] == 0:
            return
        while self.trail and owner[self.trail[-1]] != 0:
            self.trail.pop()
        self.sel = self.trail[-1] if self.trail else -1

    def click(self, r: int, c: int, ms: float = 1100, hl=None, **kw) -> "Script":
        target = self.cell(r, c)
        pre_sel, q_pre = self.sel, self._queued()
        start = self.cursor
        if self.sim.owner[target] == 0:
            self.queue.clear()
            self.sel = target
            self.trail = [target]
        f = self.frame()
        self.beats.append(Beat("hold", ms, self.caption, f, q_pre=q_pre, q_post=self._queued(),
                               sel_pre=pre_sel, sel_post=self.sel, hl=self._hl(hl),
                               cursor_from=start, cursor_to=target, click=True,
                               active_control=0, **kw))
        self.cursor = target
        return self

    def press(self, key: str, shift: bool = False, ms: float = 550, hl=None, **kw) -> "Script":
        """key: U/D/L/R (arrow keys) or C."""
        q_pre, sel_pre = self._queued(), self.sel
        control = 3
        if key == "C":
            self.queue.clear()
            self.sel = -1
            self.trail = []
        elif self.sel >= 0:
            control = 2 if shift else 1
            d = DIR_OF[key]
            r, c = divmod(self.sel, self.w)
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[d]
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.h and 0 <= cc < self.w and not self.mountain[rr * self.w + cc]:
                self.queue.append(act(self.sel, d, shift))
                self.sel = rr * self.w + cc
                self.trail.append(self.sel)
        keys = ("shift", key) if shift else (key,)
        self.beats.append(Beat("hold", ms, self.caption, self.frame(), q_pre=q_pre,
                               q_post=self._queued(), sel_pre=sel_pre, sel_post=self.sel,
                               hl=self._hl(hl), cursor_from=self.cursor, cursor_to=self.cursor,
                               keys=keys, active_control=control, **kw))
        return self


# --------------------------------------------------------------------------- #
# chapters
# --------------------------------------------------------------------------- #
def rules_scenes() -> list[Script]:
    S = []

    s = Script("The board", [
        ".  .   .   .   #   .   .   .   .",
        ".  R1  R1  .   #   .   B1  .   .",
        ".  R2  RG5 R1  .   .   BG5 B2  .",
        ".  .   R1  .   c45 .   B1  .   .",
        "#  .   .   .   .   #   .   .   .",
        ".  .   #   .   .   .   .   .   .",
        ".  .   .   .   #   .   .   .   .",
    ], panel="legend")
    s.say("Two players share one board. Both see all of it, all the time. "
          "Real games use a 23 x 23 board.").hold(3000)
    s.say("Here you are red and the bot is blue. Each number is the army on that tile.") \
        .hold(1700, hl=[(2, 1), (1, 1), (1, 2), (2, 3), (3, 2)], legend=0) \
        .hold(1700, hl=[(2, 7), (1, 6), (3, 6)], legend=1)
    s.say("The crown is a general: one per player. The ring is a castle.") \
        .hold(1700, hl=[(2, 2), (2, 6)], legend=2).hold(1700, hl=[(3, 4)], legend=3)
    s.say("Dark tiles are mountains: nothing can enter them. "
          "Grey tiles are neutral: nobody owns them yet.") \
        .hold(1700, hl=[(0, 4), (1, 4), (4, 5)], legend=4) \
        .hold(1700, hl=[(5, 4), (4, 3), (0, 6)], legend=5)
    S.append(s)

    s = Script("Generals make army", [
        ".  .   .   .   .   .   .",
        ".  .   .   .   .   .   .",
        ".  RG1 .   .   .   BG1 .",
        ".  .   .   .   .   .   .",
        ".  .   .   .   .   .   .",
    ])
    s.say("The game runs in ticks. Every 2 ticks each general gains +1 army.") \
        .hold(2200, hl=[(2, 1), (2, 5)])
    s.ticks(12, ms=420, hl=[(2, 1), (2, 5)])
    s.say("12 ticks later: 1 + 6 = 7 on each general.").hold(2400, hl=[(2, 1), (2, 5)])
    S.append(s)

    s = Script("Moving", [
        ".  .   .   .   .   .   BG1",
        ".  .   .   .   .   .   .",
        ".  RG8 .   .   .   .   .",
        ".  .   .   .   .   .   .",
        ".  .   .   .   .   .   .",
    ])
    s.say("Each tick you make one move: pick one of your tiles and a direction "
          "(up, down, left or right).").hold(2600, hl=[(2, 1)])
    s.say("A move takes all the army except 1. 8 leaves 1 behind and 7 arrive.") \
        .tick(red=(2, 1, "R"), ms=1800)
    s.say("Next tick, the 7 move on and leave 1. "
          "(Meanwhile the general made +1, so it shows 2.)").tick(red=(2, 2, "R"), ms=2200)
    s.tick(red=(2, 3, "R"), ms=900)
    s.say("A tile with only 1 army cannot move.").hold(2200, hl=[(2, 2), (2, 3)])
    S.append(s)

    s = Script("Send half", [
        ".   .   .   .   .   .   BG1",
        ".   .   .   .   .   .   .",
        ".   .   R10 .   .   .   .",
        ".   .   .   .   .   .   .",
        "RG1 .   .   .   .   .   .",
    ], step=1)
    s.say("You can also send half (rounded down). The white dot marks a half move.") \
        .hold(2000, hl=[(2, 2)])
    s.say("10 sends 5 and keeps 5.").tick(red=(2, 2, "R", "h"), ms=2200)
    s.say("5 sends 2 and keeps 3.").tick(red=(2, 2, "U", "h"), ms=2200)
    s.hold(1600)
    S.append(s)

    s = Script("Taking land", [
        ".   .   .   .   .   .   .   BG1",
        ".   .   .   .   .   .   .   .",
        ".   R6  .   .   .   .   .   .",
        ".   .   .   .   .   .   .   .",
        "RG1 .   .   .   .   .   .   .",
    ], step=1, panel="score")
    s.say("Grey land is empty (0 army). Move onto it and it becomes yours.").hold(2000)
    s.say("A stack of 6 walks right, leaving 1 on every tile it passes.") \
        .tick(red=(2, 1, "R"), ms=900)
    for c in range(2, 6):
        s.tick(red=(2, c, "R"), ms=650)
    s.say("6 army took 5 new tiles. The tiles you own are your land.") \
        .hold(2600, hl=[(2, c) for c in range(2, 7)], score_hl=True)
    S.append(s)

    s = Script("Fighting", [
        ".   .   .   .   .   .   BG1",
        ".   R9  B5  .   .   .   .",
        ".   .   .   .   .   .   .",
        ".   R5  B4  .   .   .   .",
        "RG1 .   .   .   .   .   .",
    ], step=1)
    s.say("Moving onto an enemy tile is a fight: attackers minus defenders.") \
        .hold(2000, hl=[(1, 1), (1, 2)])
    s.say("8 attack 5: 8 - 5 = 3. More attackers, so the tile turns red with 3.") \
        .tick(red=(1, 1, "R"), ms=2600, hl=[(1, 2)])
    s.say("4 attack 4: 4 - 4 = 0. On a tie the defender keeps the tile (now with 0).") \
        .tick(red=(3, 1, "R"), ms=2800, hl=[(3, 2)])
    s.hold(1200, hl=[(3, 2)])
    S.append(s)

    s = Script("Castles", [
        ".   .   .   .   .   .   .   BG1",
        ".   .   .   .   .   .   .   .",
        ".   .   R30 c40 .   .   .   .",
        ".   .   .   R20 .   .   .   .",
        "RG1 .   .   .   .   .   .   .",
    ], step=1)
    s.say("A castle starts with 40 to 50 neutral army. "
          "To take it you must send more than that.").hold(2400, hl=[(2, 3)])
    s.say("29 is not enough: the castle drops to 40 - 29 = 11 and stays grey.") \
        .tick(red=(2, 2, "R"), ms=2600, hl=[(2, 3)])
    s.say("The damage stays. 19 more beat the 11 left: 19 - 11 = 8, and the castle is red.") \
        .tick(red=(3, 3, "U"), ms=2800, hl=[(2, 3)])
    s.say("Now it works for you: +1 army every 2 ticks, like a general.") \
        .ticks(8, ms=420, hl=[(2, 3)])
    s.hold(1200, hl=[(2, 3)])
    S.append(s)

    s = Script("Land income", [
        "B1  B1  B1  .   .   .   .   BG4",
        ".   .   .   .   .   .   .   B1",
        ".   R1  R1  R1  R1  .   .   .",
        ".   R1  .   .   R1  .   .   .",
        "RG4 R1  .   .   .   .   .   .",
    ], step=45, panel="score")
    s.say("Every 50 ticks (tick 50, 100, 150, ...) every tile you own gains +1.").hold(2200)
    s.ticks(4, ms=500)
    owned = [(r, c) for r in range(5) for c in range(8)
             if s.sim.owner[s.cell(r, c)] >= 0]
    s.say("Tick 50: every owned tile got +1. The generals got +2 (+1 general, +1 land).") \
        .tick(ms=3200, hl=owned, score_hl=True)
    s.say("More land means more income, so grab land early.").ticks(2, ms=500).hold(1600)
    S.append(s)

    s = Script("Same time", [
        ".   .   .   .   .   .   BG1",
        ".   .   .   .   .   .   .",
        ".   R10 .   .   .   B7  .",
        ".   .   .   .   .   .   .",
        "RG1 .   .   .   .   .   .",
    ], step=1)
    s.say("Both players move at the same time, every tick. "
          "You only see the bot's move after it happens.").hold(2200)
    s.tick(red=(2, 1, "R"), blue=(2, 5, "L"), ms=2200)
    s.say("Both sent army to the same tile: the bigger force wins by the difference, "
          "8 - 5 = 3.").tick(red=(2, 2, "R"), blue=(2, 4, "L"), ms=3000, hl=[(2, 3)])
    s.hold(1400, hl=[(2, 3)])
    S.append(s)

    s = Script("Winning", [
        ".   .   .   .   .   B1  B1",
        ".   .   .   .   .   B3  .",
        "RG2 .   .   .   R20 BG6 B5",
        ".   .   .   .   .   B2  .",
        ".   .   .   .   .   .   .",
    ], panel="score")
    s.say("Capture the enemy general and you win the game at once.") \
        .hold(2200, hl=[(2, 5)])
    s.say("19 attack 6: blue's general falls and the game ends right away. Red wins.") \
        .tick(red=(2, 4, "R"), ms=4200, hl=[(2, 5)])
    s.say("It works both ways: if your general falls, you lose. Guard it.").hold(2600)
    S.append(s)

    s = Script("Time limit", [
        "R1  R1  R1  .   .   .   BG3",
        "R1  .   .   .   .   B1  B1",
        "RG3 R1  .   .   .   .   B30",
        "R1  R1  .   .   .   .   B1",
        "R1  .   .   .   .   .   B1",
    ], step=TICK_LIMIT - 4, panel="score", show_limit=True)
    s.say(f"If no general falls, the game stops at the tick limit (picked before the game: "
          f"300, 600 or 1000). Here it is {TICK_LIMIT}.").hold(3000)
    s.ticks(3, ms=600)
    land, army = s.totals()
    s.say(f"Then the player with more land wins, even with less army. "
          f"Red has {land[0]} tiles, blue {land[1]}.").tick(ms=4400, score_hl=True)
    s.hold(1200, score_hl=True)
    S.append(s)

    s = Script("Time limit: army", [
        "R1  R1  R1  .   .   .   BG3",
        "R1  .   .   .   .   B1  B1",
        "RG3 R25 .   .   .   .   B9",
        "R1  .   .   .   .   B1  B1",
        ".   .   .   .   .   .   B1",
    ], step=TICK_LIMIT - 3, panel="score", show_limit=True)
    s.say("If land is equal, more total army wins. "
          "If land and army are both equal, it is a draw.").hold(2400)
    s.ticks(2, ms=600)
    s.tick(ms=4200, score_hl=True)
    s.hold(1000, score_hl=True)
    S.append(s)

    s = Script("What good players do", [
        ".   .   .   .   .   .   .   .   .",
        ".   R1  R1  .   .   #   .   B1  .",
        ".   R1  .   .   .   #   .   B1  .",
        "RG3 R12 R10 R11 R14 c40 .   BG9 .",
        ".   R1  .   .   .   .   .   B1  .",
        ".   R1  R1  .   .   #   .   B1  .",
        ".   .   .   .   .   .   .   .   .",
    ], step=1)
    s.say("Expand early: land is income. Then gather many tiles into one big stack.") \
        .hold(2000, hl=[(3, 1), (3, 2), (3, 3), (3, 4)])
    s.tick(red=(3, 1, "R"), ms=800).tick(red=(3, 2, "R"), ms=800) \
        .tick(red=(3, 3, "R"), ms=800)
    s.say("A big stack takes castles: 43 beat the 40 garrison.") \
        .tick(red=(3, 4, "R"), ms=2400, hl=[(3, 5)])
    s.say("Watch the bot's big stacks, and keep army on or near your general.") \
        .hold(3000, hl=[(3, 0), (3, 7)])
    S.append(s)
    return S


CONTROL_MAP = [
    ".  .   .   .   .   .   .   .   .",
    ".  R1  R2  .   .   #   .   .   .",
    ".  R3  RG12 R1 .   #   .   B2  .",
    ".  .   R1  .   .   .   .   BG9 .",
    ".  .   .   .   c42 .   .   B1  .",
    ".  .   #   .   .   .   .   .   .",
    ".  .   .   .   .   .   .   .   .",
]


def controls_scenes() -> list[Script]:
    S = []

    s = Script("Select a tile", CONTROL_MAP, panel="controls")
    s.say("Click one of your own tiles to select it.").hold(1200, cursor="off")
    s.say("Clicking a grey or blue tile does nothing.").click(4, 1, ms=1800)
    s.say("Click one of your own tiles to select it. The brass box shows the selection.") \
        .click(2, 2, ms=2400).hold(2200)
    S.append(s)

    s = Script("Queue a path", CONTROL_MAP, panel="controls")
    s.cursor = "off"
    s.say("Select a tile, then press arrow keys (or W A S D). "
          "Each key adds one move from the box.").click(2, 2, ms=1300)
    s.press("R").press("R")
    s.say("The box moves with each key, so the next key continues the path.").press("D", ms=1400)
    s.say("Your queue plays one move per tick, in order. Each move leaves 1 behind.") \
        .ticks(3, ms=1000)
    s.hold(1500)
    S.append(s)

    s = Script("Send half", CONTROL_MAP, panel="controls")
    s.cursor = "off"
    s.say("Hold Shift with an arrow key to send half instead of all but 1.").click(2, 2, ms=1300)
    s.press("D", shift=True, ms=1800)
    s.say("12 sends 6 and keeps 6 (the tile below had 1, so it shows 7). "
          "The white dot marks a half move.").tick(ms=3000)
    s.hold(1600)
    S.append(s)

    s = Script("Clear the queue", CONTROL_MAP, panel="controls")
    s.cursor = "off"
    s.say("Changed your mind? Press C to clear the whole queue.").click(2, 2, ms=1300)
    s.press("U").press("U").press("R")
    s.press("C", ms=1800)
    s.say("Clicking another tile also drops the old queue and starts a new one.") \
        .click(2, 2, ms=1000).press("L").press("L")
    s.click(1, 2, ms=1800).hold(1600)
    S.append(s)

    s = Script("Blocked moves", CONTROL_MAP, panel="controls")
    s.cursor = "off"
    s.say("A move into a mountain or off the board is not added to the queue.") \
        .click(1, 1, ms=1300).press("R").press("R").press("R")
    s.press("R", ms=1800, hl=[(1, 5)])
    s.say("A queued move that can't be made now (only 1 army, or not your tile) is skipped, "
          "and the next one plays in the same tick.").hold(2600, hl=[(1, 1)])
    s.tick(ms=2600, hl=[(1, 1), (1, 2)])
    s.say("Next tick the last move starts from a tile with only 1, so it is skipped too. "
          "The queue is now empty.") \
        .tick(ms=2600, hl=[(1, 3)])
    s.hold(1400)
    S.append(s)

    s = Script("Reading the panel", CONTROL_MAP, panel="controls")
    s.cursor = "off"
    s.say("The panel shows land (tiles owned) and army (total) for both players.") \
        .click(2, 2, ms=1000).press("R").press("R").press("D")
    s.tick(blue=(3, 7, "L"), ms=700).tick(blue=(3, 6, "L"), ms=700, score_hl=True) \
        .tick(blue=(3, 5, "L"), ms=1400, score_hl=True)
    s.say("Watch the bot's army total and its big stacks. P pauses the game.") \
        .ticks(4, ms=500, score_hl=True).hold(2400, score_hl=True)
    S.append(s)
    return S


CHAPTERS = {"rules": rules_scenes, "controls": controls_scenes}
LEGEND = ["you (red)", "the bot (blue)", "general", "castle", "mountain", "neutral land"]
CONTROLS = [("click", "select a tile"), ("arrows / wasd", "queue a move"),
            ("shift + arrow", "send half"), ("c", "clear the queue"), ("p", "pause")]


# --------------------------------------------------------------------------- #
# playback + drawing
# --------------------------------------------------------------------------- #
class Tutorial:
    def __init__(self, rect: pygame.Rect, chapter: str = "rules"):
        if not pygame.font.get_init():
            pygame.font.init()
        self.rect = pygame.Rect(rect)
        self.chapter = chapter
        self.scenes = CHAPTERS[chapter]()
        self.index = 0
        self.t = 0.0
        self.paused = False
        self.done = False
        self._fonts()
        self._layout_for = None

    # -- control -----------------------------------------------------------
    @property
    def scene(self) -> Script:
        return self.scenes[self.index]

    def duration(self, i: int | None = None) -> float:
        return sum(b.ms for b in self.scenes[self.index if i is None else i].beats)

    def total_duration(self) -> float:
        return sum(self.duration(i) for i in range(len(self.scenes)))

    def goto(self, i: int) -> None:
        if i >= len(self.scenes):
            self.index, self.t, self.done = len(self.scenes) - 1, self.duration(), True
            return
        self.index, self.t, self.done = max(0, i), 0.0, False

    def handle(self, event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif event.key == pygame.K_LEFT:
            self.goto(self.index - 1 if self.t < 1500 else self.index)
        elif event.key == pygame.K_RIGHT:
            self.goto(self.index + 1)
        elif event.key == pygame.K_r:
            self.goto(self.index)

    def update(self, dt_ms: float) -> None:
        if self.paused or self.done:
            return
        self.t += dt_ms
        if self.t >= self.duration():
            if self.index + 1 < len(self.scenes):
                self.t -= self.duration()
                self.index += 1
            else:
                self.t = self.duration()
                self.done = True

    # -- layout ------------------------------------------------------------
    def _fonts(self) -> None:
        k = max(0.6, min(self.rect.height / 800, self.rect.width / 1280))
        self.k = k
        self.f_title = pick_font(int(15 * k), bold=True)
        self.f_cap = pick_font(int(26 * k))
        self.f_body = pick_font(int(16 * k))
        self.f_bold = pick_font(int(16 * k), bold=True)
        self.f_num = pick_font(int(16 * k), mono=True, bold=True)
        self.f_small = pick_font(int(13 * k))
        self.f_key = pick_font(int(15 * k), bold=True)
        self.f_res = pick_font(int(21 * k), bold=True)

    def _layout(self) -> None:
        key = (self.index, tuple(self.rect))
        if self._layout_for == key:
            return
        self._layout_for = key
        s, r, k = self.scene, self.rect, self.k
        pad = int(20 * k)
        top = r.y + pad + int(30 * k)
        area_h = int(r.height * 0.74) - (top - r.y)
        side = int(250 * k) if s.panel else 0
        avail_w = r.width - 2 * pad - 2 * (side + int(28 * k))
        cell = max(12, min(avail_w // s.w, area_h // s.h, int(80 * k)))
        bw, bh = cell * s.w, cell * s.h
        bx = r.x + (r.width - bw) // 2
        by = top + (area_h - bh) // 2
        self.painter = BoardPainter(s.h, s.w)
        self.painter.layout(pygame.Rect(bx, by, bw, bh))
        self.panel_rect = pygame.Rect(bx + bw + int(28 * k), by, side, bh)
        self.cap_y = r.y + int(r.height * 0.76)
        self.mini = BoardPainter(1, 1)

    def _beat(self):
        t = self.t
        beats = self.scene.beats
        for b in beats:
            if t < b.ms:
                return b, max(0.0, t / b.ms)
            t -= b.ms
        return beats[-1], 1.0

    # -- drawing -----------------------------------------------------------
    def draw(self, surface) -> None:
        self._layout()
        old_clip = surface.get_clip()
        surface.set_clip(self.rect)
        surface.fill(BG, self.rect)
        s = self.scene
        b, u = self._beat()
        k = self.k

        # board: a tick shows the arrows being drawn on the old board, then the result
        if b.kind == "tick":
            split = 0.32 if b.moves else 0.0
            if u < split:
                f, moves, q, sel = b.pre, [], b.q_pre, b.sel_pre
            else:
                f, moves, q, sel = b.post, b.moves, b.q_post, b.sel_post
                if b.ms >= 1200 and u > 0.7:      # slow tick: let the result show unobscured
                    moves = []
        else:
            split = 0.0
            after = (not b.click) or u >= 0.62
            if b.keys:
                after = u >= 0.18
            f, moves, q, sel = (b.post or b.pre), [], (b.q_post if after else b.q_pre), \
                (b.sel_post if after else b.sel_pre)
        view = BoardView(s.h, s.w, f.owner, f.army, s.mountain, f.castle, f.general,
                         last_moves=moves, queued=q, selection=sel, highlights=b.hl)
        self.painter.draw(surface, view)
        if b.kind == "tick" and u < split:
            for src, d, half, p in b.moves:
                self.painter.arrow(surface, src, d, half, PLAYER[p], progress=max(0.08, u / split))

        # tick counter
        label = f"tick {f.step}" + (f"  /  {s.limit}" if s.show_limit else "")
        img = self.f_num.render(label, True, INK2)
        surface.blit(img, (self.painter.rect.x, self.painter.rect.y - img.get_height() - int(8 * k)))

        if s.panel:
            self._panel(surface, s, b, f, u)
        if b.cursor_to is not None:
            self._cursor(surface, b, u)
        self._caption(surface, s, b)
        self._footer(surface)
        surface.set_clip(old_clip)

    def _text(self, surface, text, font, colour, x, y):
        img = font.render(text, True, colour)
        surface.blit(img, (x, y))
        return img.get_width(), img.get_height()

    def _spaced(self, surface, text, font, colour, x, y, gap):
        for ch in text:
            img = font.render(ch, True, colour)
            surface.blit(img, (x, y))
            x += img.get_width() + gap
        return x

    def _panel(self, surface, s, b, f, u):
        k = self.k
        pr = self.panel_rect
        x, y = pr.x, pr.y
        if s.panel == "legend":
            size = int(40 * k)
            samples = [(0, 1, False, False), (1, 1, False, False), (0, 5, False, True),
                       (-1, 45, True, False), (-1, 0, False, False), (-1, 0, False, False)]
            for row, (o, a, cas, gen) in enumerate(samples):
                rc = pygame.Rect(x, y, size, size)
                self.mini.layout(rc)
                mtn = np.array([row == 4])
                self.mini.draw(surface, BoardView(1, 1, np.array([o]), np.array([a]), mtn,
                                                  np.array([cas]), np.array([gen])))
                on = b.legend == row
                if on:
                    pygame.draw.rect(surface, LOOK, rc.inflate(2, 2), 3)
                self._text(surface, LEGEND[row], self.f_bold if on else self.f_body,
                           INK if on else INK3, x + size + int(14 * k),
                           y + (size - self.f_body.get_height()) // 2)
                y += size + int(14 * k)
            return
        if s.panel in ("score", "controls"):
            land, army = s.totals(f)
            for p in (0, 1):
                pygame.draw.rect(surface, PLAYER[p], pygame.Rect(x, y + int(4 * k), int(12 * k), int(12 * k)))
                name = "Red (you)" if p == 0 else "Blue (bot)"
                self._text(surface, name, self.f_body, INK, x + int(22 * k), y)
                y += int(22 * k)
                colour = INK if b.score_hl else INK2
                self._text(surface, f"{land[p]} land", self.f_num, colour, x + int(22 * k), y)
                self._text(surface, f"{army[p]} army", self.f_num, colour, x + int(122 * k), y)
                y += int(32 * k)
            if b.score_hl:
                box = pygame.Rect(x - int(8 * k), pr.y - int(8 * k), pr.width, y - pr.y + int(4 * k))
                pygame.draw.rect(surface, LOOK, box, 2)
            y += int(6 * k)
            if b.result and (b.kind == "hold" or u >= 0.32):
                self._text(surface, b.result, self.f_res, b.result_colour, x, y)
                y += int(34 * k)
        if s.panel == "controls":
            y += int(4 * k)
            pygame.draw.line(surface, LINE, (x, y), (x + pr.width - int(10 * k), y))
            y += int(14 * k)
            for row, (key, what) in enumerate(CONTROLS):
                on = b.active_control == row
                self._text(surface, key, self.f_bold, BRASS if on else INK3, x, y)
                self._text(surface, what, self.f_body, INK if on else INK3, x + int(118 * k), y)
                y += int(24 * k)
            if b.keys:
                y += int(18 * k)
                pressed = u < 0.55
                kx = x
                for key in b.keys:
                    kx += self._keycap(surface, key, kx, y, pressed) + int(10 * k)
        elif b.result and s.panel != "score" and (b.kind == "hold" or u >= 0.32):
            self._text(surface, b.result, self.f_res, b.result_colour, x, y)

    def _keycap(self, surface, key, x, y, pressed) -> int:
        k = self.k
        h = int(46 * k)
        w = int(92 * k) if key == "shift" else h
        rc = pygame.Rect(x, y + (int(3 * k) if pressed else 0), w, h)
        pygame.draw.rect(surface, BRASS if pressed else SURFACE, rc, border_radius=int(6 * k))
        pygame.draw.rect(surface, BRASS if pressed else LINE, rc, 2, border_radius=int(6 * k))
        ink = BG if pressed else INK
        if key in DIR_OF:
            cx, cy, a = rc.centerx, rc.centery, h * 0.22
            pts = {"U": [(cx, cy - a), (cx - a, cy + a * 0.7), (cx + a, cy + a * 0.7)],
                   "D": [(cx, cy + a), (cx - a, cy - a * 0.7), (cx + a, cy - a * 0.7)],
                   "L": [(cx - a, cy), (cx + a * 0.7, cy - a), (cx + a * 0.7, cy + a)],
                   "R": [(cx + a, cy), (cx - a * 0.7, cy - a), (cx - a * 0.7, cy + a)]}[key]
            pygame.draw.polygon(surface, ink, pts)
        else:
            img = self.f_key.render(key, True, ink)
            surface.blit(img, img.get_rect(center=rc.center))
        return w

    def _cursor_xy(self, where):
        if where == "off" or where is None:
            r = self.painter.rect
            return r.right + int(60 * self.k), r.bottom - int(20 * self.k)
        cx, cy = self.painter.centre(where)
        return cx + self.painter.cell * 0.24, cy + self.painter.cell * 0.2

    def _cursor(self, surface, b, u):
        x0, y0 = self._cursor_xy(b.cursor_from)
        x1, y1 = self._cursor_xy(b.cursor_to)
        m = min(1.0, u / 0.55) if b.click else 1.0
        m = m * m * (3 - 2 * m)
        x, y = x0 + (x1 - x0) * m, y0 + (y1 - y0) * m
        if b.click and 0.55 <= u < 0.9:
            rad = int(self.painter.cell * (0.15 + 0.35 * (u - 0.55) / 0.35))
            pygame.draw.circle(surface, BRASS, (int(x), int(y)), rad, 3)
        sc = self.k * 1.3
        shape = [(0, 0), (0, 17), (4.5, 13), (7.5, 20), (10.5, 18.8), (7.5, 12), (13, 12)]
        pts = [(x + px * sc, y + py * sc) for px, py in shape]
        pygame.draw.polygon(surface, (255, 255, 255), pts)
        pygame.draw.polygon(surface, (0, 0, 0), pts, 2)

    def _wrap(self, text, font, width):
        lines = []
        for para in text.split("\n"):
            line = ""
            for word in para.split():
                trial = f"{line} {word}".strip()
                if font.size(trial)[0] > width and line:
                    lines.append(line)
                    line = word
                else:
                    line = trial
            lines.append(line)
        return lines

    def _caption(self, surface, s, b):
        k, r = self.k, self.rect
        width = min(r.width - int(80 * k), int(1060 * k))
        x = r.x + (r.width - width) // 2
        y = self.cap_y
        n = len(self.scenes)
        title = f"{self.index + 1} / {n}   {s.title.upper()}"
        self._spaced(surface, title, self.f_title, BRASS, x, y, max(1, int(2 * k)))
        y += self.f_title.get_height() + int(12 * k)
        for line in self._wrap(b.caption, self.f_cap, width)[:3]:
            self._text(surface, line, self.f_cap, INK, x, y)
            y += self.f_cap.get_linesize() + int(2 * k)

    def _footer(self, surface):
        k, r = self.k, self.rect
        pad = int(20 * k)
        y = r.bottom - pad - self.f_small.get_height()
        sy = y - int(12 * k)
        n = len(self.scenes)
        gap = int(6 * k)
        width = r.width - 2 * pad
        seg = (width - gap * (n - 1)) / n
        for i in range(n):
            x0 = r.x + pad + i * (seg + gap)
            bar = pygame.Rect(int(x0), sy, int(seg), max(3, int(4 * k)))
            pygame.draw.rect(surface, LINE, bar)
            if i < self.index:
                pygame.draw.rect(surface, INK3, bar)
            elif i == self.index:
                frac = min(1.0, self.t / max(1.0, self.duration()))
                pygame.draw.rect(surface, BRASS, (bar.x, bar.y, int(bar.w * frac), bar.h))
        chapter = "THE RULES" if self.chapter == "rules" else "THE CONTROLS"
        self._spaced(surface, chapter, self.f_small, INK3, r.x + pad, y, 1)
        hint = "space  pause      left / right  scene      r  replay scene"
        if self.paused:
            hint = "paused      " + hint
        img = self.f_small.render(hint, True, BRASS if self.paused else INK3)
        surface.blit(img, (r.right - pad - img.get_width(), y))


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def export(out_dir: pathlib.Path, size=(1280, 800), fps: int = 30) -> list[pathlib.Path]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    out_dir.mkdir(parents=True, exist_ok=True)
    surf = pygame.Surface(size)
    written = []
    try:
        import imageio.v2 as imageio
        writer_for = lambda p: imageio.get_writer(  # noqa: E731
            p, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
        ext = "mp4"
    except ImportError:
        imageio = None
        ext = "gif"
    for chapter in CHAPTERS:
        tut = Tutorial(pygame.Rect(0, 0, *size), chapter)
        path = out_dir / f"{chapter}.{ext}"
        frames = []
        writer = writer_for(str(path)) if imageio else None
        while True:
            tut.draw(surf)
            arr = pygame.surfarray.array3d(surf).swapaxes(0, 1)
            if writer:
                writer.append_data(np.ascontiguousarray(arr))
            else:
                from PIL import Image
                frames.append(Image.fromarray(arr).convert("P", palette=Image.ADAPTIVE))
            if tut.done:
                break
            tut.update(1000 / fps)
        for _ in range(fps):            # one still second at the end
            if writer:
                writer.append_data(np.ascontiguousarray(arr))
        if writer:
            writer.close()
        else:
            frames[0].save(path, save_all=True, append_images=frames[1::2],
                           duration=int(2000 / fps), loop=0)
        written.append(path)
        print(f"wrote {path}  ({tut.total_duration() / 1000:.1f}s)")
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Rules and controls tutorial.")
    ap.add_argument("--export", metavar="DIR", help="render rules.mp4 and controls.mp4 into DIR")
    args = ap.parse_args(argv)
    if args.export:
        export(pathlib.Path(args.export))
        return
    pygame.init()
    screen = pygame.display.set_mode((1280, 800))
    pygame.display.set_caption("Tutorial")
    rect = screen.get_rect()
    chapters = list(CHAPTERS)
    ci = 0
    tut = Tutorial(rect, chapters[ci])
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            tut.handle(event)
        tut.update(clock.tick(60))
        if tut.done and ci + 1 < len(chapters):
            ci += 1
            tut = Tutorial(rect, chapters[ci])
        tut.draw(screen)
        pygame.display.flip()


if __name__ == "__main__":
    main()
