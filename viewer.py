#!/usr/bin/env python3
"""A native pygame window for watching and playing perfect-information Generals.io.

Runs on the JAX engine, in your real Python environment, so agents that need JAX
work exactly as they do in training.

    # play against the siege-aware search bot
    python viewer.py --p0 human --p1 final.py:act

    # play one yourself
    python viewer.py --p0 human --p1 expander

    # step through a finished game
    python viewer.py --p0 expander --p1 random --tick 0

Controls
Every game is played twice: game 1, then the same map (seed) with the two
seats swapped, so neither side keeps the first-move / position advantage.
Seeds whose map walls a general in (arena/playable.py) are skipped.

    space          pause / resume
    left / right   step back / forward through history (while paused)
    home / end     jump to the start / to live
    [ / ]          slower / faster
    r / n          next game (game 2 of the pair = same map, seats swapped;
                   after game 2, a new pair on a new map)
    esc or q       quit

When you hold a seat, the arrow keys queue moves instead:
    click          select one of your tiles
    arrows / wasd  queue a move (shift = send half)
    c              clear your queued path
    p              pause
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pygame  # noqa: E402

from genmod import EnvConfig                                      # noqa: E402
from genmod.config import DIRS, NUM_CELL_ACTIONS                  # noqa: E402
from genmod.state import army_counts, land_counts                 # noqa: E402
from server.local_server import Match, spec_to_agent              # noqa: E402
from arena.playable import check as map_check                     # noqa: E402

ROOT_DIR = pathlib.Path(__file__).resolve().parent

# Palette shared with the web client, so a screenshot from either looks the same.
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

def lighten(colour, amount: float) -> tuple[int, int, int]:
    return tuple(int(v + (255 - v) * amount) for v in colour)


SIDEBAR = 260
PAD = 14
KEY_DIR = {
    pygame.K_UP: 0, pygame.K_w: 0,
    pygame.K_DOWN: 1, pygame.K_s: 1,
    pygame.K_LEFT: 2, pygame.K_a: 2,
    pygame.K_RIGHT: 3, pygame.K_d: 3,
}


@dataclass
class Frame:
    tick: int
    owner: np.ndarray
    army: np.ndarray
    castle: np.ndarray
    general: np.ndarray
    land: np.ndarray
    army_total: np.ndarray
    last_moves: list
    done: bool
    winner: int


class Viewer:
    def __init__(self, cfg: EnvConfig, specs: list[str], seed: int, tick_ms: int):
        self.cfg = cfg
        self.base_specs = list(specs)    # seats as given on the command line (game 1)
        self.specs = list(specs)
        self.leg = 0                     # 0 = game 1 of the pair, 1 = seats swapped
        self.seed = seed
        self.tick_ms = tick_ms
        self.paused = False
        self.cursor = -1          # index into history; -1 means "live"
        self.history: list[Frame] = []
        self.hist_lock = threading.Lock()
        self.new_match(seed)

    # -- match lifecycle ---------------------------------------------------
    def save_game(self) -> None:
        """Write the finished/abandoned game to models/tournaments/human_games/
        (npz: owner/army per tick, both players' moves) so losses to a human
        can be replayed and analysed. Only games with a human seat are kept."""
        if getattr(self, "_saved", True) or self.human < 0:
            return
        with self.hist_lock:
            frames = list(self.history)
        if len(frames) < 20:
            return
        self._saved = True
        out = ROOT_DIR / "models" / "tournaments" / "human_games"
        out.mkdir(parents=True, exist_ok=True)
        moves = np.array([[(-1 if m is None else m["from"] * 9 + 1 + m["dir"] + (4 if m["half"] else 0))
                           for m in f.last_moves] for f in frames], np.int32)
        name = (time.strftime("%Y%m%d_%H%M%S")
                + f"_seed{self.seed}_human{self.human}_game{self.leg + 1}.npz")
        np.savez_compressed(
            out / name,
            owner=np.stack([f.owner for f in frames]).astype(np.int8),
            army=np.stack([f.army for f in frames]).astype(np.int32),
            castle=np.stack([f.castle for f in frames]),
            mountain=self.mountain,
            moves=moves, tick=np.array([f.tick for f in frames]),
            winner=frames[-1].winner, human=self.human, seed=self.seed,
            specs=np.array(self.specs), max_ticks=self.cfg.max_steps, leg=self.leg)
        print(f"saved game to {out / name}")

    def new_match(self, seed: int, leg: int = 0) -> None:
        """Game `leg` of a pair. leg 1 replays the same seed with the seats swapped.
        For leg 0 the seed is moved forward until the map is playable."""
        if hasattr(self, "match"):
            self.save_game()
        self._saved = False
        self.leg = leg
        self.specs = self.base_specs if leg == 0 else self.base_specs[::-1]
        agents = [spec_to_agent(s) for s in self.specs]
        for _ in range(200):
            self.match = Match(self.cfg, agents, seed, self.tick_ms)
            st = self.match.state
            ok, why = map_check(st.mountain, st.castle, st.general_idx, self.cfg.height, self.cfg.width)
            if ok or leg == 1:
                break
            print(f"seed {seed}: {why}; trying {seed + 1}")
            seed += 1
        self.seed = seed
        self.mountain = np.asarray(self.match.state.mountain)
        self.human = self.match.human_seats[0] if self.match.human_seats else -1
        self.selection = -1
        self.trail: list[int] = []
        with self.hist_lock:
            self.history = []
        self.record()
        self.cursor = -1

    def next_game(self) -> None:
        """Game 2 of the pair (same seed, seats swapped), else a new pair."""
        if self.leg == 0:
            self.new_match(self.seed, leg=1)
        else:
            self.new_match(int(time.time() * 1000) & 0x7FFFFFFF, leg=0)

    def record(self) -> None:
        m = self.match
        with m.lock:
            st = m.state
            frame = Frame(
                tick=int(st.step),
                owner=np.asarray(st.owner).reshape(-1).copy(),
                army=np.asarray(st.army).reshape(-1).copy(),
                castle=np.asarray(st.castle).reshape(-1).copy(),
                general=np.asarray(st.general).reshape(-1).copy(),
                land=np.asarray(land_counts(st, self.cfg.num_players)).copy(),
                army_total=np.asarray(army_counts(st, self.cfg.num_players)).copy(),
                last_moves=list(m.last_moves),
                done=bool(st.done),
                winner=int(st.winner),
            )
        with self.hist_lock:
            self.history.append(frame)

    def frame(self) -> Frame:
        with self.hist_lock:
            if not self.history:
                raise RuntimeError("no frames recorded")
            if self.cursor < 0 or self.cursor >= len(self.history):
                return self.history[-1]
            return self.history[self.cursor]

    def n_frames(self) -> int:
        with self.hist_lock:
            return len(self.history)

    def live(self) -> bool:
        return self.cursor < 0

    # -- clock -------------------------------------------------------------
    def run_clock(self, stop: threading.Event) -> None:
        nxt = time.perf_counter()
        while not stop.is_set():
            if self.paused or self.match.state.done:
                time.sleep(0.02)
                nxt = time.perf_counter()
                continue
            self.match.advance()
            self.record()
            if self.match.state.done:
                self.save_game()
            interval = self.tick_ms / 1000.0
            if interval <= 0:
                continue
            nxt += interval
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def step_once(self) -> None:
        if not self.match.state.done:
            self.match.advance()
            self.record()

    # -- human input -------------------------------------------------------
    def click(self, cell: int) -> None:
        f = self.frame()
        if self.human < 0 or cell < 0 or f.owner[cell] != self.human:
            return
        self.match.clear_queue(self.human)
        self.selection = cell
        self.trail = [cell]

    def push(self, direction: int, half: bool) -> None:
        if self.human < 0 or self.selection < 0:
            return
        w, h = self.cfg.width, self.cfg.height
        r, c = divmod(self.selection, w)
        dr, dc = DIRS[direction]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < h and 0 <= cc < w):
            return
        dest = rr * w + cc
        if self.mountain.reshape(-1)[dest]:
            return
        self.match.queue_moves(
            self.human, [{"from": self.selection, "dir": direction, "half": half}]
        )
        # The cursor travels with the army, so the next key chains off this move.
        self.selection = dest
        self.trail.append(dest)
        if len(self.trail) > 96:
            self.trail.pop(0)

    def clear_path(self) -> None:
        if self.human < 0:
            return
        self.match.clear_queue(self.human)
        self.selection = -1
        self.trail = []

    def reconcile_cursor(self) -> None:
        """Once nothing is queued, drop back to where the army actually is."""
        if self.human < 0 or self.selection < 0:
            return
        if self.match.queued_moves(self.human):
            return
        owner = self.frame().owner
        if owner[self.selection] == self.human:
            return
        while self.trail and owner[self.trail[-1]] != self.human:
            self.trail.pop()
        self.selection = self.trail[-1] if self.trail else -1


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def pick_font(size: int, mono: bool = False, bold: bool = False):
    name = (
        "menlo,dejavusansmono,couriernew,monospace"
        if mono
        else "helveticaneue,helvetica,arial,dejavusans,sans"
    )
    path = pygame.font.match_font(name, bold=bold)
    return pygame.font.Font(path, size) if path else pygame.font.SysFont(None, size, bold=bold)


class Renderer:
    def __init__(self, viewer: Viewer, width: int, height: int):
        self.v = viewer
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Fogless Generals")
        self.resize(width, height)

    def resize(self, width: int, height: int) -> None:
        self.w, self.h = width, height
        board_w = max(200, width - SIDEBAR - PAD * 3)
        board_h = max(200, height - PAD * 2)
        self.cell = max(10, min(board_w // self.v.cfg.width, board_h // self.v.cfg.height))
        self.board_px = (self.cell * self.v.cfg.width, self.cell * self.v.cfg.height)
        self.origin = (PAD, PAD + max(0, (board_h - self.board_px[1]) // 2))
        self.f_army = pick_font(max(9, int(self.cell * 0.44)), mono=True, bold=True)
        self.f_h1 = pick_font(19, bold=True)
        self.f_lbl = pick_font(12, bold=True)
        self.f_body = pick_font(13)
        self.f_num = pick_font(15, mono=True, bold=True)
        self.f_tiny = pick_font(11)

    def cell_at(self, pos) -> int:
        x, y = pos
        ox, oy = self.origin
        c = (x - ox) // self.cell
        r = (y - oy) // self.cell
        if 0 <= r < self.v.cfg.height and 0 <= c < self.v.cfg.width:
            return int(r) * self.v.cfg.width + int(c)
        return -1

    def cell_rect(self, i: int) -> pygame.Rect:
        r, c = divmod(i, self.v.cfg.width)
        return pygame.Rect(
            self.origin[0] + c * self.cell, self.origin[1] + r * self.cell, self.cell, self.cell
        )

    def centre(self, i: int) -> tuple[int, int]:
        rect = self.cell_rect(i)
        return rect.centerx, rect.centery

    # -- pieces ------------------------------------------------------------
    def overlay(self, i: int, draw) -> None:
        """Draw onto a per-cell alpha surface, so glyphs sit *behind* the number."""
        surf = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
        draw(surf)
        self.screen.blit(surf, self.cell_rect(i).topleft)

    def crown(self, i: int, rgba) -> None:
        c = self.cell / 2
        s = self.cell * 0.28
        pts = [
            (c - s, c + s * 0.55), (c - s, c - s * 0.35), (c - s * 0.42, c + s * 0.1),
            (c, c - s * 0.75), (c + s * 0.42, c + s * 0.1), (c + s, c - s * 0.35),
            (c + s, c + s * 0.55),
        ]
        self.overlay(i, lambda surf: pygame.draw.polygon(surf, rgba, pts))

    def castle_ring(self, i: int, rgba) -> None:
        c = int(self.cell / 2)
        self.overlay(
            i,
            lambda surf: pygame.draw.circle(
                surf, rgba, (c, c), int(self.cell * 0.27), max(1, int(self.cell * 0.09))
            ),
        )

    def arrow(self, src: int, direction: int, half: bool, colour,
              scale: float = 1.0, lift: float = 0.55) -> None:
        """A move, drawn to stay legible on top of the mover's own territory."""
        w = self.v.cfg.width
        r, c = divmod(src, w)
        dr, dc = DIRS[direction]
        x0, y0 = self.centre(src)
        x1, y1 = self.centre((r + dr) * w + (c + dc))

        bright = lighten(colour, lift)
        shaft = max(3, int(self.cell * 0.14 * scale))
        head = self.cell * 0.34 * scale
        mx, my = x0 + (x1 - x0) * 0.62, y0 + (y1 - y0) * 0.62
        tip = (x0 + (x1 - x0) * 0.94, y0 + (y1 - y0) * 0.94)
        ang = np.arctan2(y1 - y0, x1 - x0)
        wings = [
            (mx - head * np.cos(ang - 0.6), my - head * np.sin(ang - 0.6)),
            (mx - head * np.cos(ang + 0.6), my - head * np.sin(ang + 0.6)),
        ]

        # Dark under-stroke first, so the arrow reads against any tile colour.
        pygame.draw.line(self.screen, (0, 0, 0), (x0, y0), (mx, my), shaft + 4)
        pygame.draw.line(self.screen, bright, (x0, y0), (mx, my), shaft)
        pygame.draw.polygon(self.screen, bright, [tip, *wings])
        pygame.draw.polygon(self.screen, (0, 0, 0), [tip, *wings], 2)

        if half:
            # A "send half" move keeps a marker at the tail.
            pygame.draw.circle(self.screen, (0, 0, 0), (int(x0), int(y0)),
                               max(3, int(self.cell * 0.13)))
            pygame.draw.circle(self.screen, (255, 255, 255), (int(x0), int(y0)),
                               max(2, int(self.cell * 0.10)))

    def text(self, surf_text: str, font, colour, x: int, y: int, right=False) -> int:
        img = font.render(surf_text, True, colour)
        rect = img.get_rect()
        rect.topright = (x, y) if right else rect.topleft
        if not right:
            rect.topleft = (x, y)
        self.screen.blit(img, rect)
        return rect.height

    # -- frame -------------------------------------------------------------
    def draw(self) -> None:
        v = self.v
        f = v.frame()
        self.screen.fill(BG)
        mountain = v.mountain.reshape(-1)

        for i in range(v.cfg.height * v.cfg.width):
            rect = self.cell_rect(i)
            owner = int(f.owner[i])
            is_castle = bool(f.castle[i])
            is_general = bool(f.general[i])
            if mountain[i]:
                colour = MOUNTAIN
            elif owner >= 0:
                colour = PLAYER[owner % len(PLAYER)]
            elif is_castle or is_general:
                colour = NEUTRAL_CASTLE
            else:
                colour = NEUTRAL_LAND
            pygame.draw.rect(self.screen, colour, rect)
            pygame.draw.rect(self.screen, GRID, rect, 1)
            if mountain[i]:
                continue

            if is_general:
                self.crown(i, (255, 255, 255, 90) if owner >= 0 else (0, 0, 0, 70))
            elif is_castle:
                self.castle_ring(i, (255, 255, 255, 82) if owner >= 0 else (*INK3, 170))
            army = int(f.army[i])
            if army:
                label = self.f_army.render(str(army), True,
                                           (255, 255, 255) if owner >= 0 else INK)
                shadow = self.f_army.render(str(army), True, (0, 0, 0))
                pos = label.get_rect(center=self.centre(i))
                self.screen.blit(shadow, pos.move(1, 1))
                self.screen.blit(label, pos)

        for p, move in enumerate(f.last_moves):
            if move:
                self.arrow(move["from"], move["dir"], move["half"], PLAYER[p % len(PLAYER)])

        if v.human >= 0 and v.live():
            src = v.selection
            for action in v.match.queued_moves(v.human):
                k = action % NUM_CELL_ACTIONS
                if k == 0:
                    continue
                m = k - 1
                self.arrow(action // NUM_CELL_ACTIONS, m % 4, m >= 4, BRASS, 0.8, lift=0.12)
            if src >= 0:
                pygame.draw.rect(self.screen, BRASS, self.cell_rect(src), 3)

        self.sidebar(f)
        pygame.display.flip()

    def sidebar(self, f: Frame) -> None:
        v = self.v
        x = self.origin[0] + self.board_px[0] + PAD
        panel = pygame.Rect(x, PAD, self.w - x - PAD, self.h - PAD * 2)
        pygame.draw.rect(self.screen, SURFACE, panel)
        pygame.draw.rect(self.screen, LINE, panel, 1)
        x += 14
        y = PAD + 14
        width = panel.width - 28

        self.text("PERFECT INFORMATION", self.f_lbl, BRASS, x, y)
        y += 18
        self.text("Fogless Generals", self.f_h1, INK, x, y)
        y += 30

        total = v.n_frames() - 1
        shown = f.tick
        state = "live" if v.live() else f"rewound  {shown}/{total}"
        if v.paused and v.live():
            state = "paused"
        self.text(f"game {v.leg + 1} of 2" + ("  (seats swapped)" if v.leg else ""),
                  self.f_tiny, BRASS, x, y)
        y += 18
        self.text(f"tick {shown}", self.f_num, INK, x, y)
        self.text(state, self.f_tiny, BRASS if not v.live() or v.paused else INK3, x + 90, y + 3)
        y += 26
        speed = "max" if v.tick_ms == 0 else f"{v.tick_ms}ms"
        self.text(f"speed {speed}", self.f_tiny, INK3, x, y)
        y += 22

        pygame.draw.line(self.screen, LINE, (x, y), (x + width, y))
        y += 14

        for p in range(v.cfg.num_players):
            colour = PLAYER[p % len(PLAYER)]
            pygame.draw.rect(self.screen, colour, pygame.Rect(x, y + 4, 10, 10))
            alive = f.land[p] > 0
            name = v.match.names[p]
            limit = 17 if p == v.human else 24
            if len(name) > limit:
                name = name[: limit - 1] + "…"
            if p == v.human:
                name += "  (you)"
            self.text(name, self.f_body, INK if alive else INK3, x + 18, y)
            y += 19
            self.text(f"{int(f.land[p])} land", self.f_num, INK2, x + 18, y)
            self.text(f"{int(f.army_total[p])} army", self.f_num, INK2, x + 120, y)
            y += 26
            note = v.match.notes[p]
            if note:
                for line in wrap(note, 30)[:3]:
                    self.text(line, self.f_tiny, (224, 104, 92), x + 18, y)
                    y += 14
                y += 4

        y += 4
        pygame.draw.line(self.screen, LINE, (x, y), (x + width, y))
        y += 14

        if f.done:
            if f.winner < 0:
                self.text("Draw", self.f_h1, INK, x, y)
            else:
                self.text(f"{v.match.names[f.winner][:18]} wins", self.f_h1,
                          PLAYER[f.winner % len(PLAYER)], x, y)
            y += 26
            reason = ("general captured" if f.land[1 - f.winner] == 0 and f.winner >= 0
                      else "decided on land")
            self.text(reason, self.f_tiny, INK3, x, y)
            y += 24

        keys = (
            [("click", "select a tile"),
             ("arrows / wasd", "queue a move"),
             ("shift + dir", "send half"),
             ("c", "clear path"),
             ("p", "pause"),
             ("r", "next game")]
            if v.human >= 0
            else [("space", "pause / resume"),
                  ("left / right", "step history"),
                  ("home / end", "start / live"),
                  ("[  ]", "slower / faster"),
                  ("r / n", "next game"),
                  ("q", "quit")]
        )
        self.text("CONTROLS", self.f_lbl, INK3, x, y)
        y += 18
        for key, what in keys:
            self.text(key, self.f_tiny, BRASS, x, y)
            self.text(what, self.f_tiny, INK3, x + 96, y)
            y += 16


def wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
SPEEDS = [0, 40, 80, 120, 200, 300, 500, 750, 1000, 1500]


def nearest_speed(ms: int) -> int:
    return min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - ms))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--p0", default="human", help="agent spec for seat 0 (red)")
    ap.add_argument("--p1", default="final.py:act", help="agent spec for seat 1 (blue)")
    ap.add_argument("--size", type=int, default=23)
    ap.add_argument("--tick", type=int, default=200, help="ms per tick; 0 = as fast as possible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-spawn", type=int, default=17)
    ap.add_argument("--max-ticks", type=int, default=2000)
    ap.add_argument("--window", default="1180x820", help="initial window size, WxH")
    args = ap.parse_args()

    cfg = EnvConfig(
        height=args.size, width=args.size,
        min_spawn_dist=min(args.min_spawn, args.size - 4),
        max_spawn_dist=10**9, max_steps=args.max_ticks, bfs_iters=4 * args.size,
    )
    try:
        win_w, win_h = (int(v) for v in args.window.lower().split("x"))
    except ValueError:
        raise SystemExit("--window must look like 1180x820")

    print(f"{args.p0}  vs  {args.p1}   ({cfg.height}x{cfg.width})")
    print("loading agents…")
    viewer = Viewer(cfg, [args.p0, args.p1], args.seed, max(0, args.tick))

    pygame.init()
    pygame.key.set_repeat(260, 45)
    renderer = Renderer(viewer, win_w, win_h)

    stop = threading.Event()
    threading.Thread(target=viewer.run_clock, args=(stop,), daemon=True).start()

    speed_idx = nearest_speed(viewer.tick_ms)
    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    renderer.resize(event.w, event.h)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    viewer.click(renderer.cell_at(event.pos))
                elif event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    if event.key in (pygame.K_ESCAPE,) or (
                        event.key == pygame.K_q and viewer.human < 0
                    ):
                        running = False
                    elif event.key in (pygame.K_r, pygame.K_n) and not (
                            event.key == pygame.K_n and viewer.human >= 0 and not viewer.match.state.done):
                        viewer.next_game()
                        speed_idx = nearest_speed(viewer.tick_ms)
                    elif event.key == pygame.K_LEFTBRACKET:
                        speed_idx = min(len(SPEEDS) - 1, speed_idx + 1)
                        viewer.tick_ms = SPEEDS[speed_idx]
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_idx = max(0, speed_idx - 1)
                        viewer.tick_ms = SPEEDS[speed_idx]
                    elif viewer.human >= 0 and event.key in KEY_DIR:
                        viewer.push(KEY_DIR[event.key], bool(mods & pygame.KMOD_SHIFT))
                    elif viewer.human >= 0 and event.key == pygame.K_c:
                        viewer.clear_path()
                    elif event.key in (pygame.K_SPACE, pygame.K_p):
                        viewer.paused = not viewer.paused
                        if not viewer.paused:
                            viewer.cursor = -1  # resuming always returns to live
                    elif event.key == pygame.K_LEFT:
                        viewer.paused = True
                        total = viewer.n_frames()
                        cur = total - 1 if viewer.cursor < 0 else viewer.cursor
                        viewer.cursor = max(0, cur - (10 if mods & pygame.KMOD_SHIFT else 1))
                    elif event.key == pygame.K_RIGHT:
                        total = viewer.n_frames()
                        if viewer.paused and viewer.cursor >= 0:
                            step = 10 if mods & pygame.KMOD_SHIFT else 1
                            viewer.cursor = min(total - 1, viewer.cursor + step)
                            if viewer.cursor >= total - 1:
                                viewer.cursor = -1
                        elif viewer.paused:
                            viewer.step_once()   # advance one tick while paused
                    elif event.key == pygame.K_HOME:
                        viewer.paused = True
                        viewer.cursor = 0
                    elif event.key == pygame.K_END:
                        viewer.cursor = -1

            viewer.reconcile_cursor()
            renderer.draw()
            clock.tick(60)
    finally:
        viewer.save_game()
        stop.set()
        pygame.quit()


if __name__ == "__main__":
    main()
