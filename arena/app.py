"""The arena desktop app: a friend types their name, picks a bot, and plays it.

    python play_arena.py                 # 1280x800, resizable
    python play_arena.py --smoke         # headless run, screenshots to /tmp/arena_shots

Screens (one class each, all drawn every frame from the current window size so
everything reflows on resize):

    HomeScreen         name field, recent names, bot menu, options, Play
    GameScreen         board (BoardPainter) + sidebar; input as in viewer.py
    ResultScreen       who won, how, final counts; after game 1 of a pair: play
                       game 2 (same map, seats swapped); after game 2: the pair
                       score and Rematch / New player
    LeaderboardScreen  Bots and Players tables (arena.store)
    TutorialScreen     full-window host for arena.tutorial.Tutorial

Look: the dark slate + brass palette of viewer.py / board_draw.py, square
corners, hairline rules instead of boxes, tracked uppercase labels in brass,
monospace numbers, a brass bar for whatever is selected or focused. Player
colours (red/blue) are only ever used for the players.

The engine (arena.session, arena.bots) and the tutorial (arena.tutorial) are
imported lazily, so a missing or broken module shows up as a message on screen
instead of a crash.
"""
from __future__ import annotations

import pathlib
import threading
import traceback

import pygame

from arena import store
from arena.board_draw import (BG, BRASS, INK, INK2, INK3, LINE, PLAYER, SURFACE,
                              BoardPainter, lighten, pick_font)

# --------------------------------------------------------------------------- #
# palette additions (derived from the shared palette, no new hues)
# --------------------------------------------------------------------------- #
ROW_HOVER = (15, 21, 24)              # between BG and SURFACE
ROW_SEL = SURFACE
BRASS_DIM = (120, 90, 38)             # brass on brass (hints inside the Play button)
SHADE = (8, 12, 13)                   # translucent strips over the board

SEATS = [("RED", "moves first"), ("BLUE", "moves second")]
LENGTHS = [300, 600, 1000]
SPEEDS = [("SLOW", 500), ("NORMAL", 350), ("FAST", 200)]
TIER_INK = {"fast": INK2, "ok": INK3, "slow": BRASS, "broken": INK3, "unknown": INK3}
KEY_DIR = {
    pygame.K_UP: 0, pygame.K_w: 0,
    pygame.K_DOWN: 1, pygame.K_s: 1,
    pygame.K_LEFT: 2, pygame.K_a: 2,
    pygame.K_RIGHT: 3, pygame.K_d: 3,
}
NAME_MAX = 24
DOUBLE_CLICK_MS = 400


# --------------------------------------------------------------------------- #
# type and text
# --------------------------------------------------------------------------- #
_FONTS: dict = {}


def font(size: int, mono: bool = False, bold: bool = False) -> pygame.font.Font:
    key = (size, mono, bold)
    if key not in _FONTS:
        _FONTS[key] = pick_font(size, mono=mono, bold=bold)
    return _FONTS[key]


class T:
    """The type scale. Attributes are filled by T.load() once pygame is up."""
    label = small = body = body_b = h2 = h1 = display = huge = None
    mono_s = mono = mono_b = mono_l = mono_xl = None

    @classmethod
    def load(cls) -> None:
        cls.label = font(11, bold=True)          # tracked uppercase labels
        cls.small = font(12)
        cls.body = font(14)
        cls.body_b = font(15, bold=True)
        cls.h2 = font(18, bold=True)
        cls.h1 = font(26, bold=True)
        cls.display = font(38, bold=True)        # the name field
        cls.huge = font(52, bold=True)           # the result line
        cls.mono_s = font(11, mono=True)
        cls.mono = font(13, mono=True)
        cls.mono_b = font(13, mono=True, bold=True)
        cls.mono_l = font(19, mono=True, bold=True)
        cls.mono_xl = font(30, mono=True, bold=True)


_TEXT: dict = {}


def render(text: str, f: pygame.font.Font, colour, track: float = 0.0) -> pygame.Surface:
    """Cached text surface; `track` > 0 letter-spaces it (for the small labels)."""
    key = (id(f), text, colour, track)
    surf = _TEXT.get(key)
    if surf is None:
        if len(_TEXT) > 4000:
            _TEXT.clear()
        if track <= 0 or not text:
            surf = f.render(text, True, colour)
        else:
            glyphs = [f.render(ch, True, colour) for ch in text]
            width = sum(g.get_width() for g in glyphs) + int(track * (len(glyphs) - 1))
            surf = pygame.Surface((max(1, width), f.get_height()), pygame.SRCALPHA)
            x = 0.0
            for g in glyphs:
                surf.blit(g, (int(x), 0))
                x += g.get_width() + track
        _TEXT[key] = surf
    return surf


def put(surface, text, f, colour, x, y, anchor: str = "topleft", track: float = 0.0) -> pygame.Rect:
    img = render(text, f, colour, track)
    rect = img.get_rect(**{anchor: (int(x), int(y))})
    surface.blit(img, rect)
    return rect


def label(surface, text, x, y, colour=BRASS, anchor="topleft") -> pygame.Rect:
    """The small-caps label used for every section heading and column title."""
    return put(surface, text.upper(), T.label, colour, x, y, anchor, track=1.6)


def fit(text: str, f: pygame.font.Font, width: int, from_left: bool = False) -> str:
    """Trim `text` with an ellipsis so it renders within `width` pixels."""
    if f.size(text)[0] <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cand = ("…" + text[-mid:]) if from_left else (text[:mid].rstrip() + "…")
        if f.size(cand)[0] <= width:
            lo = mid
        else:
            hi = mid - 1
    return ("…" + text[-lo:]) if from_left else (text[:lo].rstrip() + "…")


def wrap(text: str, f: pygame.font.Font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        cand = f"{line} {word}".strip()
        if line and f.size(cand)[0] > width:
            lines.append(line)
            line = word
        else:
            line = cand
    if line:
        lines.append(line)
    return lines


def hline(surface, x0, x1, y, colour=LINE, width=1) -> None:
    pygame.draw.line(surface, colour, (x0, y), (x1, y), width)


def bar(surface, x, y, h, colour=BRASS, w=3) -> None:
    """The brass selection / focus bar."""
    pygame.draw.rect(surface, colour, pygame.Rect(x, y, w, h))


def header(surface, width: int, section: str, right: str = "", margin: int = 32) -> int:
    """Top strip shared by Home and Leaderboard. Returns the y below it."""
    r = label(surface, "Fogless Generals", margin, 24)
    put(surface, "/", T.label, INK3, r.right + 10, 24)
    label(surface, section, r.right + 24, 24, INK2)
    if right:
        put(surface, right, T.mono_s, INK3, width - margin, 24, "topright")
    hline(surface, margin, width - margin, 52)
    return 52


# --------------------------------------------------------------------------- #
# widgets (stateless draw helpers; screens keep the rects for hit-testing)
# --------------------------------------------------------------------------- #
def button(surface, rect: pygame.Rect, text: str, *, primary=False, enabled=True,
           focused=False, hover=False, hint: str = "") -> None:
    """A square button. Primary = solid brass; plain = hairline outline."""
    if primary and enabled:
        pygame.draw.rect(surface, lighten(BRASS, 0.12) if hover else BRASS, rect)
        ink, hint_ink = BG, BRASS_DIM
    else:
        if hover and enabled:
            pygame.draw.rect(surface, ROW_HOVER, rect)
        pygame.draw.rect(surface, LINE, rect, 1)
        ink = (INK if (hover or focused) else INK2) if enabled else INK3
        hint_ink = INK3
    put(surface, text.upper(), T.label, ink, rect.x + 16, rect.centery, "midleft", track=1.6)
    if hint:
        put(surface, hint, T.mono_s, hint_ink, rect.right - 14, rect.centery, "midright")
    if focused:
        bar(surface, rect.x - 10, rect.y, rect.height)


def segments(surface, rect: pygame.Rect, options, selected: int, hover: int = -1) -> list[pygame.Rect]:
    """A row of equal hairline cells; the chosen one gets INK text and a brass
    rule along its bottom. `options` items: (text, sub, swatch-rgb or None)."""
    n = len(options)
    rects = []
    for k, (text, sub, swatch) in enumerate(options):
        x0 = rect.x + rect.width * k // n
        x1 = rect.x + rect.width * (k + 1) // n
        r = pygame.Rect(x0, rect.y, x1 - x0 + (1 if k < n - 1 else 0), rect.height)
        rects.append(r)
        on = k == selected
        if on:
            pygame.draw.rect(surface, ROW_SEL, r)
        elif k == hover:
            pygame.draw.rect(surface, ROW_HOVER, r)
        pygame.draw.rect(surface, LINE, r, 1)
        x = r.x + 12
        if swatch is not None:
            pygame.draw.rect(surface, swatch, pygame.Rect(x, r.centery - 5, 10, 10))
            x += 18
        t = put(surface, text, T.label, INK if on else INK3, x, r.centery, "midleft", track=1.4)
        if sub and t.right + 8 + T.mono_s.size(sub)[0] < r.right - 8:
            put(surface, sub, T.mono_s, INK2 if on else INK3, t.right + 8, r.centery, "midleft")
        if on:
            pygame.draw.rect(surface, BRASS, pygame.Rect(r.x, r.bottom - 3, r.width, 3))
    return rects


def scrollbar(surface, x, y, h, first, visible, total) -> None:
    if total <= visible or h <= 0:
        return
    pygame.draw.rect(surface, LINE, pygame.Rect(x, y, 2, h))
    th = max(18, h * visible // total)
    ty = y + (h - th) * first // max(1, total - visible)
    pygame.draw.rect(surface, INK3, pygame.Rect(x, ty, 2, th))


def strip(surface, rect: pygame.Rect, alpha: int = 232) -> None:
    """A translucent band across the board with brass hairlines top and bottom."""
    shade = pygame.Surface(rect.size, pygame.SRCALPHA)
    shade.fill((*SHADE, alpha))
    surface.blit(shade, rect.topleft)
    hline(surface, rect.x, rect.right - 1, rect.y, BRASS)
    hline(surface, rect.x, rect.right - 1, rect.bottom - 1, BRASS)


def pct(x) -> str:
    return "—" if x is None else f"{round(100 * x)}%"


def rel(path: str) -> str:
    """A path relative to the repo root when it lives inside it."""
    try:
        return str(pathlib.Path(path).resolve().relative_to(store.ROOT))
    except (ValueError, OSError):
        return str(path)


def is_final(b) -> bool:
    return b.id.lower().startswith("final") or b.name.lower().startswith("final")


# --------------------------------------------------------------------------- #
# screens
# --------------------------------------------------------------------------- #
class Screen:
    """Base: screens draw from the live surface size and keep their own rects."""
    text_input = False     # Home wants TEXTINPUT; the game turns it off (no IME popups)

    def __init__(self, app: "App"):
        self.app = app
        self.mouse = (-1, -1)

    def enter(self) -> None:
        pass

    def handle(self, ev) -> None:
        pass

    def update(self, now_ms: float, dt_ms: float) -> None:
        pass

    def draw(self, surface) -> None:
        pass

    def close(self) -> None:
        """Window is closing while this screen is up."""

    def hovering(self, rect) -> bool:
        return rect is not None and rect.collidepoint(self.mouse)


# ----------------------------------------------------------------------------
class HomeScreen(Screen):
    text_input = True
    FOCUS = ["name", "recent", "bots", "seat", "length", "speed", "play", "board", "rules", "controls"]
    ROW_H = 52

    def __init__(self, app, focus: str = "name"):
        super().__init__(app)
        self.focus = focus
        self.recent_idx = 0
        self.first_row = 0           # bot list scroll (row index)
        self.visible_rows = 8
        self.flash = ("", 0.0)
        self.last_click = (-1, 0)    # (row, ms) for double-click
        self.rects: dict = {}
        self.bot_rects: list = []
        self.chip_rects: list = []
        self.seg_rects: dict = {}
        self.pending_reveal = True   # scroll the chosen bot into view on first draw

    # -- data helpers ----------------------------------------------------------
    @property
    def bots(self) -> list:
        return self.app.bots or []

    def bot_index(self) -> int:
        for k, b in enumerate(self.bots):
            if b.id == self.app.bot_id:
                return k
        return -1

    def select_bot(self, k: int) -> None:
        if not self.bots:
            return
        k = max(0, min(len(self.bots) - 1, k))
        self.app.bot_id = self.bots[k].id
        self.reveal(k)

    def reveal(self, k: int) -> None:
        if k < self.first_row:
            self.first_row = k
        elif k >= self.first_row + self.visible_rows:
            self.first_row = k - self.visible_rows + 1
        self.first_row = max(0, min(self.first_row, max(0, len(self.bots) - self.visible_rows)))

    def chosen(self):
        k = self.bot_index()
        return self.bots[k] if k >= 0 else None

    def can_play(self) -> str:
        """'' when ready, else the reason Play is disabled."""
        if not self.app.name.strip():
            return "Type a name to play"
        b = self.chosen()
        if b is None:
            return "Pick a bot"
        if b.tier == "broken":
            return f"{b.name} is broken"
        return ""

    def say(self, text: str) -> None:
        self.flash = (text, pygame.time.get_ticks())

    def focus_order(self) -> list[str]:
        return [f for f in self.FOCUS if f != "recent" or self.app.recent]

    def move_focus(self, step: int) -> None:
        order = self.focus_order()
        k = order.index(self.focus) if self.focus in order else 0
        self.focus = order[(k + step) % len(order)]

    # -- actions ----------------------------------------------------------------
    def play(self) -> None:
        why = self.can_play()
        if why:
            self.say(why)
            return
        self.app.start_game()

    def activate(self, what: str) -> None:
        if what == "play":
            self.play()
        elif what == "board":
            self.app.go(LeaderboardScreen(self.app))
        elif what == "rules":
            self.app.go(TutorialScreen(self.app, "rules"))
        elif what == "controls":
            self.app.go(TutorialScreen(self.app, "controls"))

    def cycle_option(self, what: str, step: int) -> None:
        a = self.app
        if what == "seat":
            a.seat = (a.seat + step) % 2
        elif what == "length":
            a.max_ticks = LENGTHS[(LENGTHS.index(a.max_ticks) + step) % len(LENGTHS)]
        elif what == "speed":
            ms = [s for _, s in SPEEDS]
            a.tick_ms = ms[(ms.index(a.tick_ms) + step) % len(ms)]

    # -- events -----------------------------------------------------------------
    def handle(self, ev) -> None:
        a = self.app
        if ev.type == pygame.TEXTINPUT:
            text = "".join(ch for ch in ev.text if ch.isprintable())
            if text and len(a.name) < NAME_MAX:
                if not a.name and text.isspace():
                    return
                a.name = (a.name + text)[:NAME_MAX]
            return
        if ev.type == pygame.MOUSEMOTION:
            self.mouse = ev.pos
        elif ev.type == pygame.MOUSEWHEEL:
            if self.rects.get("list") and self.rects["list"].collidepoint(self.mouse):
                self.first_row = max(0, min(self.first_row - ev.y,
                                            max(0, len(self.bots) - self.visible_rows)))
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.mouse = ev.pos
            self.click(ev.pos)
        elif ev.type == pygame.KEYDOWN:
            self.key(ev)

    def click(self, pos) -> None:
        a = self.app
        if self.rects.get("name") and self.rects["name"].collidepoint(pos):
            self.focus = "name"
            return
        for k, r in enumerate(self.chip_rects):
            if r.collidepoint(pos):
                a.name = a.recent[k]
                self.focus, self.recent_idx = "recent", k
                return
        for k, r in self.bot_rects:
            if r.collidepoint(pos):
                now = pygame.time.get_ticks()
                double = self.last_click[0] == k and now - self.last_click[1] < DOUBLE_CLICK_MS
                self.select_bot(k)
                self.focus = "bots"
                self.last_click = (k, now)
                if double:
                    self.play()
                return
        for what, rects in self.seg_rects.items():
            for k, r in enumerate(rects):
                if r.collidepoint(pos):
                    self.focus = what
                    if what == "seat":
                        a.seat = k
                    elif what == "length":
                        a.max_ticks = LENGTHS[k]
                    else:
                        a.tick_ms = SPEEDS[k][1]
                    return
        for what in ("play", "board", "rules", "controls"):
            if self.rects.get(what) and self.rects[what].collidepoint(pos):
                self.focus = what
                self.activate(what)
                return

    def key(self, ev) -> None:
        a, k = self.app, ev.key
        shift = bool(ev.mod & pygame.KMOD_SHIFT)
        if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.focus in ("board", "rules", "controls"):
                self.activate(self.focus)
            elif self.focus == "recent" and a.recent and a.name != a.recent[self.recent_idx]:
                a.name = a.recent[self.recent_idx]
            else:
                self.play()
        elif k == pygame.K_TAB:
            self.move_focus(-1 if shift else 1)
        elif k == pygame.K_BACKSPACE:
            if ev.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT | pygame.KMOD_META):
                a.name = a.name.rstrip().rpartition(" ")[0] if " " in a.name.strip() else ""
            else:
                a.name = a.name[:-1]
            self.focus = "name" if self.focus == "recent" else self.focus
        elif k == pygame.K_ESCAPE:
            a.name = ""
            self.focus = "name"
        elif k in (pygame.K_UP, pygame.K_DOWN):
            step = -1 if k == pygame.K_UP else 1
            if self.focus in ("name", "bots", "recent"):
                # the fast path: type a name, arrow to a bot, Enter
                k0 = self.bot_index()
                self.select_bot(step if k0 < 0 and step > 0 else k0 + step)
            else:
                self.move_focus(step)
        elif k in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
            step = self.visible_rows * (-1 if k == pygame.K_PAGEUP else 1)
            self.select_bot(max(0, self.bot_index()) + step)
        elif k in (pygame.K_LEFT, pygame.K_RIGHT):
            step = -1 if k == pygame.K_LEFT else 1
            if self.focus in ("seat", "length", "speed"):
                self.cycle_option(self.focus, step)
            elif self.focus == "recent" and a.recent:
                self.recent_idx = (self.recent_idx + step) % len(a.recent)
        elif k == pygame.K_F1:
            self.activate("rules")
        elif k == pygame.K_F2:
            self.activate("controls")
        elif k == pygame.K_F3:
            self.activate("board")

    # -- drawing ----------------------------------------------------------------
    def draw(self, s) -> None:
        a = self.app
        W, H = s.get_size()
        s.fill(BG)
        M = 32 if W >= 1100 else 22
        n_games = len(a.games)
        header(s, W, "Arena", f"23×23 · FULL INFORMATION · {n_games} GAME{'S' * (n_games != 1)} ON RECORD", M)

        lx = M + 12                                   # room for the focus bar
        lw = max(300, min(440, int(W * 0.34)))
        rx = lx + lw + max(32, int(W * 0.035))
        rw = W - M - rx
        self.draw_left(s, lx, 76, lw, H)
        self.draw_bots(s, rx, 76, rw, H - 76 - 24)

    def draw_left(self, s, x, y, w, H) -> None:
        a = self.app
        now = pygame.time.get_ticks()
        roomy = H >= 760                  # short windows tighten the vertical rhythm
        put(s, "01", T.label, BRASS, x, y)
        label(s, "Player", x + 26, y, INK2)
        y += 24

        # name field: big type on a hairline, brass when it has focus
        field = pygame.Rect(x, y, w, 58)
        self.rects["name"] = field
        focused = self.focus == "name"
        f = T.display
        if a.name:
            shown = fit(a.name, f, w - 12, from_left=True)
            t = put(s, shown, f, INK, x, y + 6)
            cx = t.right + 3
        else:
            put(s, "Type a name", f, (52, 66, 71), x + (10 if focused else 0), y + 6)
            cx = x + 1
        if focused and (now // 530) % 2 == 0:
            pygame.draw.rect(s, BRASS, pygame.Rect(cx, y + 12, 3, f.get_height() - 14))
        if focused:
            hline(s, x, x + w, y + 56, BRASS, 2)
        else:
            hline(s, x, x + w, y + 56, INK3)
        if self.focus == "name":
            bar(s, x - 12, y + 4, 50)
        y += 66 if roomy else 62
        put(s, "Typing goes straight in.  ENTER starts the game.", T.small, INK3, x, y)
        y += 30 if roomy else 22

        # recent names, as flowing hairline chips
        self.chip_rects = []
        if a.recent:
            label(s, "Recent", x, y, INK3)
            y += 20
            cx, cy, rows = x, y, 1
            for k, name in enumerate(a.recent):
                text = fit(name, T.body, 160)
                cw = T.body.size(text)[0] + 22
                if cx + cw > x + w and cx > x:
                    rows += 1
                    if rows > (2 if roomy else 1):
                        break
                    cx, cy = x, cy + 32
                r = pygame.Rect(cx, cy, cw, 26)
                self.chip_rects.append(r)
                on = self.focus == "recent" and k == self.recent_idx
                current = name.strip().lower() == a.name.strip().lower()
                if self.hovering(r) or current:
                    pygame.draw.rect(s, ROW_SEL if current else ROW_HOVER, r)
                pygame.draw.rect(s, LINE, r, 1)
                put(s, text, T.body, INK if (on or current) else INK2, r.x + 11, r.centery, "midleft")
                if on:
                    pygame.draw.rect(s, BRASS, pygame.Rect(r.x, r.bottom - 2, r.width, 2))
                cx += cw + 6
            if self.focus == "recent":
                bar(s, x - 12, y, cy + 26 - y)
            y = cy + 26 + (26 if roomy else 18)
        else:
            y += 4

        # setup: side, length, speed
        put(s, "03", T.label, BRASS, x, y)
        label(s, "Setup", x + 26, y, INK2)
        y += 24
        rows = [
            ("seat", "Game 1 side", [(n, sub, PLAYER[k]) for k, (n, sub) in enumerate(SEATS)], a.seat),
            ("length", "Length", [(str(t), "ticks", None) for t in LENGTHS], LENGTHS.index(a.max_ticks)),
            ("speed", "Speed", [(n, f"{ms}ms", None) for n, ms in SPEEDS],
             [ms for _, ms in SPEEDS].index(a.tick_ms)),
        ]
        lab_w = 92
        seg_h = 36 if roomy else 32
        self.seg_rects = {}
        for key, title, opts, sel in rows:
            label(s, title, x, y + seg_h // 2 - 6, INK3)
            r = pygame.Rect(x + lab_w, y, w - lab_w, seg_h)
            hover = -1
            rects = segments(s, r, opts, sel)
            for k, rr in enumerate(rects):
                if self.hovering(rr) and k != sel:
                    hover = k
            if hover >= 0:
                rects = segments(s, r, opts, sel, hover)
            self.seg_rects[key] = rects
            if self.focus == key:
                bar(s, x - 12, y, seg_h)
            y += seg_h + (8 if roomy else 6)
        y += 14 if roomy else 6

        # play
        why = self.can_play()
        b = self.chosen()
        r = pygame.Rect(x, y, w, 56 if roomy else 46)
        self.rects["play"] = r
        button(s, r, "Play" if not why else why, primary=True, enabled=not why,
               focused=self.focus == "play", hover=self.hovering(r), hint="ENTER" if not why else "")
        y += r.height + 8
        text, at = self.flash
        if text and now - at < 2600:          # a refused action explains itself here
            put(s, fit(text, T.small, w), T.small, BRASS, x, y)
        elif b is not None and a.name.strip():
            side = SEATS[a.seat][0].lower()
            speed = next(n for n, ms in SPEEDS if ms == a.tick_ms).lower()
            line = (f"{a.name.strip()} ({side}, then swapped) vs {b.name}  ·  2 x {a.max_ticks} ticks"
                    f"  ·  {speed}")
            put(s, fit(line, T.small, w), T.small, INK2, x, y)
        y += 32 if roomy else 24

        # secondary destinations, as a ruled list
        hline(s, x, x + w, y)
        for key, text, hint in (("board", "Leaderboard", "F3"),
                                ("rules", "How the game works", "F1"),
                                ("controls", "Controls", "F2")):
            r = pygame.Rect(x, y, w, 38 if roomy else 32)
            self.rects[key] = r
            on = self.focus == key or self.hovering(r)
            if self.hovering(r):
                pygame.draw.rect(s, ROW_HOVER, r.inflate(0, -1))
            put(s, text, T.body, INK if on else INK2, x + 2, r.centery, "midleft")
            put(s, hint, T.mono_s, INK3, x + w, r.centery, "midright")
            if self.focus == key:
                bar(s, x - 12, y + 6, r.height - 12)
            y += r.height
            hline(s, x, x + w, y)

    def draw_bots(self, s, x, y, w, h) -> None:
        a = self.app
        top = y
        put(s, "02", T.label, BRASS, x, y)
        label(s, "Opponent", x + 26, y, INK2)
        if self.focus == "bots":
            bar(s, x - 12, y - 2, 16)
        if a.bots is not None:
            label(s, f"{len(a.bots)} bots", x + w, y, INK3, "topright")
        y += 24

        # column grid, measured from the right edge
        c_elo = x + w - 8
        c_win = c_elo - 64
        c_games = c_win - 76
        c_tier = c_games - 78 - 104
        name_w = c_tier - x - 30
        hline(s, x, x + w, y)
        hy = y + 8
        label(s, "Bot", x + 16, hy, INK3)
        label(s, "Tier", c_tier, hy, INK3)
        label(s, "Games", c_games, hy, INK3, "topright")
        label(s, "Bot win", c_win, hy, INK3, "topright")
        label(s, "Elo", c_elo, hy, INK3, "topright")
        y += 30
        hline(s, x, x + w, y)

        detail_h = 66
        bottom = top + h
        rows_fit = max(1, (bottom - y - detail_h) // self.ROW_H)
        list_rect = pygame.Rect(x, y, w, rows_fit * self.ROW_H)
        self.rects["list"] = list_rect
        self.visible_rows = max(1, list_rect.height // self.ROW_H)
        self.bot_rects = []

        if a.bots is None:
            msg = "Loading bots…" if not a.bots_error else "Could not load the bot list."
            put(s, msg, T.body, INK2, x + 16, y + 18)
            if a.bots_error:
                for k, line in enumerate(wrap(a.bots_error, T.mono_s, w - 32)[:6]):
                    put(s, line, T.mono_s, INK3, x + 16, y + 44 + 16 * k)
            return
        if not a.bots:
            put(s, "No bots found.", T.body, INK2, x + 16, y + 18)
            return

        sel = self.bot_index()
        if self.pending_reveal and sel >= 0:
            self.pending_reveal = False
            self.first_row = max(0, sel - self.visible_rows // 2)
        self.first_row = max(0, min(self.first_row, max(0, len(a.bots) - self.visible_rows)))
        stats = a.bot_stats
        clip = s.get_clip()
        s.set_clip(list_rect)
        for k in range(self.first_row, min(len(a.bots), self.first_row + self.visible_rows)):
            b = a.bots[k]
            ry = y + (k - self.first_row) * self.ROW_H
            r = pygame.Rect(x, ry, w, self.ROW_H)
            self.bot_rects.append((k, r))
            broken = b.tier == "broken"
            if k == sel:
                pygame.draw.rect(s, ROW_SEL, r)
                bar(s, x, ry, self.ROW_H - 1)
            elif self.hovering(r):
                pygame.draw.rect(s, ROW_HOVER, r)
            ink = INK3 if broken else INK
            nm = put(s, fit(b.name, T.body_b, name_w), T.body_b, ink, x + 16, ry + 9)
            if is_final(b) and not b.name.lower().startswith("final"):
                label(s, "Final", nm.right + 10, ry + 12, BRASS)
            put(s, fit(b.summary or "—", T.small, name_w), T.small, INK3, x + 16, ry + 30)

            # tier: a hairline tag; slow in brass, broken struck through
            tier = (b.tier or "unknown").lower()
            text = "—" if tier == "unknown" else tier.upper()
            tag = render(text, T.mono_s, TIER_INK.get(tier, INK3))
            tr = pygame.Rect(c_tier, ry + 12, tag.get_width() + 12, 18)
            if tier in ("slow", "broken"):
                pygame.draw.rect(s, BRASS if tier == "slow" else INK3, tr, 1)
                s.blit(tag, tag.get_rect(center=tr.center))
            else:
                s.blit(tag, (tr.x, tr.centery - tag.get_height() // 2))
            if broken:
                hline(s, tr.x + 3, tr.right - 4, tr.centery, INK3)
            if b.ms_mean is not None:
                put(s, f"{b.ms_mean:.0f} ms/move", T.mono_s, INK3, c_tier, ry + 31)

            row = stats.get(b.id)
            nums = (("—", "—", "—") if row is None else
                    (str(row["games"]), pct(row["win_rate"]), str(row["elo"])))
            for cx, v in zip((c_games, c_win, c_elo), nums):
                put(s, v, T.mono, INK3 if v == "—" else INK2, cx, ry + 12, "topright")
            hline(s, x, x + w, ry + self.ROW_H - 1)
        s.set_clip(clip)
        scrollbar(s, x + w + 6, y, self.visible_rows * self.ROW_H, self.first_row,
                  self.visible_rows, len(a.bots))

        # detail strip for the chosen bot
        b = self.chosen()
        dy = list_rect.bottom + 12
        hline(s, x, x + w, dy - 6)
        if b is not None:
            meta = f"{b.id}  ·  {b.kind}  ·  {rel(b.path) if b.path else 'builtin'}"
            put(s, fit(meta, T.mono_s, w - 16), T.mono_s, INK3, x + 16, dy + 2)
            if b.tier == "broken":
                text, ink = "This bot failed its health check and cannot be played.", BRASS
            elif b.tier == "slow":
                text, ink = "Slow to move: the board keeps ticking while it thinks.", BRASS
            else:
                text, ink = (b.summary or ""), INK2
            put(s, fit(text, T.small, w - 16), T.small, ink, x + 16, dy + 22)


# ----------------------------------------------------------------------------
class GameScreen(Screen):
    SIDEBAR = 300
    PAD = 16

    def __init__(self, app, bot, player: str, seat: int, max_ticks: int, tick_ms: int,
                 pair: dict | None = None):
        super().__init__(app)
        self.bot, self.player = bot, player
        # {"id", "seed", "first", "leg", "results"}: game `leg` of a two-game pair
        self.pair = pair
        self.seat, self.max_ticks, self.tick_ms = seat, max_ticks, tick_ms
        self.session = None
        self.error = ""
        self.started = pygame.time.get_ticks()
        self.selection = -1
        self.trail: list[int] = []
        self.confirm = None            # None | "leave" | "resign"
        self.was_paused = False
        self.record = None
        self.done_at = None
        self.painter = None
        self.board_rect = None
        self.rects: dict = {}
        self.stats = None
        self.view = None
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        """Build the session off the main thread so the window never freezes."""
        try:
            from arena.session import GameSession
            pair = self.pair or {}
            sess = GameSession(self.bot.id, self.player, human_seat=self.seat,
                               seed=pair.get("seed"), max_ticks=self.max_ticks, tick_ms=self.tick_ms,
                               pair_id=pair.get("id"), leg=pair.get("leg", 0))
            self.painter = BoardPainter(sess.h, sess.w)
            self.session = sess
        except Exception as e:  # noqa: BLE001 - shown on screen
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    @property
    def human(self) -> int:
        return self.session.human_seat if self.session else self.seat

    def names(self) -> list[str]:
        bot_name = getattr(getattr(self.session, "bot", None), "name", None) or self.bot.name
        out = ["", ""]
        out[self.human] = self.player
        out[1 - self.human] = bot_name
        return out

    # -- paused flag (tolerate a read-only attribute) ----------------------------
    def set_paused(self, value: bool) -> None:
        try:
            self.session.paused = value
        except AttributeError:
            pass

    # -- input (same semantics as viewer.py) -----------------------------------
    def click_cell(self, cell: int) -> None:
        if cell < 0 or self.view is None or int(self.view.owner[cell]) != self.human:
            return
        self.session.clear_queue()
        self.selection = cell
        self.trail = [cell]

    def push(self, direction: int, half: bool) -> None:
        if self.selection < 0:
            return
        dest = self.painter.neighbour(self.selection, direction)
        if dest < 0 or (self.view is not None and bool(self.view.mountain[dest])):
            return
        if self.session.queue_move(self.selection, direction, half):
            # the cursor travels with the army, so the next key chains off this move
            self.selection = dest
            self.trail.append(dest)
            if len(self.trail) > 96:
                self.trail.pop(0)

    def clear_path(self) -> None:
        self.session.clear_queue()
        self.selection = -1
        self.trail = []

    def reconcile(self) -> None:
        """Once nothing is queued, drop the cursor back to where the army really is."""
        if self.selection < 0 or self.view is None or self.session.queued():
            return
        owner = self.view.owner
        if int(owner[self.selection]) == self.human:
            return
        while self.trail and int(owner[self.trail[-1]]) != self.human:
            self.trail.pop()
        self.selection = self.trail[-1] if self.trail else -1

    def open_confirm(self, what: str) -> None:
        self.confirm = what
        self.was_paused = bool(self.session.paused)
        self.set_paused(True)

    def close_confirm(self) -> None:
        self.confirm = None
        self.set_paused(self.was_paused)

    def leave(self) -> None:
        """Esc → confirmed: record as a quit, then show the result."""
        self.confirm = None
        self._finish("quit")
        self.app.go(ResultScreen(self.app, self))

    def _finish(self, reason=None) -> None:
        if self.record is not None or self.session is None:
            return
        try:
            self.record = self.session.finish(reason) if reason else self.session.finish()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.record = {"error": f"{type(e).__name__}: {e}"}
        self.stats = self.session.stats()
        self.app.refresh_games()

    def close(self) -> None:
        if self.session is not None and self.record is None:
            self._finish(None if self.session.stats().get("done") else "quit")

    def handle(self, ev) -> None:
        if ev.type == pygame.MOUSEMOTION:
            self.mouse = ev.pos
            return
        if self.session is None:
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                self.app.go(HomeScreen(self.app, focus="bots"))
            elif ev.type == pygame.MOUSEBUTTONDOWN and self.hovering(self.rects.get("back")):
                self.app.go(HomeScreen(self.app, focus="bots"))
            return
        if self.record is not None:
            return                                  # game over, result coming up
        if self.confirm:
            self.handle_confirm(ev)
            return
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.mouse = ev.pos
            if self.hovering(self.rects.get("resign")):
                self.open_confirm("resign")
            elif self.hovering(self.rects.get("pause")):
                self.set_paused(not self.session.paused)
            else:
                self.click_cell(self.painter.cell_at(ev.pos))
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                self.open_confirm("leave")
            elif k in KEY_DIR:
                self.push(KEY_DIR[k], bool(ev.mod & pygame.KMOD_SHIFT))
            elif k == pygame.K_c:
                self.clear_path()
            elif k in (pygame.K_p, pygame.K_SPACE):
                self.set_paused(not self.session.paused)
            elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                ms = [m for _, m in SPEEDS]
                cur = min(range(len(ms)), key=lambda i: abs(ms[i] - self.session.tick_ms))
                cur = max(0, min(len(ms) - 1, cur + (-1 if k == pygame.K_LEFTBRACKET else 1)))
                self.session.set_speed(ms[cur])

    def handle_confirm(self, ev) -> None:
        go = (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER)) or \
             (ev.type == pygame.MOUSEBUTTONDOWN and self.hovering(self.rects.get("yes")))
        stay = (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE) or \
               (ev.type == pygame.MOUSEBUTTONDOWN and self.hovering(self.rects.get("no")))
        if go and self.confirm == "leave":
            self.leave()
        elif go and self.confirm == "resign":
            self.confirm = None
            self.set_paused(False)
            self.session.resign()
        elif stay:
            self.close_confirm()

    # -- clock --------------------------------------------------------------------
    def update(self, now_ms: float, dt_ms: float) -> None:
        if self.session is None:
            return
        if self.record is None:
            self.session.update(now_ms)
        self.view = self.session.view(self.selection if self.record is None else -1)
        self.stats = self.session.stats()
        if self.record is None:
            self.reconcile()
            if self.stats.get("done"):
                self._finish()
                self.done_at = now_ms
        elif self.done_at is not None and now_ms - self.done_at > 900:
            self.app.go(ResultScreen(self.app, self))

    # -- drawing ------------------------------------------------------------------
    def draw(self, s) -> None:
        W, H = s.get_size()
        s.fill(BG)
        P = self.PAD
        # board flush left with square cells; the sidebar takes the rest (>= SIDEBAR)
        bh_, bw_ = (self.session.h, self.session.w) if self.session else (23, 23)
        cell = max(8, min((W - self.SIDEBAR - 2 * P) // bw_, (H - 2 * P) // bh_))
        bw, bh = cell * bw_, cell * bh_
        self.board_rect = pygame.Rect(P, P + (H - 2 * P - bh) // 2, bw, bh)
        side_x = self.board_rect.right + P
        if self.session is None:
            self.draw_loading(s, self.board_rect)
        else:
            if self.painter.rect.size == (0, 0) or getattr(self, "_laid", None) != self.board_rect:
                self.painter.layout(self.board_rect)
                self._laid = self.board_rect.copy()
            self.painter.draw(s, self.view)
            if self.confirm:
                self.draw_confirm(s)
            elif self.record is not None:
                self.draw_banner(s, "Game over", "")
            elif self.session.paused:
                self.draw_banner(s, "Paused", "P to resume")
        self.draw_sidebar(s, pygame.Rect(side_x, 0, W - side_x, H))

    def draw_loading(self, s, r) -> None:
        x, y = r.x + 24, r.y + max(40, r.height // 3)
        if self.error:
            label(s, "Could not start the game", x, y)
            for k, line in enumerate(wrap(self.error, T.mono, r.width - 60)[:8]):
                put(s, line, T.mono, INK2, x, y + 28 + 18 * k)
            b = pygame.Rect(x, y + 200, 160, 40)
            self.rects["back"] = b
            button(s, b, "Back", hint="ESC", hover=self.hovering(b))
            return
        label(s, "Preparing match", x, y)
        put(s, f"Loading {self.bot.name}…", T.h1, INK, x, y + 24)
        secs = (pygame.time.get_ticks() - self.started) / 1000
        put(s, f"{secs:4.1f}s", T.mono, INK3, x, y + 64)
        # a quiet brass scanner instead of a spinner
        track = pygame.Rect(x, y + 92, min(360, r.width - 60), 2)
        pygame.draw.rect(s, LINE, track)
        pos = (pygame.time.get_ticks() // 4) % (track.width + 60) - 60
        seg = pygame.Rect(track.x + max(0, pos), track.y, 60 - max(0, -pos), 2).clip(track)
        pygame.draw.rect(s, BRASS, seg)

    def band_rect(self, h: int) -> pygame.Rect:
        br = self.painter.rect
        return pygame.Rect(br.x, br.centery - h // 2, br.width, h)

    def draw_banner(self, s, title: str, hint: str) -> None:
        r = self.band_rect(64)
        strip(s, r)
        put(s, title.upper(), T.h2, INK, r.x + 24, r.centery, "midleft", track=2.0)
        if hint:
            put(s, hint, T.mono, INK3, r.right - 24, r.centery, "midright")

    def draw_confirm(self, s) -> None:
        r = self.band_rect(132)
        strip(s, r, 240)
        x = r.x + 24
        if self.confirm == "leave":
            title, text, yes = "Leave this game?", "It will be recorded as a quit.", "Leave"
        else:
            title, text, yes = "Resign?", "The bot is credited with the win.", "Resign"
        label(s, "Confirm", x, r.y + 18)
        put(s, title, T.h1, INK, x, r.y + 36)
        put(s, text, T.body, INK2, x, r.y + 74)
        by = r.y + 34
        no = pygame.Rect(r.right - 24 - 170, by, 170, 40)
        yb = pygame.Rect(no.x - 12 - 170, by, 170, 40)
        self.rects["yes"], self.rects["no"] = yb, no
        if yb.x < x + 260:                         # narrow board: stack under the text
            yb.topleft, no.topleft = (x, r.y + 96 + 4), (x + 182, r.y + 96 + 4)
            yb.height = no.height = 28
        button(s, yb, yes, primary=True, hover=self.hovering(yb), hint="ENTER")
        button(s, no, "Keep playing", hover=self.hovering(no), hint="ESC")

    def draw_sidebar(self, s, panel: pygame.Rect) -> None:
        pygame.draw.rect(s, SURFACE, panel)
        pygame.draw.line(s, LINE, panel.topleft, panel.bottomleft)
        x, w = panel.x + 24, min(360, panel.width - 48)
        y = 22
        st = self.stats or {}
        names = self.names()
        leg = (self.pair or {}).get("leg", 0)
        label(s, f"Game {leg + 1} of 2" + ("  ·  sides swapped" if leg else ""), x, y)
        label(s, f"{self.bot.tier} bot" if self.bot.tier not in (None, "unknown") else "bot",
              x + w, y, INK3, "topright")
        y += 22

        # tick counter and progress hairline
        tick = int(st.get("tick", 0))
        max_t = int(getattr(self.session, "max_ticks", self.max_ticks))
        t = put(s, str(tick), T.mono_xl, INK, x, y)
        put(s, f"/ {max_t}", T.mono, INK3, t.right + 8, t.bottom - 18)
        if self.session is not None:
            if self.record is not None:
                state, ink = "OVER", INK2
            elif self.session.paused:
                state, ink = "PAUSED", BRASS
            else:
                state, ink = "LIVE", INK3
            label(s, state, x + w, y + 14, ink, "topright")
        y += 44
        pygame.draw.rect(s, LINE, pygame.Rect(x, y, w, 2))
        pygame.draw.rect(s, INK3, pygame.Rect(x, y, int(w * min(1.0, tick / max(1, max_t))), 2))
        y += 10
        ms = int(getattr(self.session, "tick_ms", self.tick_ms))
        speed = next((n for n, m in SPEEDS if m == ms), f"{ms}ms").lower()
        put(s, f"{ms} ms / tick  ·  {speed}", T.mono_s, INK3, x, y)
        put(s, "[ ]", T.mono_s, INK3, x + w, y, "topright")
        y += 26
        hline(s, x, x + w, y)
        y += 14

        # one block per seat, red first
        land, army, castles = st.get("land", [0, 0]), st.get("army", [0, 0]), st.get("castles", [0, 0])
        thinking = bool(st.get("bot_thinking"))
        for p in (0, 1):
            pygame.draw.rect(s, PLAYER[p], pygame.Rect(x, y + 4, 10, 10))
            tag = "you" if p == self.human else "bot"
            tag_r = label(s, tag, x + w, y + 3, INK3, "topright")
            put(s, fit(names[p], T.body_b, w - 22 - tag_r.width - 12), T.body_b, INK, x + 18, y)
            y += 24
            if p != self.human and thinking and self.record is None:
                dots = (pygame.time.get_ticks() // 300) % 4
                label(s, "thinking" + "." * dots, x + 18, y, BRASS)
            elif p != self.human:
                label(s, "ready", x + 18, y, INK3)
            else:
                sel = "tile selected" if self.selection >= 0 else "click a tile"
                label(s, sel, x + 18, y, INK3)
            y += 20
            cols = [("Land", land), ("Army", army), ("Castles", castles)]
            cw = (w - 18) // 3
            for k, (name, vals) in enumerate(cols):
                cx = x + 18 + k * cw
                label(s, name, cx, y, INK3)
                lead = int(vals[p]) > int(vals[1 - p])
                put(s, str(int(vals[p])), T.mono_l, INK if lead else INK2, cx, y + 16)
            y += 50
            hline(s, x, x + w, y)
            y += 14

        # finished-but-not-yet-switched note
        if st.get("done"):
            win = int(st.get("winner", -1))
            text = "Draw" if win < 0 else f"{names[win]} wins"
            put(s, fit(text, T.h2, w), T.h2, PLAYER[win] if win >= 0 else INK, x, y)
            y += 32

        # bottom: pause + resign; above them, as much of the cheat-sheet as fits
        bottom = panel.bottom - 22
        rb = pygame.Rect(x, bottom - 40, w, 40)
        pb = pygame.Rect(x, rb.y - 48, w, 40)
        self.rects["resign"], self.rects["pause"] = rb, pb
        live = self.session is not None and self.record is None
        button(s, pb, "Resume" if (self.session and self.session.paused) else "Pause",
               enabled=live, hover=live and self.hovering(pb), hint="P")
        button(s, rb, "Resign", enabled=live, hover=live and self.hovering(rb))

        keys = [("click", "select your tile"),
                ("arrows / wasd", "queue a move"),
                ("shift + move", "send half"),
                ("c", "clear queued path"),
                ("p", "pause / resume"),
                ("[  ]", "slower / faster"),
                ("esc", "leave the game")]
        room = pb.y - 16 - y - 22
        n = max(0, min(len(keys), room // 18))
        if n >= 3:
            label(s, "Controls", x, y, INK3)
            y += 22
            for key, what in keys[:n]:
                put(s, key, T.mono_s, BRASS, x, y)
                put(s, what, T.small, INK2, x + 108, y - 1)
                y += 18


# ----------------------------------------------------------------------------
class ResultScreen(Screen):
    END_TEXT = {
        "capture": "General captured on tick {t}.",
        "timeout": "Time ran out at tick {t}; most land wins.",
        "resign": "{p} resigned on tick {t}.",
        "quit": "{p} left the game on tick {t}.",
    }

    def __init__(self, app, game: GameScreen):
        super().__init__(app)
        self.game = game
        self.rects: dict = {}
        self.painter = game.painter
        self._laid = None
        st = game.stats or (game.session.stats() if game.session else {})
        rec = game.record if isinstance(game.record, dict) else {}
        hs = game.human
        # perspective: values as [human, bot]; prefer the engine's record
        def pair(key):
            if key in rec and isinstance(rec[key], (list, tuple)) and len(rec[key]) == 2:
                return [int(v) for v in rec[key]]
            v = st.get(key, [0, 0])
            return [int(v[hs]), int(v[1 - hs])]
        self.land, self.army, self.castles = pair("land"), pair("army"), pair("castles")
        if rec.get("winner") in ("human", "bot", "draw"):
            self.winner = rec["winner"]
        else:
            w = int(st.get("winner", -1))
            self.winner = "draw" if w < 0 else ("human" if w == hs else "bot")
        self.end = rec.get("end") or st.get("end") or ("timeout" if self.winner == "draw" else "capture")
        self.ticks = int(rec.get("ticks", st.get("tick", 0)))
        self.max_ticks = int(rec.get("max_ticks", getattr(game.session, "max_ticks", game.max_ticks)))
        self.duration = rec.get("duration_s")
        self.game_id = rec.get("id", "")
        self.error = rec.get("error", "")
        self.counted = not (self.end == "quit" and self.ticks < 25)
        self.pair = game.pair or {}
        self.leg = int(self.pair.get("leg", 1))
        res = self.pair.setdefault("results", [])
        if len(res) <= self.leg:
            res.append(self.winner)

    def handle(self, ev) -> None:
        if ev.type == pygame.MOUSEMOTION:
            self.mouse = ev.pos
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.mouse = ev.pos
            for key in ("next", "rematch", "new", "board", "home"):
                if self.hovering(self.rects.get(key)):
                    self.act(key)
                    return
        elif ev.type == pygame.KEYDOWN:
            first = "next" if self.leg == 0 else "rematch"
            act = {pygame.K_RETURN: first, pygame.K_KP_ENTER: first, pygame.K_r: first,
                   pygame.K_n: "new", pygame.K_l: "board", pygame.K_ESCAPE: "home"}.get(ev.key)
            if act:
                self.act(act)

    def act(self, what: str) -> None:
        a = self.app
        if what == "next":
            a.next_leg(self.pair)
        elif what == "rematch":
            a.start_game()
        elif what == "new":
            a.name = ""
            a.go(HomeScreen(a, focus="name"))
        elif what == "board":
            a.go(LeaderboardScreen(a))
        else:
            a.go(HomeScreen(a, focus="bots"))

    def draw(self, s) -> None:
        W, H = s.get_size()
        s.fill(BG)
        g = self.game
        hs = g.human
        names = g.names()
        human, bot = names[hs], names[1 - hs]

        # final position on the left, same painter as the game
        board = pygame.Rect(16, 16, int(W * 0.5) - 16, H - 32)
        if g.view is not None and self.painter is not None:
            if self._laid != board:
                self.painter.layout(board)
                self._laid = board.copy()
            self.painter.draw(s, g.view)
            x = self.painter.rect.right + 48
        else:
            x = 48
        w = min(520, W - x - 40)
        y = max(40, H // 2 - 300)

        label(s, "Result", x, y)
        if self.game_id:
            put(s, self.game_id, T.mono_s, INK3, x + w, y, "topright")
        y += 26
        if self.winner == "draw":
            title, ink = "Draw", INK
        elif self.winner == "human":
            title, ink = f"{human} wins", PLAYER[hs]
        else:
            title, ink = f"{bot} wins", PLAYER[1 - hs]
        f = T.huge if T.huge.size(title)[0] <= w else T.h1
        put(s, fit(title, f, w), f, ink, x - 2, y)
        y += f.get_height() + 6
        sub = self.END_TEXT.get(self.end, "{t} ticks.").format(t=self.ticks, p=human)
        if self.winner == "human" and self.end == "capture":
            sub = f"{human} captured the bot's general on tick {self.ticks}."
        elif self.winner == "bot" and self.end == "capture":
            sub = f"{bot} captured {human}'s general on tick {self.ticks}."
        put(s, fit(sub, T.h2, w), T.h2, INK2, x, y)
        y += 46

        # final counts, human vs bot
        c1, c2 = x + 120, x + 120 + (w - 120) // 2
        hline(s, x, x + w, y)
        y += 12
        pygame.draw.rect(s, PLAYER[hs], pygame.Rect(c1, y + 2, 9, 9))
        put(s, fit(human, T.body_b, c2 - c1 - 30), T.body_b, INK, c1 + 16, y - 2)
        pygame.draw.rect(s, PLAYER[1 - hs], pygame.Rect(c2, y + 2, 9, 9))
        put(s, fit(bot, T.body_b, x + w - c2 - 16), T.body_b, INK, c2 + 16, y - 2)
        y += 28
        for name, vals in (("Land", self.land), ("Army", self.army), ("Castles", self.castles)):
            label(s, name, x, y + 6, INK3)
            for cx, k in ((c1, 0), (c2, 1)):
                lead = vals[k] > vals[1 - k]
                put(s, str(vals[k]), T.mono_l, INK if lead else INK2, cx + 16, y)
            y += 34
        hline(s, x, x + w, y)
        y += 14
        dur = ""
        if isinstance(self.duration, (int, float)):
            dur = f"  ·  {int(self.duration) // 60}:{int(self.duration) % 60:02d} played"
        put(s, f"{self.ticks} / {self.max_ticks} ticks{dur}", T.mono, INK2, x, y)
        y += 22
        if self.error:
            put(s, fit(f"Not recorded: {self.error}", T.small, w), T.small, BRASS, x, y)
            y += 20
        elif not self.counted:
            put(s, "Not counted on the leaderboard (left before tick 25).", T.small, INK3, x, y)
            y += 20
        y += 26

        # the pair: game 1 always continues to game 2 with the seats swapped
        res = self.pair.get("results", [])
        if self.leg == 1 and len(res) >= 2:
            hw = sum(r == "human" for r in res[:2]) + 0.5 * sum(r == "draw" for r in res[:2])
            score = f"{hw:g} - {2 - hw:g}"
            put(s, fit(f"Pair on this map: {human} {score} {bot}", T.body_b, w), T.body_b, INK, x, y - 10)
            y += 22

        # actions
        rb = pygame.Rect(x, y, w, 54)
        if self.leg == 0:
            other = SEATS[1 - hs][0].lower()
            self.rects.pop("rematch", None)
            button(s, rb, f"Game 2 of 2: same map, you play {other}", primary=True,
                   hover=self.hovering(rb), hint="ENTER")
            self.rects["next"] = rb
        else:
            self.rects["rematch"] = rb
            self.rects.pop("next", None)
            button(s, rb, f"Rematch {bot} (new pair)", primary=True, hover=self.hovering(rb), hint="ENTER")
        y += 66
        half = (w - 12) // 2
        nb = pygame.Rect(x, y, half, 42)
        lb = pygame.Rect(x + half + 12, y, w - half - 12, 42)
        self.rects["new"], self.rects["board"] = nb, lb
        button(s, nb, "New player", hover=self.hovering(nb), hint="N")
        button(s, lb, "Leaderboard", hover=self.hovering(lb), hint="L")
        y += 54
        hb = pygame.Rect(x, y, w, 34)
        self.rects["home"] = hb
        put(s, "Back to bot menu", T.body, INK if self.hovering(hb) else INK3, x, y + 8)
        put(s, "ESC", T.mono_s, INK3, x + w, y + 10, "topright")


# ----------------------------------------------------------------------------
class LeaderboardScreen(Screen):
    ROW_H = 36
    TABS = ("bots", "players")

    def __init__(self, app, tab: str = "bots"):
        super().__init__(app)
        self.tab = tab
        self.sel = {"bots": 0, "players": 0}
        self.first = {"bots": 0, "players": 0}
        self.visible = 10
        self.rects: dict = {}
        self.row_rects: list = []
        self.last_click = (-1, 0)
        games = app.games
        self.rows = {"bots": store.bot_board(games), "players": store.player_board(games)}

    def cols(self):
        """(title, width or 0 for flex, align, cell(row, rank) -> (text, font, ink))."""
        if self.tab == "bots":
            return [
                ("#", 44, "r", lambda r, i: (str(i + 1), T.mono, INK3)),
                ("Bot", 0, "l", lambda r, i: (r["bot_name"], T.body_b, INK)),
                ("Games", 80, "r", lambda r, i: (str(r["games"]), T.mono, INK2)),
                ("Bot win", 90, "r", lambda r, i: (pct(r["win_rate"]), T.mono_b, INK)),
                ("Human wins", 110, "r", lambda r, i: (str(r["human_wins"]), T.mono, INK2)),
                ("Elo", 80, "r", lambda r, i: (str(r["elo"]), T.mono_b, INK)),
                ("Avg ticks", 100, "r", lambda r, i: (str(r["avg_ticks"]), T.mono, INK2)),
            ]
        return [
            ("#", 44, "r", lambda r, i: (str(i + 1), T.mono, INK3)),
            ("Player", 220, "l", lambda r, i: (r["player"], T.body_b, INK)),
            ("Games", 70, "r", lambda r, i: (str(r["games"]), T.mono, INK2)),
            ("W", 50, "r", lambda r, i: (str(r["wins"]), T.mono_b, INK)),
            ("L", 50, "r", lambda r, i: (str(r["losses"]), T.mono, INK2)),
            ("D", 50, "r", lambda r, i: (str(r["draws"]), T.mono, INK2)),
            ("Elo", 80, "r", lambda r, i: (str(r["elo"]), T.mono_b, INK)),
            ("Bots beaten", 0, "l", lambda r, i: (
                (f"{len(r['bots_beaten'])}  " + ", ".join(r["bots_beaten"])) if r["bots_beaten"] else "—",
                T.small, INK2 if r["bots_beaten"] else INK3)),
        ]

    def current(self):
        rows = self.rows[self.tab]
        k = self.sel[self.tab]
        return rows[k] if 0 <= k < len(rows) else None

    def move(self, step: int) -> None:
        rows = self.rows[self.tab]
        if not rows:
            return
        k = max(0, min(len(rows) - 1, self.sel[self.tab] + step))
        self.sel[self.tab] = k
        f = self.first[self.tab]
        if k < f:
            f = k
        elif k >= f + self.visible:
            f = k - self.visible + 1
        self.first[self.tab] = f

    def choose(self) -> None:
        """Bots tab: back to Home with this bot picked. Players tab: as this player."""
        row = self.current()
        a = self.app
        if row is None:
            return
        if self.tab == "bots":
            a.bot_id = row["bot_id"]
            a.go(HomeScreen(a, focus="name" if not a.name.strip() else "play"))
        else:
            a.name = row["player"]
            a.go(HomeScreen(a, focus="bots"))

    def handle(self, ev) -> None:
        if ev.type == pygame.MOUSEMOTION:
            self.mouse = ev.pos
        elif ev.type == pygame.MOUSEWHEEL:
            n = len(self.rows[self.tab])
            self.first[self.tab] = max(0, min(self.first[self.tab] - ev.y, max(0, n - self.visible)))
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.mouse = ev.pos
            for tab in self.TABS:
                if self.hovering(self.rects.get(tab)):
                    self.tab = tab
                    return
            if self.hovering(self.rects.get("back")):
                self.app.go(HomeScreen(self.app, focus="bots"))
                return
            if self.hovering(self.rects.get("play")):
                self.choose()
                return
            for k, r in self.row_rects:
                if r.collidepoint(ev.pos):
                    now = pygame.time.get_ticks()
                    double = self.last_click[0] == k and now - self.last_click[1] < DOUBLE_CLICK_MS
                    self.sel[self.tab] = k
                    self.last_click = (k, now)
                    if double:
                        self.choose()
                    return
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                self.app.go(HomeScreen(self.app, focus="bots"))
            elif k in (pygame.K_TAB, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_1, pygame.K_2):
                self.tab = {pygame.K_1: "bots", pygame.K_2: "players"}.get(
                    k, "players" if self.tab == "bots" else "bots")
            elif k in (pygame.K_UP, pygame.K_DOWN):
                self.move(-1 if k == pygame.K_UP else 1)
            elif k in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                self.move(self.visible * (-1 if k == pygame.K_PAGEUP else 1))
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_p):
                self.choose()

    def draw(self, s) -> None:
        W, H = s.get_size()
        s.fill(BG)
        M = 32 if W >= 1100 else 22
        games = len(self.app.games)
        header(s, W, "Leaderboard", f"{games} GAME{'S' * (games != 1)} ON RECORD · QUITS BEFORE TICK 25 NOT COUNTED", M)
        x, w = M + 12, W - 2 * M - 12
        y = 76

        # tabs: text with a brass rule under the active one
        tx = x
        for tab in self.TABS:
            n = len(self.rows[tab])
            t = render(tab.upper(), T.h2, INK if tab == self.tab else INK3, 2.0)
            r = pygame.Rect(tx, y, t.get_width() + 46, 38)
            self.rects[tab] = r
            s.blit(t, (tx, y + 6))
            put(s, str(n), T.mono, INK3, tx + t.get_width() + 10, y + 9)
            if tab == self.tab:
                pygame.draw.rect(s, BRASS, pygame.Rect(tx, y + 35, r.width - 14, 3))
            tx = r.right + 24

        # actions top right
        back = pygame.Rect(x + w - 120, y - 2, 120, 36)
        self.rects["back"] = back
        button(s, back, "Back", hover=self.hovering(back), hint="ESC")
        row = self.current()
        if self.tab == "bots":
            play_txt = "Play this bot"
        else:
            play_txt = "Play as player"
        play = pygame.Rect(back.x - 12 - 210, y - 2, 210, 36)
        self.rects["play"] = play if row is not None else None
        if row is not None:
            button(s, play, play_txt, primary=True, hover=self.hovering(play), hint="ENTER")
        y += 38
        hline(s, x, x + w, y)
        y += 16

        rows = self.rows[self.tab]
        cols = self.cols()
        fixed = sum(c[1] for c in cols)
        flex = max(120, (w - 16 - fixed) // max(1, sum(1 for c in cols if c[1] == 0)))
        xs, cx = [], x + 16
        for title, cw, align, _ in cols:
            cw = cw or flex
            xs.append((cx, cw, align))
            cx += cw
        # header
        for (cx, cw, align), (title, *_rest) in zip(xs, cols):
            if align == "r":
                label(s, title, cx + cw - 12, y, INK3, "topright")
            else:
                label(s, title, cx + 16, y, INK3)
        y += 22
        hline(s, x, x + w, y)

        self.row_rects = []
        if not rows:
            y += 28
            put(s, "No games recorded yet.", T.h1, INK, x + 16, y)
            put(s, "Play a game from the home screen and it appears here, ranked by how often each bot wins.",
                T.body, INK2, x + 16, y + 40)
            put(s, "ESC back to the bot menu", T.mono, INK3, x + 16, y + 70)
            return

        body = pygame.Rect(x, y + 1, w, H - y - 60)
        self.visible = max(1, body.height // self.ROW_H)
        first = self.first[self.tab] = max(0, min(self.first[self.tab], max(0, len(rows) - self.visible)))
        sel = self.sel[self.tab]
        for k in range(first, min(len(rows), first + self.visible)):
            ry = body.y + (k - first) * self.ROW_H
            r = pygame.Rect(x, ry, w, self.ROW_H)
            self.row_rects.append((k, r))
            if k == sel:
                pygame.draw.rect(s, ROW_SEL, r)
                bar(s, x, ry, self.ROW_H - 1)
            elif self.hovering(r):
                pygame.draw.rect(s, ROW_HOVER, r)
            for (cx, cw, align), col in zip(xs, cols):
                text, f, ink = col[3](rows[k], k)
                if align == "r":
                    put(s, fit(text, f, cw - 16), f, ink, cx + cw - 12, r.centery, "midright")
                else:
                    put(s, fit(text, f, cw - 24), f, ink, cx + 16, r.centery, "midleft")
            hline(s, x, x + w, ry + self.ROW_H - 1)
        scrollbar(s, x + w + 6, body.y, self.visible * self.ROW_H, first, self.visible, len(rows))
        hint = ("↑↓ select   ENTER play this bot   double-click   TAB players   ESC back"
                if self.tab == "bots" else
                "↑↓ select   ENTER play as this player   TAB bots   ESC back")
        put(s, hint, T.mono_s, INK3, x + 16, H - 34)


# ----------------------------------------------------------------------------
class TutorialScreen(Screen):
    BAR = 48
    TITLES = {"rules": "How the game works", "controls": "Controls"}

    def __init__(self, app, chapter: str):
        super().__init__(app)
        self.chapter = chapter
        self.tut = None
        self.error = ""
        self.rect = None
        self.rects: dict = {}

    def build(self, rect: pygame.Rect) -> None:
        self.rect = rect.copy()
        if self.tut is not None:
            # keep the scene/time on resize: re-target the existing player
            if hasattr(self.tut, "layout"):
                self.tut.layout(rect)
                return
            if hasattr(self.tut, "rect"):
                self.tut.rect = rect.copy()
                if hasattr(self.tut, "_fonts"):
                    self.tut._fonts()
                return
        try:
            from arena.tutorial import Tutorial
            self.tut = Tutorial(rect, chapter=self.chapter)
        except Exception as e:  # noqa: BLE001 - shown on screen
            self.tut = None
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    def handle(self, ev) -> None:
        if ev.type == pygame.MOUSEMOTION:
            self.mouse = ev.pos
        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self.app.go(HomeScreen(self.app))
            return
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.hovering(self.rects.get("back")):
            self.app.go(HomeScreen(self.app))
            return
        if self.tut is not None:
            try:
                self.tut.handle(ev)
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                self.tut = None

    def update(self, now_ms: float, dt_ms: float) -> None:
        if self.tut is not None:
            try:
                self.tut.update(dt_ms)
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                self.tut = None

    def draw(self, s) -> None:
        W, H = s.get_size()
        s.fill(BG)
        area = pygame.Rect(0, self.BAR + 1, W, H - self.BAR - 1)
        if self.rect != area and not self.error:
            self.build(area)
        if self.tut is not None:
            try:
                self.tut.draw(s)
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                self.tut = None
        if self.tut is None:
            x, y = 44, self.BAR + 80
            label(s, "Tutorial unavailable", x, y)
            put(s, "The tutorial module (arena/tutorial.py) could not be loaded.", T.h2, INK, x, y + 26)
            for k, line in enumerate(wrap(self.error or "unknown error", T.mono, W - 2 * x)[:8]):
                put(s, line, T.mono, INK3, x, y + 64 + 18 * k)

        # slim top bar, drawn last so it always sits above the animation
        pygame.draw.rect(s, BG, pygame.Rect(0, 0, W, self.BAR))
        hline(s, 0, W, self.BAR)
        x = 24
        r = label(s, "Fogless Generals", x, 18)
        put(s, "/", T.label, INK3, r.right + 10, 18)
        r = label(s, self.TITLES.get(self.chapter, self.chapter), r.right + 24, 18, INK)
        if self.tut is not None and getattr(self.tut, "done", False):
            label(s, "finished", r.right + 18, 18, BRASS)
        back = pygame.Rect(W - 24 - 110, 8, 110, 32)
        self.rects["back"] = back
        button(s, back, "Back", hover=self.hovering(back), hint="ESC")
        hint = "SPACE pause   ← → previous / next scene"
        if back.x - 24 - T.mono_s.size(hint)[0] > r.right + 120:
            put(s, hint, T.mono_s, INK3, back.x - 24, 24, "midright")


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, size=(1280, 800)):
        pygame.init()
        pygame.key.set_repeat(260, 45)
        self.surface = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption("Fogless Generals · Arena")
        # No minimum window size: pygame._sdl2's Window wrapper destroys the
        # display window when garbage-collected (bus error), so layouts instead
        # tighten themselves for short / narrow windows.
        T.load()
        self.clock = pygame.time.Clock()
        self.running = True
        # shared choices, kept across screens and rematches
        self.name = ""
        self.bot_id: str | None = None
        self.seat = 0                    # side in game 1; game 2 swaps it
        self.max_ticks = 600
        self.tick_ms = 350
        self.bots = None                 # list[BotInfo] once loaded
        self.bots_error = ""
        self.games: list = []
        self.bot_stats: dict = {}
        self.recent: list[str] = []
        self.refresh_games()
        threading.Thread(target=self._load_bots, daemon=True).start()
        self.screen: Screen = HomeScreen(self)
        self._last = pygame.time.get_ticks()
        self._apply_text_input()

    # -- data -------------------------------------------------------------------
    def _load_bots(self) -> None:
        try:
            from arena.bots import list_bots
            bots = list(list_bots(include_broken=True))
            # final first, broken last, otherwise the engine's order
            order = {b.id: k for k, b in enumerate(bots)}
            bots.sort(key=lambda b: (not is_final(b), b.tier == "broken", order[b.id]))
            if self.bot_id is None or all(b.id != self.bot_id for b in bots):
                playable = [b for b in bots if b.tier != "broken"]
                self.bot_id = playable[0].id if playable else None
            self.bots = bots
        except Exception as e:  # noqa: BLE001 - shown on the home screen
            self.bots_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    def refresh_games(self) -> None:
        """Reload the results file: bot stats for the menu and recent names."""
        try:
            self.games = store.load_games()
        except Exception:  # noqa: BLE001
            self.games = []
        try:
            self.bot_stats = {r["bot_id"]: r for r in store.bot_board(self.games)}
        except Exception:  # noqa: BLE001
            self.bot_stats = {}
        seen, recent = set(), []
        for g in sorted(self.games, key=lambda g: g.get("ended_at", ""), reverse=True):
            name = str(g.get("player", "")).strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                recent.append(name)
        self.recent = recent[:10]

    def bot_info(self, bot_id: str):
        for b in self.bots or []:
            if b.id == bot_id:
                return b
        from arena.bots import get_bot
        return get_bot(bot_id)

    # -- navigation ----------------------------------------------------------------
    def go(self, screen: Screen) -> None:
        self.screen = screen
        screen.mouse = pygame.mouse.get_pos()
        screen.enter()
        self._apply_text_input()

    def _apply_text_input(self) -> None:
        if self.screen.text_input:
            pygame.key.start_text_input()
        else:
            pygame.key.stop_text_input()

    def start_game(self) -> None:
        if not self.name.strip() or not self.bot_id:
            return
        try:
            bot = self.bot_info(self.bot_id)
        except Exception as e:  # noqa: BLE001
            if isinstance(self.screen, HomeScreen):
                self.screen.say(f"Unknown bot {self.bot_id}: {e}")
            return
        from arena import maps, store
        pair = {"id": store.new_game_id(), "seed": maps.random_seed(), "first": self.seat,
                "leg": 0, "results": []}
        self.go(GameScreen(self, bot, self.name.strip(), self.seat, self.max_ticks, self.tick_ms, pair))

    def next_leg(self, pair: dict) -> None:
        """Game 2 of a pair: the same map with the seats swapped."""
        if not pair or pair.get("leg", 0) != 0 or not self.name.strip() or not self.bot_id:
            self.start_game()
            return
        nxt = dict(pair, leg=1)
        bot = self.bot_info(self.bot_id)
        self.go(GameScreen(self, bot, self.name.strip(), 1 - pair["first"], self.max_ticks,
                           self.tick_ms, nxt))

    # -- loop -----------------------------------------------------------------------
    def frame(self) -> None:
        """One frame: events, update, draw. Never blocks beyond the 60 fps tick."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.VIDEORESIZE:
                pass                     # surfaces are re-read every frame
            else:
                self.screen.handle(ev)
        now = pygame.time.get_ticks()
        dt, self._last = now - self._last, now
        self.screen.update(now, dt)
        self.surface = pygame.display.get_surface()
        self.screen.draw(self.surface)
        pygame.display.flip()
        self.clock.tick(60)

    def run(self) -> None:
        try:
            while self.running:
                self.frame()
        finally:
            self.screen.close()
            pygame.quit()


def main(size=(1280, 800)) -> None:
    App(size).run()
