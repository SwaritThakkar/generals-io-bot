#!/usr/bin/env python3
"""Render docs/media/demo.gif: one real game of the final bot (v51, red) against a
collect-and-charge opponent (blue), with what the bot is thinking on the right.

    python docs/media/make_demo_gif.py

The game is played live in the exact simulator (seed 93003, 600-tick cap); every
board is drawn with the arena's own painter.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "models"), str(ROOT / "models" / "tools36")]

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from arena.board_draw import (BG, BRASS, INK, INK2, INK3, LINE, PLAYER, SURFACE,  # noqa: E402
                              BoardPainter, BoardView, lighten, pick_font)

OUT = pathlib.Path(__file__).with_name("demo.gif")
W, H = 980, 560
BOARD = 540
SEED, CAP = 93003, 600

STYLE = {"rusher": "rusher", "castle_rusher": "castle rusher", "expander": "land grabber", "turtle": "turtle",
         "gather_early": "builds up, attacks early", "gather_mid": "builds up, attacks mid-game",
         "gather_late": "builds up, then attacks", "spread": "spreads and collects", "econ": "economy player",
         "planner": "planner"}


def doing(why: str, phase_tick: int) -> tuple[str, str]:
    """(headline, detail) in plain words for the bot's current reason."""
    tag = why.split(":")[-1]
    if tag == "fast_defence":
        return "Defending", "an attack is coming: every move pulls soldiers into the general (Steiner tree)"
    if tag == "capture":
        return "Capture", "a stack next to the enemy general beats it"
    if tag in ("rush", "rush_walk"):
        return "Proven attack", "the walk beats the enemy general against every reply"
    if tag.startswith("castle"):
        return "Taking a castle", "it pays back before the game ends and home stays safe"
    if tag in ("strike", "gather_finish"):
        return "Massing for a strike", "collecting on the castle nearest the enemy general"
    if tag in ("defend", "defender", "fallback_defender", "consolidate_home", "gather_home"):
        return "Guarding home", "the safety score is low: soldiers come home"
    if tag == "counter":
        return "Counter-strike", "home is losing the race: hit their general first"
    if phase_tick < 60:
        return "Opening", "grab free land, one square per move"
    return "Expanding", "nothing threatens the general: grab land, plan castles"


def record():
    spec = importlib.util.spec_from_file_location("v51_demo", ROOT / "models" / "v51_fast_defence.py")
    v51 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v51)
    import adversaries38
    from arena36 import start
    ctx, sim = start(SEED, CAP)
    me = v51.Planner51(ctx, 0)
    op = adversaries38.GatherRusher(ctx, 1, t0=200)
    frames = []
    while not sim.done:
        pre = (np.array(sim.owner), np.array(sim.army), list(sim.gen))
        a = me.act(sim)
        a = a if a and a > 0 else 0
        b = op.act(sim)
        b = b if b and b > 0 else 0
        land, tot = sim.totals()
        frames.append({"t": sim.step, "owner": pre[0], "army": pre[1], "gen": pre[2], "a": a, "b": b,
                       "why": str(me.why), "m0": me.margin, "def": list(me.def_moves),
                       "inc": me.incoming, "tp": dict(me.type_p), "land": land, "tot": tot,
                       "castles": [sum(1 for c in ctx.castles if sim.owner[c] == q) for q in (0, 1)]})
        sim.step2(a, b)
    last = frames[-1]
    final = {"t": sim.step, "owner": np.array(sim.owner), "army": np.array(sim.army), "gen": list(sim.gen),
             "a": 0, "b": 0, "why": "capture", "m0": last["m0"], "def": [], "inc": None, "tp": last["tp"],
             "land": sim.totals()[0], "tot": sim.totals()[1], "castles": last["castles"]}
    frames.append(final)
    return ctx, frames, sim.winner


def decode(a):
    cell, k = divmod(a, 9)
    return cell, (k - 1) % 4, k >= 5


class Renderer:
    def __init__(self, ctx):
        pygame.init()
        b = ctx.b
        self.h, self.w = b.h, b.w
        self.mountain = np.array(b.mountain, bool)
        self.castle = np.array(b.castle, bool)
        self.painter = BoardPainter(b.h, b.w)
        self.painter.layout(pygame.Rect(10, 10, BOARD, BOARD))
        self.f_title = pick_font(15, bold=True)
        self.f_tick = pick_font(44, bold=True)
        self.f_label = pick_font(11, bold=True)
        self.f_body = pick_font(15)
        self.f_head = pick_font(22, bold=True)
        self.f_small = pick_font(12)
        self.f_mono = pick_font(13, mono=True, bold=True)
        self.f_banner = pick_font(30, bold=True)
        self.margins = []

    def text(self, surf, s, font, colour, x, y, wrap=None):
        if wrap is None:
            surf.blit(font.render(s, True, colour), (x, y))
            return y + font.get_linesize()
        words, line = s.split(), ""
        for w_ in words:
            trial = (line + " " + w_).strip()
            if font.size(trial)[0] > wrap and line:
                surf.blit(font.render(line, True, colour), (x, y))
                y += font.get_linesize()
                line = w_
            else:
                line = trial
        if line:
            surf.blit(font.render(line, True, colour), (x, y))
            y += font.get_linesize()
        return y

    def bar(self, surf, x, y, w, a, b, label):
        self.text(surf, label, self.f_label, INK3, x, y)
        tot = max(1, a + b)
        y += 16
        pygame.draw.rect(surf, PLAYER[0], (x, y, int(w * a / tot), 10), border_radius=3)
        pygame.draw.rect(surf, PLAYER[1], (x + int(w * a / tot), y, w - int(w * a / tot), 10), border_radius=3)
        self.text(surf, f"{a:,}", self.f_mono, lighten(PLAYER[0], .35), x, y + 14)
        s = f"{b:,}"
        self.text(surf, s, self.f_mono, lighten(PLAYER[1], .35), x + w - self.f_mono.size(s)[0], y + 14)
        return y + 36

    def frame(self, f, caption=None):
        surf = pygame.Surface((W, H))
        surf.fill(BG)
        gen = np.zeros(self.h * self.w, bool)
        for g in f["gen"]:
            if g >= 0:
                gen[g] = True
        moves = []
        for a, p in ((f["a"], 0), (f["b"], 1)):
            if a:
                c, d, half = decode(a)
                moves.append((c, d, half, p))
        self.painter.draw(surf, BoardView(self.h, self.w, f["owner"], f["army"], self.mountain, self.castle,
                                          gen, last_moves=moves))
        over = pygame.Surface((W, H), pygame.SRCALPHA)
        if f["why"].endswith("fast_defence"):
            for child, par in f["def"][:20]:
                x0, y0 = self.painter.centre(child)
                x1, y1 = self.painter.centre(par)
                pygame.draw.line(over, (*BRASS, 230), (x0, y0), (x0 + (x1 - x0) * .8, y0 + (y1 - y0) * .8), 4)
        surf.blit(over, (0, 0))
        g0 = f["gen"][0]
        if g0 >= 0 and (f["why"].endswith("fast_defence") or f["why"].endswith("capture")):
            pygame.draw.rect(surf, BRASS, self.painter.cell_rect(g0 if f["why"].endswith("fast_defence")
                                                                  else max(f["gen"][1], 0)).inflate(8, 8), 3,
                             border_radius=4)

        # ---- side panel ---------------------------------------------------------
        x, pw = BOARD + 30, W - BOARD - 44
        pygame.draw.rect(surf, SURFACE, (x - 12, 10, pw + 24, BOARD), border_radius=10)
        y = 24
        pygame.draw.circle(surf, PLAYER[0], (x + 6, y + 9), 6)
        self.text(surf, "v51 (our bot)", self.f_title, INK, x + 18, y)
        pygame.draw.circle(surf, PLAYER[1], (x + 160, y + 9), 6)
        self.text(surf, "collect-and-charge bot", self.f_title, INK2, x + 172, y)
        y += 34
        self.text(surf, "TICK", self.f_label, INK3, x, y)
        self.text(surf, str(f["t"]), self.f_tick, BRASS, x, y + 12)
        gx = x + 150
        self.text(surf, "CASTLES", self.f_label, INK3, gx, y)
        self.text(surf, f"{f['castles'][0]}", self.f_head, lighten(PLAYER[0], .3), gx, y + 22)
        self.text(surf, f"{f['castles'][1]}", self.f_head, lighten(PLAYER[1], .3), gx + 40, y + 22)
        g1 = f["gen"][1]
        self.text(surf, "GENERALS", self.f_label, INK3, gx + 110, y)
        ga = int(f["army"][g0]) if g0 >= 0 else 0
        gb = int(f["army"][g1]) if g1 >= 0 and f["owner"][g1] == 1 else 0
        self.text(surf, f"{ga}", self.f_head, lighten(PLAYER[0], .3), gx + 110, y + 22)
        self.text(surf, f"{gb}" if gb else "taken", self.f_head, lighten(PLAYER[1], .3), gx + 165, y + 22)
        y += 72
        y = self.bar(surf, x, y, pw, int(f["tot"][0]), int(f["tot"][1]), "ARMY")
        y = self.bar(surf, x, y + 4, pw, int(f["land"][0]), int(f["land"][1]), "LAND")
        y += 8
        pygame.draw.line(surf, LINE, (x, y), (x + pw, y))
        y += 14
        head, detail = doing(f["why"], f["t"])
        self.text(surf, "WHAT THE BOT IS DOING", self.f_label, BRASS, x, y)
        y = self.text(surf, head, self.f_head, INK, x, y + 16)
        y = self.text(surf, detail, self.f_body, INK2, x, y + 2, wrap=pw)
        y += 14
        top = max(f["tp"], key=f["tp"].get)
        self.text(surf, "ITS GUESS ABOUT THE OPPONENT", self.f_label, BRASS, x, y)
        y = self.text(surf, STYLE.get(top, top), self.f_body, INK, x, y + 16)
        p = f["tp"][top]
        pygame.draw.rect(surf, LINE, (x, y + 4, pw - 50, 8), border_radius=3)
        pygame.draw.rect(surf, BRASS, (x, y + 4, int((pw - 50) * p), 8), border_radius=3)
        self.text(surf, f"{p:.0%}", self.f_mono, INK2, x + pw - 40, y - 1)
        y += 26
        m = f["m0"]
        if m is not None and m < 10 ** 8:
            self.margins.append((f["t"], m))
        self.text(surf, "SAFETY SCORE OF ITS GENERAL", self.f_label, BRASS, x, y)
        y += 18
        hist = [mm for t, mm in self.margins if t <= f["t"]][-120:]
        ch = 46
        pygame.draw.rect(surf, BG, (x, y, pw, ch), border_radius=4)
        if len(hist) > 1:
            lo, hi = 0, max(60, max(hist))
            pts = [(x + int(i * (pw - 1) / (len(hist) - 1)), y + ch - 2 - int((min(max(v, lo), hi) - lo) / (hi - lo) * (ch - 4)))
                   for i, v in enumerate(hist)]
            pygame.draw.lines(surf, BRASS, False, pts, 2)
        if caption:
            band = pygame.Surface((BOARD, 58), pygame.SRCALPHA)
            band.fill((11, 16, 18, 225))
            surf.blit(band, (10, 10))
            pygame.draw.line(surf, BRASS, (10, 67), (10 + BOARD, 67), 2)
            t = self.f_banner.render(caption, True, BRASS)
            surf.blit(t, (10 + (BOARD - t.get_width()) // 2, 10 + 29 - t.get_height() // 2))
        return Image.frombytes("RGB", (W, H), pygame.image.tostring(surf, "RGB"))


def main():
    ctx, frames, winner = record()
    by_t = {f["t"]: f for f in frames}
    end = frames[-1]["t"]
    inc = [f["t"] for f in frames if f["why"].endswith("fast_defence")]
    d0 = d1 = inc[0] if inc else end
    for t in inc[1:]:                           # first defence episode only
        if t - d1 > 6:
            break
        d1 = t
    print(f"winner {winner}, end tick {end}, fast defence ticks {d0}-{d1}")
    r = Renderer(ctx)
    plan = []                                   # (tick, duration ms, caption)
    for t in range(0, end + 1):
        if t not in by_t:
            continue
        busy = d0 - 12 <= t <= d1 + 3 or t >= end - 6
        if busy or t % 4 == 0:
            plan.append((t, 90 if busy else 55, None))
    # hold the key moments
    plan = [(t, ms, None) for t, ms, _ in plan]
    # the attack lands where our general's army drops sharply after defence began
    hit = next((f["t"] for f, g in zip(frames[1:], frames) if f["t"] > d0 and f["gen"][0] >= 0
                and g["army"][g["gen"][0]] - f["army"][f["gen"][0]] > 25), d1)
    plan = [p for p in plan if p[0] <= hit + 3 or p[0] > hit + 3]
    for t in range(d1, hit + 4):
        if t in by_t and all(p[0] != t for p in plan):
            plan.append((t, 90, None))
    plan.append((hit, 1700, "Attack absorbed: general keeps " + str(int(by_t[hit]["army"][by_t[hit]["gen"][0]]))))
    plan.sort(key=lambda p: (p[0], p[2] is not None))
    plan.append((end, 2800, "Enemy general captured"))
    imgs, durs = [], []
    for t, ms, cap in plan:
        if t in by_t:
            imgs.append(r.frame(by_t[t], cap))
            durs.append(ms)
    pal = imgs[len(imgs) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(OUT, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True, disposal=1)
    print(f"wrote {OUT} ({len(q)} frames, {sum(durs) / 1000:.1f} s, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
