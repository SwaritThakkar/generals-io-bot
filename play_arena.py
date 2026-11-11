#!/usr/bin/env python3
"""Launch the arena: a friend types their name, picks a bot, and plays it.

    python play_arena.py                    # 1280x800 window (resizable)
    python play_arena.py --window 1440x900
    python play_arena.py --smoke            # headless: drive every screen, save
                                            # screenshots to /tmp/arena_shots/*.png

Results go to arena_results/ (see arena/store.py). The smoke run points the
store at a scratch folder next to the screenshots, so it never touches real
results.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# headless smoke run
# --------------------------------------------------------------------------- #
def smoke(outdir: pathlib.Path, size=(1280, 800)) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    outdir.mkdir(parents=True, exist_ok=True)

    from arena import store
    scratch = outdir / "results"
    store.RESULTS, store.GAMES, store.REPLAYS = scratch, scratch / "games.jsonl", scratch / "replays"

    import pygame
    from arena import app as arena_app

    app = arena_app.App(size)

    def frames(n=1):
        for _ in range(n):
            app.frame()

    def until(cond, timeout_s=60.0, what="condition"):
        t0 = time.time()
        while not cond():
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"smoke: timed out waiting for {what}")
            app.frame()

    def key(k, mod=0, n=2):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode="", scancode=0))
        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=k, mod=mod, unicode="", scancode=0))
        frames(n)

    def type_text(text):
        for ch in text:
            pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text=ch))
        frames(2)

    def click(pos, n=2):
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1))
        frames(n)

    def shot(name):
        app.frame()
        path = outdir / f"{name}.png"
        pygame.image.save(app.surface, str(path))
        print("saved", path)

    def resize(w, h):
        pygame.display.set_mode((w, h), pygame.RESIZABLE)
        frames(3)

    until(lambda: app.bots is not None or app.bots_error, 30, "bot list")
    shot("01_home_empty")
    type_text("Swarit")
    key(pygame.K_DOWN)
    key(pygame.K_DOWN)
    key(pygame.K_TAB)                        # focus moves along
    shot("02_home_ready")

    key(pygame.K_F3)
    shot("03_leaderboard_empty")
    key(pygame.K_ESCAPE)

    key(pygame.K_F1)
    frames(90)
    shot("04_tutorial_rules")
    key(pygame.K_RIGHT, n=60)
    shot("05_tutorial_rules_next")
    key(pygame.K_ESCAPE)
    key(pygame.K_F2)
    frames(120)
    shot("06_tutorial_controls")
    key(pygame.K_ESCAPE)

    key(pygame.K_PAGEUP)                     # back to the top of the list: final
    key(pygame.K_RETURN, n=1)
    shot("07_game_loading")
    game = app.screen
    until(lambda: getattr(game, "session", None) is not None or getattr(game, "error", ""), 60, "session")
    if game.error:
        shot("07_game_error")
        raise SystemExit(f"smoke: session failed: {game.error}")
    frames(10)
    # click our general, then queue a short path (one move sent as half)
    view = game.session.view()
    gen = next(i for i in range(len(view.owner))
               if view.general[i] and int(view.owner[i]) == game.human)
    click(game.painter.centre(gen))
    for k in (pygame.K_RIGHT, pygame.K_RIGHT, pygame.K_DOWN):
        key(k, n=1)
    key(pygame.K_DOWN, mod=pygame.KMOD_SHIFT, n=1)
    shot("08_game_queued")
    t0 = time.time()
    while time.time() - t0 < 4.0:            # let a dozen ticks play in real time
        app.frame()
    shot("09_game_running")
    key(pygame.K_p)
    shot("10_game_paused")
    key(pygame.K_p)
    key(pygame.K_ESCAPE)
    shot("11_game_confirm_leave")
    key(pygame.K_ESCAPE)                     # keep playing

    resize(1024, 660)
    shot("12_game_small_window")
    resize(1280, 800)

    # resign through the sidebar button
    click(game.rects["resign"].center)
    shot("13_game_confirm_resign")
    key(pygame.K_RETURN)
    until(lambda: isinstance(app.screen, arena_app.ResultScreen), 10, "result screen")
    shot("14_result")

    key(pygame.K_l)
    shot("15_leaderboard_bots")
    key(pygame.K_TAB)
    shot("16_leaderboard_players")
    key(pygame.K_TAB)
    key(pygame.K_RETURN)                     # bot row -> home with it preselected
    shot("17_home_from_leaderboard")

    resize(1000, 640)
    shot("18_home_small_window")
    app.screen.close()
    pygame.quit()
    print("smoke ok")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default="1280x800", help="initial window size, WxH")
    ap.add_argument("--smoke", action="store_true", help="headless run that saves screenshots")
    ap.add_argument("--shots", default="/tmp/arena_shots", help="screenshot folder for --smoke")
    args = ap.parse_args()
    try:
        size = tuple(int(v) for v in args.window.lower().split("x"))
        assert len(size) == 2
    except (ValueError, AssertionError):
        raise SystemExit("--window must look like 1280x800")

    if args.smoke:
        smoke(pathlib.Path(args.shots), size)
        return
    from arena.app import main as run
    run(size)


if __name__ == "__main__":
    main()
