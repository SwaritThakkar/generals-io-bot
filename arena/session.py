"""One live arena game: a human seat with a move queue against one bot.

    s = GameSession("v46_opponent_learner", player="Asha", human_seat=0)
    while running:
        s.update(pygame.time.get_ticks())       # resolves at most one tick
        painter.draw(screen, s.view(selection))
    s.finish("quit")                             # if the window closes early

Timing model:

* The bot computes its move for tick t on a worker thread (on a copy of the
  Sim) as soon as tick t-1 resolves, so the UI thread never runs bot code.
* `update(now_ms)` resolves tick t once `tick_ms` has passed since tick t-1
  AND the bot's move is ready. A slow bot delays the tick, not the frame;
  while the tick is held, stats()["bot_thinking"] is True.
* A bot that needs more than `max_think_ms` for one move passes that tick
  (it keeps thinking; its late answer is dropped and it is asked again for
  the then-current tick), so a hung bot cannot freeze the game.

Human queue (as on generals.io): each tick pops the first queued move; a
move that is not legal right now (source not ours or army < 2) is discarded
and the next one tried, so a stale queue never blocks the game.

The game ends at a general capture or at `max_ticks` (sim36 applies the
land-then-army tiebreak); `finish()` then runs automatically and writes the
record + replay through arena/store.py.
"""
from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque

import numpy as np

from arena import store
from arena.board_draw import BoardView
from arena.bots import BotInfo, Board, Sim, make_agent  # bots puts models/ on sys.path
from arena.maps import load_map


def encode(src: int, direction: int, half: bool) -> int:
    """(cell, dir 0..3, half) -> flat action, as core36.mv."""
    return src * 9 + 1 + direction + (4 if half else 0)


def decode(action: int):
    """flat action -> (cell, dir, half), or None for a pass."""
    if action <= 0:
        return None
    cell, k = divmod(action, 9)
    if k == 0:
        return None
    return cell, (k - 1) % 4, k >= 5


class GameSession:
    def __init__(self, bot_id: str, player: str, human_seat: int = 0, seed: int | None = None,
                 max_ticks: int = 600, tick_ms: int = 350, max_think_ms: int = 8000,
                 pair_id: str | None = None, leg: int = 0):
        self.player = (player or "").strip() or "anonymous"
        # every game is half of a pair: same map, seats swapped (removes seat bias)
        self.pair_id = pair_id or store.new_game_id()
        self.leg = int(leg)
        self.human_seat = int(human_seat)
        self.bot_seat = 1 - self.human_seat
        self.max_ticks = int(max_ticks)
        self.tick_ms = int(tick_ms)
        self.max_think_ms = int(max_think_ms)
        self.paused = False

        self.seed, mountain, castle, owner, army, general_idx = load_map(seed)
        self.h = self.w = 23
        self.board = Board(mountain, castle, self.h, self.w)
        self.sim = Sim(self.board, owner, army, general_idx, 0, self.max_ticks)
        self._mountain = np.asarray(mountain, bool)
        self._castle = np.asarray(castle, bool)
        self._castles = [i for i in range(self.board.n) if self.board.castle[i]]
        self._general_idx = [int(g) for g in general_idx]
        # imports happen on the worker thread at the first decide (lazy)
        self.agent = make_agent(bot_id, self.bot_seat, self.board, self.sim, lazy=True)
        self.bot: BotInfo = self.agent.info

        self._queue: deque = deque()            # human moves (src, dir, half)
        self._qlock = threading.Lock()
        self._last_moves: list = []             # (src, dir, half, player) of the last tick

        # replay history: board after every tick (index 0 = start), joint actions
        self._owners = [np.asarray(self.sim.owner, np.int8)]
        self._armies = [np.asarray(self.sim.army, np.int32)]
        self._moves: list = []

        # bot worker state (guarded by _block)
        self._block = threading.Lock()
        self._busy = False            # a worker thread is running
        self._req_tick = -1           # tick the running/last request is for
        self._result: tuple | None = None    # (tick, action) ready to play
        self._req_started = 0.0

        self._last_tick_ms: float | None = None
        self._was_paused = False
        self._waiting = False         # tick held up by the bot
        self.bot_passes = 0           # ticks the bot missed (think budget)

        self.started_at = store.now_iso()
        self._t0 = time.time()
        self.game_id = store.new_game_id()
        self._end: str | None = None
        self._winner = -1             # seat, -1 draw
        self.record: dict | None = None
        self._request_bot()

    # ------------------------------------------------------------------ #
    # bot worker
    # ------------------------------------------------------------------ #
    def _request_bot(self) -> None:
        """Start computing the bot's move for the current tick (if idle)."""
        with self._block:
            if self._busy or self._end is not None:
                return
            tick = self.sim.step
            if self._result is not None and self._result[0] == tick:
                return
            self._busy = True
            self._req_tick = tick
            self._req_started = time.perf_counter()
            snapshot = self.sim.copy()
        threading.Thread(target=self._think, args=(tick, snapshot), daemon=True,
                         name=f"bot-{self.bot.id}").start()

    def _think(self, tick: int, snapshot: Sim) -> None:
        try:
            action = self.agent.decide(snapshot)     # never raises
        except BaseException as exc:                 # belt and braces
            print(f"[arena.session] bot thread: {exc!r}", file=sys.stderr)
            action = 0
        stale = False
        with self._block:
            self._busy = False
            if tick == self.sim.step and self._end is None:
                self._result = (tick, action)
            else:
                stale = True
        if stale:                 # we passed that tick for it; catch up now
            self._request_bot()

    def _bot_move(self):
        """The bot's action for this tick, 0 if it is over budget, None if not ready."""
        with self._block:
            tick = self.sim.step
            if self._result is not None and self._result[0] == tick:
                return self._result[1]
            over = (self._busy and self._req_tick <= tick and
                    1000 * (time.perf_counter() - self._req_started) > self.max_think_ms)
        if over:
            self.bot_passes += 1
            return 0
        self._request_bot()       # in case nothing is running for this tick
        return None

    # ------------------------------------------------------------------ #
    # human input
    # ------------------------------------------------------------------ #
    def queue_move(self, src: int, direction: int, half: bool) -> bool:
        """Append to the human's queue; False if src/dst is off-board or a mountain.
        Ownership is checked only when the move is played."""
        n = self.board.n
        if not (0 <= src < n) or not (0 <= direction < 4) or self.board.mountain[src]:
            return False
        if self.board.step_to[src][direction] < 0:      # off-board or mountain
            return False
        with self._qlock:
            if len(self._queue) >= 256:
                return False
            self._queue.append((int(src), int(direction), bool(half)))
        return True

    def clear_queue(self) -> None:
        with self._qlock:
            self._queue.clear()

    def queued(self) -> list[tuple[int, int, bool]]:
        with self._qlock:
            return list(self._queue)

    def _human_move(self) -> int:
        """Pop queued moves until one is legal now; 0 if the queue runs dry."""
        with self._qlock:
            while self._queue:
                a = encode(*self._queue.popleft())
                if self.sim.legal(self.human_seat, a):
                    return a
        return 0

    # ------------------------------------------------------------------ #
    # clock
    # ------------------------------------------------------------------ #
    def set_speed(self, tick_ms: int) -> None:
        self.tick_ms = max(0, int(tick_ms))

    def update(self, now_ms: float) -> bool:
        """Call every frame. Resolves exactly one tick when due and the bot is
        ready; returns True if a tick happened."""
        if self._end is not None:
            return False
        if self.paused:
            self._was_paused = True
            self._waiting = False
            return False
        if self._last_tick_ms is None or self._was_paused:
            # first frame / after a pause: the human gets a full tick
            self._last_tick_ms = now_ms
            self._was_paused = False
            if self.tick_ms > 0:
                return False
        if now_ms - self._last_tick_ms < self.tick_ms:
            self._waiting = False
            return False
        bot = self._bot_move()
        if bot is None:
            self._waiting = True
            return False
        self._waiting = False
        self._resolve(bot)
        self._last_tick_ms = now_ms
        return True

    def _resolve(self, bot_action: int) -> None:
        sim = self.sim
        human = self._human_move()
        if bot_action and not sim.legal(self.bot_seat, bot_action):
            bot_action = 0
        acts = [0, 0]
        acts[self.human_seat], acts[self.bot_seat] = human, bot_action
        gens_before = list(sim.gen)
        sim.step2(acts[0], acts[1])
        self._last_moves = [(*decode(a), p) for p, a in enumerate(acts) if decode(a)]
        self._owners.append(np.asarray(sim.owner, np.int8))
        self._armies.append(np.asarray(sim.army, np.int32))
        self._moves.append(acts)
        with self._block:
            self._result = None
        if sim.done:
            captured = any(g >= 0 and sim.owner[g] != p for p, g in enumerate(gens_before))
            self._finish_sim("capture" if captured else "timeout")
        else:
            self._request_bot()

    # ------------------------------------------------------------------ #
    # views
    # ------------------------------------------------------------------ #
    def view(self, selection: int = -1) -> BoardView:
        sim = self.sim
        general = np.zeros(self.board.n, bool)
        for p, g in enumerate(sim.gen):
            if g >= 0 and sim.owner[g] == p:
                general[g] = True
        return BoardView(
            h=self.h, w=self.w,
            owner=np.asarray(sim.owner, np.int8), army=np.asarray(sim.army, np.int32),
            mountain=self._mountain, castle=self._castle, general=general,
            last_moves=list(self._last_moves), queued=self.queued(), selection=selection,
        )

    def stats(self) -> dict:
        sim = self.sim
        land, army = sim.totals()
        castles = [sum(1 for c in self._castles if sim.owner[c] == p) for p in (0, 1)]
        return {"tick": sim.step, "land": land, "army": army, "castles": castles,
                "done": self._end is not None, "winner": self._winner if self._end else -1,
                "end": self._end, "bot_thinking": self._waiting}

    # ------------------------------------------------------------------ #
    # ending
    # ------------------------------------------------------------------ #
    def resign(self) -> None:
        self.finish("resign")

    def _finish_sim(self, end: str) -> None:
        self._end = end
        self._winner = self.sim.winner
        self.finish(end)

    def finish(self, reason: str | None = None) -> dict:
        """Idempotent. Ends the game if still running (reason "resign"/"quit",
        both count as a bot win), writes record + replay, returns the record."""
        if self.record is not None:
            return self.record
        if self._end is None:
            self._end = reason if reason in ("resign", "quit") else "quit"
            self._winner = self.bot_seat
        sim = self.sim
        land, army = sim.totals()
        castles = [sum(1 for c in self._castles if sim.owner[c] == p) for p in (0, 1)]
        h, b = self.human_seat, self.bot_seat
        winner = "draw" if self._winner < 0 else ("human" if self._winner == h else "bot")
        rec = {
            "id": self.game_id, "started_at": self.started_at, "ended_at": store.now_iso(),
            "player": self.player, "bot_id": record_bot_id(self.bot), "bot_name": self.bot.name,
            "human_seat": h, "seed": int(self.seed), "max_ticks": self.max_ticks,
            "ticks": int(sim.step), "winner": winner, "end": self._end,
            "land": [land[h], land[b]], "army": [army[h], army[b]],
            "castles": [castles[h], castles[b]],
            "duration_s": round(time.time() - self._t0, 1), "tick_ms": self.tick_ms,
            "pair_id": self.pair_id, "leg": self.leg,
        }
        replay = {
            "owner": np.stack(self._owners).astype(np.int8),          # (T+1, n), row 0 = start
            "army": np.stack(self._armies).astype(np.int32),
            "moves": np.asarray(self._moves, np.int32).reshape(-1, 2),  # (T, 2) actions by seat
            "mountain": self._mountain, "castle": self._castle,
            "general_idx": np.asarray(self._general_idx, np.int32),
            "h": np.int32(self.h), "w": np.int32(self.w),
            "human_seat": np.int32(h), "seed": np.int64(self.seed),
        }
        self.record = rec
        try:
            store.record_game(rec, replay)
        except Exception as exc:         # a full disk must not crash the app
            print(f"[arena.session] could not save game {self.game_id}: {exc!r}", file=sys.stderr)
        return rec


def record_bot_id(bot) -> str:
    """final.py changes over time: record which champion was played,
    e.g. "final@v46_opponent_learner", so the leaderboard keeps them apart."""
    if bot.id != "final":
        return bot.id
    m = re.search(r"\((v\d+[^)]*)\)", bot.name)
    return "final@" + (re.sub(r"\W+", "_", m.group(1)).strip("_") if m else "latest")
