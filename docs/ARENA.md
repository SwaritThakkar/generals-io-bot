# Arena: friends vs our bots

A local pygame app (no web server) for collecting human-vs-bot results.

```
python play_arena.py          # opens the 1280x800 window (use the miniforge python that has pygame)
```

1. Type the player's name (recent names are one click away).
2. Pick a bot from the menu, or from the leaderboard (Leaderboard → select → Play this bot).
3. Choose the side for game 1 (red moves first / blue), length (300 / 600 / 1000 ticks) and speed, then Enter.
   Every match is a **pair**: after game 1 the result screen's Enter plays game 2 on the same map with
   the seats swapped, so neither side keeps a position advantage. After game 2 it shows the pair score.
4. New players: show **How the game works** (F1) and **Controls** (F2) first.

In game: click one of your tiles, arrows / WASD queue moves, shift + arrow sends half,
c clears the queue, p pauses, [ ] change speed, esc leaves (counts as a quit).

## Where results go

```
arena_results/games.jsonl     one line per game (player, bot, seat, winner, end, land/army/castles, ticks, ...)
arena_results/replays/*.npz   full replay of each game (owner/army per tick, moves, map)
```

Each record carries `pair_id` and `leg` (0 = game 1, 1 = seats swapped). Maps come only from
seeds that pass `arena/playable.py` (generals connected without crossing a castle, at least 60
free tiles each); `python -m arena.maps` lists the rejected seeds (9, 83, 107 of the 200 cached).
`viewer.py` uses the same check and the same pairing (r / n = next game).

Leaderboards are computed from `games.jsonl`: bots ranked by the 95% Wilson lower
bound of their win rate (so 3/3 does not beat 40/50), and one Elo pool for players and bots.
Quits before tick 25 are not counted. Games against `final.py` are recorded with the champion's
version (e.g. `final@v46_opponent_learner`), so changing final.py later keeps results apart.

## Pieces

| File | What it does |
|---|---|
| `play_arena.py`, `arena/app.py` | the window: Home, Game, Result, Leaderboard, Tutorial screens |
| `arena/session.py` | one game on the fast simulator (`models/sim36.py`); the bot thinks on a background thread |
| `arena/bots.py`, `arena/bot_index.json` | every bot in `models/` + final.py + builtins; `python -m arena.bots --bench` re-times them |
| `arena/maps.py`, `arena/maps_cache.npz` | 200 pre-generated maps, so a game starts without JAX |
| `arena/board_draw.py` | the board renderer (same look as `viewer.py`), used by the game and the tutorial |
| `arena/tutorial.py`, `arena/media/*.mp4` | animated rules (13 scenes) and controls (6 scenes); `python -m arena.tutorial --export arena/media` re-renders |
| `arena/store.py` | results file, Wilson ranking, Elo |
| `arena/selftest.py` | `python -m arena.selftest`: a full headless game against every fast/ok bot |

Arena rule differences from the full JAX engine: capturing the general ends the game at once
(the loser's land is not transferred, which does not change the winner); an illegal queued move is
skipped and the next one plays. The slow v14–v34 search bots (0.2–0.7 s per move) make a tick wait for them.
