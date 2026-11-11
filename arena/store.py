"""Results store for the arena app: one JSON line per finished game, plus a
compressed replay per game.

    arena_results/
        games.jsonl                  one record per game (schema below)
        replays/<game_id>.npz        owner/army per tick, moves, map

Record schema (all keys always present):
    id            str   "20261103-201455-ab12"
    started_at    str   ISO time
    ended_at      str   ISO time
    player        str   participant's name as typed (trimmed)
    bot_id        str   file stem in models/, e.g. "v46_opponent_learner", or a builtin
    bot_name      str   display name
    human_seat    int   0 or 1
    seed          int   map seed
    max_ticks     int
    ticks         int   ticks played
    winner        str   "human" | "bot" | "draw"
    end           str   "capture" | "timeout" | "resign" | "quit"
    land          [human, bot]
    army          [human, bot]
    castles       [human, bot]
    duration_s    float wall-clock seconds
    tick_ms       int   game speed used
    pair_id       str   both games of a pair share it (same map, seats swapped)
    leg           int   0 = first game of the pair, 1 = the same map with seats swapped
"""
from __future__ import annotations

import json
import math
import pathlib
import secrets
import time
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "arena_results"
GAMES = RESULTS / "games.jsonl"
REPLAYS = RESULTS / "replays"

KEYS = ("id", "started_at", "ended_at", "player", "bot_id", "bot_name", "human_seat", "seed",
        "max_ticks", "ticks", "winner", "end", "land", "army", "castles", "duration_s", "tick_ms",
        "pair_id", "leg")


def new_game_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def record_game(rec: dict, replay: dict | None = None) -> dict:
    """Append one finished game. `replay` (optional) is saved with np.savez_compressed:
    keys like owner (T, n) int8, army (T, n) int32, moves (T, 2) int32, mountain, castle."""
    missing = [k for k in KEYS if k not in rec]
    if missing:
        raise ValueError(f"game record missing {missing}")
    RESULTS.mkdir(exist_ok=True)
    with open(GAMES, "a") as f:
        f.write(json.dumps({k: rec[k] for k in KEYS}) + "\n")
    if replay is not None:
        REPLAYS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(REPLAYS / f"{rec['id']}.npz", **replay)
    return rec


def load_games() -> list[dict]:
    if not GAMES.exists():
        return []
    out = []
    for line in GAMES.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def counted(g: dict) -> bool:
    """Games that count for the leaderboard: finished ones, not quits before tick 25."""
    return not (g["end"] == "quit" and g["ticks"] < 25)


# --------------------------------------------------------------------------- #
# leaderboards
# --------------------------------------------------------------------------- #
def wilson_low(wins: float, n: int, z: float = 1.96) -> float:
    """Lower bound of the 95% Wilson interval: ranks 3/3 below 40/50."""
    if n == 0:
        return 0.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d


def elo(games: list[dict], k: float = 24.0, base: float = 1200.0) -> dict:
    """Elo over humans and bots in one pool, games in time order.
    Keys: "bot:<id>" and "player:<lowercased name>"."""
    rating = defaultdict(lambda: base)
    for g in sorted(games, key=lambda g: g["started_at"]):
        if not counted(g):
            continue
        a, b = "player:" + g["player"].strip().lower(), "bot:" + g["bot_id"]
        ea = 1 / (1 + 10 ** ((rating[b] - rating[a]) / 400))
        sa = 1.0 if g["winner"] == "human" else 0.0 if g["winner"] == "bot" else 0.5
        rating[a] += k * (sa - ea)
        rating[b] += k * ((1 - sa) - (1 - ea))
    return dict(rating)


def bot_board(games: list[dict] | None = None) -> list[dict]:
    """One row per bot that has played, best first (Wilson bound on bot wins)."""
    games = load_games() if games is None else games
    ratings = elo(games)
    rows = defaultdict(lambda: {"games": 0, "bot_wins": 0, "human_wins": 0, "draws": 0,
                                "ticks": 0, "players": set(), "last": ""})
    names = {}
    for g in games:
        if not counted(g):
            continue
        r = rows[g["bot_id"]]
        names[g["bot_id"]] = g["bot_name"]
        r["games"] += 1
        r["bot_wins"] += g["winner"] == "bot"
        r["human_wins"] += g["winner"] == "human"
        r["draws"] += g["winner"] == "draw"
        r["ticks"] += g["ticks"]
        r["players"].add(g["player"].strip().lower())
        r["last"] = max(r["last"], g["started_at"])
    out = []
    for bot_id, r in rows.items():
        out.append({
            "bot_id": bot_id, "bot_name": names[bot_id], "games": r["games"],
            "bot_wins": r["bot_wins"], "human_wins": r["human_wins"], "draws": r["draws"],
            "win_rate": r["bot_wins"] / r["games"],
            "score": wilson_low(r["bot_wins"] + 0.5 * r["draws"], r["games"]),
            "elo": round(ratings.get("bot:" + bot_id, 1200.0)),
            "avg_ticks": round(r["ticks"] / r["games"]),
            "players": len(r["players"]), "last": r["last"],
        })
    out.sort(key=lambda r: (-r["score"], -r["games"]))
    return out


def player_board(games: list[dict] | None = None) -> list[dict]:
    """One row per participant, best first (Elo)."""
    games = load_games() if games is None else games
    ratings = elo(games)
    rows = defaultdict(lambda: {"games": 0, "wins": 0, "draws": 0, "beaten": set(), "name": ""})
    for g in games:
        if not counted(g):
            continue
        key = g["player"].strip().lower()
        r = rows[key]
        r["name"] = g["player"].strip()
        r["games"] += 1
        r["wins"] += g["winner"] == "human"
        r["draws"] += g["winner"] == "draw"
        if g["winner"] == "human":
            r["beaten"].add(g["bot_name"])
    out = [{"player": r["name"], "games": r["games"], "wins": r["wins"], "draws": r["draws"],
            "losses": r["games"] - r["wins"] - r["draws"], "bots_beaten": sorted(r["beaten"]),
            "elo": round(ratings.get("player:" + key, 1200.0))} for key, r in rows.items()]
    out.sort(key=lambda r: (-r["elo"], -r["wins"]))
    return out


def bot_summary(bot_id: str, games: list[dict] | None = None) -> dict | None:
    for row in bot_board(games):
        if row["bot_id"] == bot_id:
            return row
    return None
