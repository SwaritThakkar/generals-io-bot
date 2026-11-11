"""Static configuration for the modified (perfect-information) Generals.io environment.

Everything in `EnvConfig` is *static*: it is baked in at trace time, so branches on
these fields cost nothing at runtime and unused machinery is eliminated by XLA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Direction order used everywhere: 0=up, 1=down, 2=left, 3=right.
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_NAMES = ("up", "down", "left", "right")

# Per-cell action layout (9 actions per cell):
#   0            -> pass
#   1 + d        -> move ALL BUT ONE unit in direction d
#   5 + d        -> move HALF (floor) the units in direction d
NUM_CELL_ACTIONS = 9
ACTION_PASS = 0


@dataclass(frozen=True)
class EnvConfig:
    # ---- board -------------------------------------------------------------
    height: int = 23
    width: int = 23
    num_players: int = 2

    #: If True, each episode samples an active sub-rectangle; the rest of the
    #: buffer is filled with mountains and flagged unplayable (fixed shapes).
    random_dims: bool = False
    min_height: int = 15
    min_width: int = 15

    # ---- map generation ----------------------------------------------------
    mountain_density: float = 0.20
    min_castles: int = 9
    max_castles: int = 11
    castle_army_min: int = 40
    castle_army_max: int = 50

    #: Generals are placed at BFS distance in [min_spawn_dist, max_spawn_dist].
    #: Lowering `max_spawn_dist` implements the paper's spawn-distance curriculum.
    min_spawn_dist: int = 17
    max_spawn_dist: int = 10_000
    #: Every player is guaranteed a neutral castle within this BFS radius.
    guaranteed_castle_radius: int = 6
    #: Number of min-plus relaxation rounds used for grid BFS. Must exceed the
    #: longest shortest-path you care about; 2*(H+W) is ample for 20% mountains.
    bfs_iters: int = 96

    # ---- economy -----------------------------------------------------------
    #: Generals and owned castles gain +1 every this many steps.
    general_growth_interval: int = 2
    #: Every owned cell gains +1 every this many steps.
    land_growth_interval: int = 50

    # ---- episode -----------------------------------------------------------
    max_steps: int = 2000
    #: What happens when `max_steps` is hit without a capture.
    timeout_result: Literal["draw", "land"] = "land"

    # ---- observation -------------------------------------------------------
    #: "full"      -> the MODIFIED game: no fog, both players see everything.
    #: "opponent"  -> opponent army/territory/general always visible, but
    #:                unexplored neutral terrain stays fogged.
    #: "fog"       -> the ORIGINAL game (own cells + Moore neighbourhood).
    visibility: Literal["full", "opponent", "fog"] = "full"
    #: Number of stacked per-cell army deltas (0 disables the history channels).
    history_length: int = 7
    #: Sliding window of (land, army) scoreboard totals kept per player, newest
    #: first, exposed as the paper's two non-spatial tokens. 0 disables it.
    #: Largely redundant under "full" visibility, so it defaults to off.
    series_length: int = 0
    #: Divisor applied after log1p to army-count channels.
    army_log_scale: float = 6.0

    # ---- derived -----------------------------------------------------------
    @property
    def num_cells(self) -> int:
        return self.height * self.width

    @property
    def num_actions(self) -> int:
        return self.height * self.width * NUM_CELL_ACTIONS

    def __post_init__(self):
        if self.num_players < 2:
            raise ValueError("num_players must be >= 2")
        if self.max_castles < self.num_players:
            raise ValueError("max_castles must be >= num_players")
        if self.random_dims and (self.min_height > self.height or self.min_width > self.width):
            raise ValueError("min_height/min_width must not exceed height/width")
