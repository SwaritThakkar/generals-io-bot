"""A fast JAX simulator for *perfect-information* Generals.io.

The rules are those of Straka, Lisy & Schmid (arXiv:2606.23348) Section IV, with
one modification: the fog of war is removed, so each player observes the entire
board.  The original fogged observation is retained behind
`EnvConfig.visibility="fog"` so both regimes share identical dynamics.
"""

from .config import ACTION_PASS, DIR_NAMES, DIRS, NUM_CELL_ACTIONS, EnvConfig
from .env import (
    decode_action,
    encode_action,
    legal_action_mask,
    reset,
    step,
    step_auto_reset,
    visibility,
)
from .obs import channel_names, num_channels, observe, scoreboard_series
from .rollout import (
    bot_policy,
    make_observe_batch,
    make_rollout,
    make_step_batch,
    random_policy,
    reset_batch,
)
from .state import State, army_counts, land_counts

__all__ = [
    "EnvConfig", "State", "DIRS", "DIR_NAMES", "NUM_CELL_ACTIONS", "ACTION_PASS",
    "reset", "step", "step_auto_reset", "observe", "scoreboard_series",
    "legal_action_mask", "visibility", "encode_action", "decode_action",
    "channel_names", "num_channels", "land_counts", "army_counts",
    "reset_batch", "make_step_batch", "make_observe_batch", "make_rollout",
    "bot_policy", "random_policy",
]
