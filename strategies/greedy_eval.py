"""Bot 1: evaluation + neighbours + pick the best. Hill climbing (AIMA 4.1.1).

The whole idea in one line:

    play the move whose resulting board scores highest

Three pieces. Every future improvement is a change to exactly one of them:

    neighbours(...)   which moves to consider        -> search
    simulate(...)     the board after a move         -> the rules, already written
    evaluate(...)     how good that board is         -> your judgement

Run it:
    python viewer.py --p0 strategies/greedy_eval.py --p1 expander --tick 80
    python server/local_server.py --p0 strategies/greedy_eval.py --p1 expander \
        --games 50 --headless
"""

from __future__ import annotations

import pathlib
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import genmod  # noqa: E402
from genmod.state import army_counts, land_counts  # noqa: E402

# --------------------------------------------------------------------------- #
# Weights, DERIVED from the rules rather than guessed.
# See strategies/RESEARCH-DIRECTIONS.md section 1.
#
#   a producer (general/castle) makes  1 army / 2 ticks  = 0.50 army/tick
#   a land tile makes                  1 army / 50 ticks = 0.02 army/tick
#   => one producer is worth 25 land tiles of production
#
#   a land tile also keeps paying: over N remaining ticks it yields 0.02*N army,
#   so mid-game a tile is worth several army. Hence army is weighted low --
#   army in the bank is worth less than the land that keeps printing it.
# --------------------------------------------------------------------------- #
W_LAND = 1.0
W_ARMY = 0.25
W_PRODUCER = 25.0 * W_LAND

# Set this to 0.0 to get the undefended bot back (it lost 80% of games).
W_DANGER = 1e6

WIN = 1e9


def evaluate(st, player: int, num_players: int = 2):
    """Score a board from `player`'s point of view. Higher is better.

    START HERE when you want the bot to play better. Everything else is plumbing.

    Written in jnp (not numpy) so it can be compiled and run on many candidate
    boards at once -- see `_best_action` below.
    """
    me, opp = player, 1 - player

    land = land_counts(st, num_players)
    army = army_counts(st, num_players)

    producers = st.castle | st.general
    prod_me = ((st.owner == me) & producers).sum()
    prod_opp = ((st.owner == opp) & producers).sum()

    score = (
        W_LAND * (land[me] - land[opp])
        + W_ARMY * (army[me] - army[opp])
        + W_PRODUCER * (prod_me - prod_opp)
        - W_DANGER * _danger(st, me, opp)
    )

    # Winning and losing are not opinions. Score them past everything else.
    score = jnp.where(~st.alive[opp], WIN, score)
    score = jnp.where(~st.alive[me], -WIN, score)
    return score


def _danger(st, me: int, opp: int):
    """Can the opponent capture my general on their very next move?

    This is minimax (AIMA 5.2) with the opponent's move list cut down to one
    thing: moves that take my general. Considering all ~400 of their replies is
    unaffordable; considering the four that actually end the game is nearly free.

    An enemy stack sitting next to my general sends `army - 1` units (one always
    stays behind), and captures if that beats my garrison.

    An earlier version of this scored threats by Manhattan distance from anywhere
    on the board. It made the bot WORSE (20% -> 15%), because that number barely
    changes between my candidate moves -- and a term that is the same for every
    neighbour cannot change an argmax. Only the four adjacent cells are things my
    move can actually do something about.
    """
    h, w = st.owner.shape
    g = st.general_idx[me]
    gr, gc = g // w, g % w

    adjacent = (jnp.abs(jnp.arange(h) - gr)[:, None]
                + jnp.abs(jnp.arange(w) - gc)[None, :]) == 1

    garrison = st.army.reshape(-1)[g]
    can_take = adjacent & (st.owner == opp) & (st.army - 1 > garrison)

    lost = can_take.any() & (g >= 0)
    return jnp.where(lost, 1.0, 0.0)


def neighbours(state, player: int) -> list[int]:
    """The moves worth considering. Right now: all of them.

    The other place to get cleverer. Ordering these by a cheap guess, or dropping
    obviously pointless ones, is what buys you the budget to search deeper.
    """
    return state.legal_actions(player)


def simulate(st, action, player: int, cfg):
    """The board after `player` plays `action` and the opponent does nothing.

    "The opponent does nothing" is a lie, and it is the first lie worth fixing.
    Replace it with "the opponent plays their best reply" and this bot becomes
    minimax (AIMA 5.2).
    """
    acts = jnp.zeros((2,), jnp.int32).at[player].set(action)
    return genmod.step(st, acts, cfg)[0]


# --------------------------------------------------------------------------- #
# Making it fast.
#
# The obvious loop -- simulate each move, score it, keep the best -- calls the
# engine ~400 times per tick and is far too slow in Python. But every candidate
# move is independent, so they can all be simulated in ONE batched call with
# jax.vmap. Same algorithm, ~100x less overhead.
#
# Candidate lists are padded to fixed bucket sizes so JAX compiles a handful of
# shapes instead of a new one every tick.
# --------------------------------------------------------------------------- #
BUCKETS = (32, 64, 128, 256, 512, 1024)


@partial(jax.jit, static_argnames=("cfg", "player", "size"))
def _score_all(st, actions, player: int, cfg, size: int):
    """Simulate `size` candidate moves at once and score every resulting board."""
    boards = jax.vmap(lambda a: simulate(st, a, player, cfg))(actions)
    return jax.vmap(lambda b: evaluate(b, player, cfg.num_players))(boards)


def act(state, player: int) -> int:
    moves = neighbours(state, player)
    if not moves:
        return 0  # pass

    cfg = state.env_config()
    st = state.to_genmod_state(cfg)

    size = next((b for b in BUCKETS if b >= len(moves)), len(moves))
    padded = np.zeros((size,), np.int32)
    padded[: len(moves)] = moves

    scores = np.asarray(_score_all(st, jnp.asarray(padded), player, cfg, size))
    return int(moves[int(np.argmax(scores[: len(moves)]))])
