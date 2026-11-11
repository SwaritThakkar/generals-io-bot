"""Template for your own agent.

An agent is a plain function ``act(state, player) -> action``. It works in three
places without changing a line:

    # 1. watch it play locally, on the JAX engine
    python server/local_server.py --p0 server/agents/example_agent.py:act --p1 expander

    # 2. play against it yourself
    python server/local_server.py --p0 human --p1 server/agents/example_agent.py:act --open

    # 3. serve it to the web app
    python server/agents/example_agent.py --port 8801

`state` is a `BotState`: plain numpy arrays, nothing hidden, since this is the
perfect-information variant. Return a flat action index, a `(row, col, dir,
half)` tuple, or a list of those to queue a path.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from server.bot_protocol import BotState, encode, serve  # noqa: E402


def act(state: BotState, player: int):
    """Grab the nearest thing we can afford; otherwise push the biggest stack out.

    Deliberately simple -- it exists to show the shape of an agent, not to be
    strong. `genmod.bots.expander_bot` is the stronger scripted baseline.
    """
    h, w = state.h, state.w
    owner, army = state.owner, state.army

    mine = (owner == player) & (army >= 2)
    if not mine.any():
        return 0  # pass

    best = None
    best_score = -1
    for r, c in zip(*np.nonzero(mine)):
        surplus = int(army[r, c]) - 1
        for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w) or state.mountain[rr, cc]:
                continue
            target_owner = int(owner[rr, cc])
            defence = int(army[rr, cc]) if target_owner != player else -1
            if target_owner == player:
                score = surplus            # consolidating is the fallback
            elif surplus > defence:
                score = 10_000 + surplus   # a capture we can actually afford
            else:
                continue
            if score > best_score:
                best_score = score
                best = (int(r), int(c), d)

    if best is None:
        return 0
    r, c, d = best
    return encode(r, c, d, half=False, width=w)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--name", default="Example agent")
    serve(act, **vars(ap.parse_args()))
