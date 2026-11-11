"""Serve one of the built-in scripted bots over the `genmod-bot/1` protocol.

    python server/agents/builtin_service.py --bot expander --port 8801

Handy as a known-good opponent while you wire up the web app, and as the thing
to point `GENMOD_BOTS` at during development.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from genmod.bots import BOTS                             # noqa: E402
from server.bot_protocol import load_agent, serve        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="expander", choices=sorted(BOTS))
    ap.add_argument("--port", type=int, default=8801)
    args = ap.parse_args()
    serve(load_agent(args.bot), port=args.port, name=f"{args.bot} (built-in)")


if __name__ == "__main__":
    main()
