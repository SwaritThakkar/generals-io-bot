#!/usr/bin/env python3
"""Play one game with scripted bots and write a scrubbable HTML replay.

    python demo.py --out replay.html --seed 7 --p0 expander --p1 random
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # project root

import argparse

import jax

from genmod import EnvConfig
from genmod.bots import BOTS
from genmod.render import export_html, record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="replay.html")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--height", type=int, default=23)
    ap.add_argument("--width", type=int, default=23)
    ap.add_argument("--p0", default="expander", choices=sorted(BOTS))
    ap.add_argument("--p1", default="expander", choices=sorted(BOTS))
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--min-spawn", type=int, default=17)
    ap.add_argument("--max-spawn", type=int, default=10_000)
    ap.add_argument("--fragment", action="store_true", help="emit a body fragment, not a full page")
    args = ap.parse_args()

    cfg = EnvConfig(
        height=args.height,
        width=args.width,
        max_steps=args.max_steps,
        min_spawn_dist=args.min_spawn,
        max_spawn_dist=args.max_spawn,
        bfs_iters=2 * (args.height + args.width),
    )
    rep = record(cfg, jax.random.PRNGKey(args.seed), [BOTS[args.p0], BOTS[args.p1]])

    result = "draw" if rep.winner < 0 else f"player {rep.winner} wins"
    print(f"{args.p0} vs {args.p1}: {result} after {rep.turns - 1} turns "
          f"(land {rep.land[-1].tolist()}, army {rep.armies[-1].tolist()})")

    path = export_html(
        rep,
        args.out,
        subtitle=f"{args.p0} vs {args.p1} · seed {args.seed} · {cfg.height}×{cfg.width} · "
                 f"spawns ≥ {cfg.min_spawn_dist} apart",
        player_names=[f"Red · {args.p0}", f"Blue · {args.p1}"],
        standalone=not args.fragment,
    )
    print("wrote", path.resolve())


if __name__ == "__main__":
    main()
