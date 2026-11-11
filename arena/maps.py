"""Pre-generated 23x23 maps, so starting an arena game never imports JAX.

Maps come from `genmod.reset(PRNGKey(seed), cfg)` with the same config as the
sim36 harness (`models/tools36/arena36.start`): 23x23, bfs_iters=92, every
other field at its `EnvConfig` default. Build the cache once:

    python -m arena.maps --build            # 200 maps -> arena/maps_cache.npz

    seed, mountain, castle, owner, army, general_idx = load_map()      # random
    seed, mountain, castle, owner, army, general_idx = load_map(1234)  # exact

Arrays are flat (length 23*23); `general_idx` is a list of two cell indices.
An uncached seed is generated on the fly (JAX import, a few seconds) and
appended to the cache. A random pick only uses maps that pass
arena.playable.check (generals connected without crossing a castle, room to
expand); `python -m arena.maps` lists the rejected seeds.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import threading

import numpy as np

from arena.playable import check

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path(__file__).resolve().with_name("maps_cache.npz")
H = W = 23
BUILD_SEEDS = range(1, 201)

_lock = threading.Lock()
_maps: dict | None = None        # seed -> (mountain, castle, owner, army, general_idx)


# --------------------------------------------------------------------------- #
# generation (JAX) and the npz cache
# --------------------------------------------------------------------------- #
def generate(seeds) -> dict:
    """Run genmod.reset for each seed. Imports JAX (slow); only for building."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import jax
    from genmod import EnvConfig, reset

    cfg = EnvConfig(height=H, width=W, bfs_iters=92)
    out = {}
    for seed in seeds:
        st = reset(jax.random.PRNGKey(int(seed)), cfg)
        out[int(seed)] = (np.asarray(st.mountain, bool).reshape(-1),
                          np.asarray(st.castle, bool).reshape(-1),
                          np.asarray(st.owner, np.int8).reshape(-1),
                          np.asarray(st.army, np.int32).reshape(-1),
                          [int(g) for g in np.asarray(st.general_idx)])
    return out


def _save(maps: dict) -> None:
    seeds = sorted(maps)
    np.savez_compressed(
        CACHE,
        seeds=np.asarray(seeds, np.int64),
        mountain=np.stack([maps[s][0] for s in seeds]),
        castle=np.stack([maps[s][1] for s in seeds]),
        owner=np.stack([maps[s][2] for s in seeds]),
        army=np.stack([maps[s][3] for s in seeds]),
        general_idx=np.asarray([maps[s][4] for s in seeds], np.int32),
    )


def _load() -> dict:
    """Read the cache into memory once (about 1 MB)."""
    global _maps
    if _maps is None:
        maps = {}
        if CACHE.exists():
            z = np.load(CACHE)
            for i, s in enumerate(z["seeds"]):
                maps[int(s)] = (z["mountain"][i].astype(bool), z["castle"][i].astype(bool),
                                z["owner"][i].astype(np.int8), z["army"][i].astype(np.int32),
                                [int(g) for g in z["general_idx"][i]])
        _maps = maps
    return _maps


def cached_seeds() -> list[int]:
    with _lock:
        return sorted(_load())


_ok: dict = {}                   # seed -> (playable, reason), computed once per seed


def _is_playable(seed: int, maps: dict) -> bool:
    if seed not in _ok:
        m, c, _o, _a, gi = maps[seed]
        _ok[seed] = check(m, c, gi, H, W)
    return _ok[seed][0]


def playable_seeds() -> list[int]:
    with _lock:
        maps = _load()
        return [s for s in sorted(maps) if _is_playable(s, maps)]


def random_seed() -> int:
    """A random cached map that is playable (the arena never starts on a walled-in general)."""
    seeds = playable_seeds()
    if not seeds:
        raise RuntimeError("no playable maps in the cache; run python -m arena.maps --build")
    return random.choice(seeds)


def load_map(seed: int | None = None):
    """-> (seed, mountain, castle, owner, army, general_idx); seed=None = random cached map."""
    with _lock:
        maps = _load()
        if seed is None:
            if not maps:                      # no cache at all: build a small one
                maps.update(generate(range(1, 21)))
                _save(maps)
            seed = random.choice([x for x in sorted(maps) if _is_playable(x, maps)])
        seed = int(seed)
        if seed not in maps:
            maps.update(generate([seed]))
            _save(maps)
        m, c, o, a, gi = maps[seed]
        # copies, so a caller can never mutate the cache
        return seed, m.copy(), c.copy(), o.copy(), a.copy(), list(gi)


def build(seeds=BUILD_SEEDS) -> None:
    global _maps
    maps = generate(seeds)
    with _lock:
        _maps = maps
        _save(maps)
    print(f"wrote {len(maps)} maps to {CACHE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="(re)generate the cache")
    ap.add_argument("--n", type=int, default=len(BUILD_SEEDS), help="number of maps (seeds 1..n)")
    a = ap.parse_args()
    if a.build:
        build(range(1, a.n + 1))
    else:
        maps = _load()
        bad = [(x, _ok[x][1]) for x in sorted(maps) if not _is_playable(x, maps)]
        print(f"{len(maps)} cached maps in {CACHE}; {len(maps) - len(bad)} playable")
        for x, why in bad:
            print(f"  seed {x}: {why}")
