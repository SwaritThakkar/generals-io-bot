"""Registry of every playable bot, and a uniform `Agent` to drive one on a Sim.

Three kinds of bot:

    planner   models/vNN_*.py (v37+) and final.py expose `Planner(ctx, p)`
              with `.act(sim) -> action`; driven directly on the sim36 Sim.
    act       older bots expose `act(state, player)` (the genmod-bot/1
              protocol); they get a `server.bot_protocol.BotState` built from
              the Sim. They may return a list of actions, which is queued and
              replayed (as in server/local_server.py) until a step is illegal.
    builtin   cheap scripts from core36/core37 ("expander", "rusher", ...)
              and a uniformly random legal mover.

    for info in list_bots():                 # final first, newest first
        print(info.id, info.name, info.tier)
    agent = make_agent("v46_opponent_learner", seat=1, board=board, sim=sim)
    action = agent.decide(sim)               # never raises; 0 = pass

Each bot file is executed privately (`importlib`, unique module name) so two
bots never share module globals. models/ is put on sys.path because the bots
import core36, sim36, ... by name.

Speed tiers come from the bench (arena/bot_index.json):

    python -m arena.bots --bench             # ~40 ticks vs expander, 2 maps
    python -m arena.bots                     # print the registry
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import itertools
import json
import multiprocessing as mp
import pathlib
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
INDEX = pathlib.Path(__file__).resolve().with_name("bot_index.json")
for _p in (str(MODELS), str(ROOT)):      # models first: bots import core36 etc.
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sim36 import Board, Sim  # noqa: E402

FAST_MS, OK_MS = 30.0, 150.0
#: shown for builtins; the bench measures them too
BUILTINS = {
    "expander": ("Expander", "Scripted expansion: takes the cheapest neutral land, never attacks on purpose."),
    "rusher": ("Rusher", "Banks at its general, then goes all-in on yours the moment the walk would win."),
    "castle_rusher": ("Castle rusher", "Banks, takes the nearest castle when affordable, then rushes your general."),
    "econ": ("Econ (v36 script)", "v36's scripted economy: castles, expansion and a home-safety rule."),
    "random": ("Random", "Plays a uniformly random legal move every tick."),
}


@dataclass(frozen=True)
class BotInfo:
    id: str            # file stem, e.g. "v46_opponent_learner"; "final"; builtin names
    name: str          # short display name, e.g. "v46 · Opponent learner"
    version: int | None
    summary: str       # one plain sentence from the module docstring
    kind: str          # "planner" | "act" | "builtin"
    path: str | None
    tier: str          # "fast" | "ok" | "slow" | "broken" | "unknown"
    ms_mean: float | None


# --------------------------------------------------------------------------- #
# discovery (source text only: nothing is imported to build the list)
# --------------------------------------------------------------------------- #
_PLANNER_RE = re.compile(r"^(class Planner\b|Planner\s*=)", re.M)
_ACT_RE = re.compile(r"^def act\(", re.M)
_VER_RE = re.compile(r"^v(\d+)_(.+)$")
_GENERIC_DOC = "Search bot for the repo's perfect-information"


def _readme_rows() -> dict:
    """`models/README.md` table: file -> approach text (better than the
    boilerplate docstring the v1-v35 files share)."""
    rows = {}
    try:
        for line in (MODELS / "README.md").read_text().splitlines():
            m = re.match(r"^\|\s*`([\w]+)\.py`\s*\|\s*(.+?)\s*\|\s*$", line)
            if m:
                rows.setdefault(m.group(1), m.group(2))
    except OSError:
        pass
    return rows


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    text = re.sub(r"^(Current|Baseline|Initial|Experimental) (candidate|submission):\s*", "", text)
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    s = (m.group(1) if m else text).strip()
    if re.match(r"v\d", s):          # keep "v44 + ..." lower case
        return s
    return s[:1].upper() + s[1:]


def _summary(stem: str, src: str, readme: dict) -> str:
    try:
        doc = ast.get_docstring(ast.parse(src)) or ""
    except SyntaxError:
        doc = ""
    if doc and not doc.startswith(_GENERIC_DOC):
        first = doc.split("\n\n")[0]
        first = re.sub(r"^v\d+:\s*", "", first.strip())          # "v46: v44 + ..." -> "v44 + ..."
        if stem == "final":
            m = re.match(r"Final bot:\s*(v\d+[\w ]*?)\s*\(", first)
            return f"The current champion: {m.group(1) if m else 'the latest bot'}."
        return _first_sentence(first)
    if stem in readme:
        return _first_sentence(readme[stem])
    return "Weighted A* route search bot."


def _display_name(stem: str) -> tuple[str, int | None]:
    m = _VER_RE.match(stem)
    if m:
        words = m.group(2).replace("_", " ")
        words = re.sub(r"\bwastar\b", "WA*", words)
        words = re.sub(r"\bastar(\d*)\b", r"A*\1", words)
        words = re.sub(r"\bml\b", "ML", words)
        words = re.sub(r"\bprobcut\b", "ProbCut", words)
        return f"v{m.group(1)} · {words[:1].upper() + words[1:]}", int(m.group(1))
    words = stem.replace("_", " ")
    return words[:1].upper() + words[1:], None


def _bench_index() -> dict:
    try:
        return json.loads(INDEX.read_text()).get("bots", {})
    except (OSError, ValueError):
        return {}


_CACHE: dict = {}


def _discover() -> list[BotInfo]:
    """Cached on the mtimes of models/, final.py and the bench index, so the
    UI can call list_bots() every frame-ish without re-parsing 60 files."""
    stamp = []
    for p in (MODELS, ROOT / "final.py", INDEX, MODELS / "README.md"):
        try:
            stamp.append(p.stat().st_mtime_ns)
        except OSError:
            stamp.append(0)
    stamp = tuple(stamp)
    if _CACHE.get("stamp") != stamp:
        _CACHE["bots"] = _scan()
        _CACHE["stamp"] = stamp
    return _CACHE["bots"]


def _scan() -> list[BotInfo]:
    bench = _bench_index()
    readme = _readme_rows()

    def tier(bot_id):
        row = bench.get(bot_id)
        return (row["tier"], row.get("ms_mean")) if row else ("unknown", None)

    out = []
    files = [(ROOT / "final.py")] + sorted(MODELS.glob("*.py"))
    for path in files:
        try:
            src = path.read_text()
        except OSError:
            continue
        if _PLANNER_RE.search(src):
            kind = "planner"
        elif _ACT_RE.search(src):
            kind = "act"
        else:
            continue
        stem = path.stem
        if stem == "final":
            # name the champion's version, so results stay separate when final.py changes
            m = re.search(r"Final bot:\s*(v\d+[\w ]*?)\s*\(", src)
            champ = m.group(1).strip() if m else "latest"
            bot_id, name, version = "final", f"Final \u00b7 champion ({champ})", None
        else:
            bot_id = stem
            name, version = _display_name(stem)
        t, ms = tier(bot_id)
        out.append(BotInfo(bot_id, name, version, _summary(stem, src, readme), kind, str(path), t, ms))
    for bot_id, (name, summary) in BUILTINS.items():
        t, ms = tier(bot_id)
        out.append(BotInfo(bot_id, name, None, summary, "builtin", None, t, ms))
    return out


def _sort_key(b: BotInfo):
    group = 0 if b.id == "final" else (2 if b.kind == "builtin" else 1)
    return (group, -(b.version if b.version is not None else -1), b.id)


def list_bots(include_broken: bool = False) -> list[BotInfo]:
    """Sorted: final first, then version descending (unversioned scripts after
    v1), builtins last. Re-reads models/ and the bench index on every call."""
    bots = sorted(_discover(), key=_sort_key)
    if not include_broken:
        bots = [b for b in bots if b.tier != "broken"]
    return bots


def get_bot(bot_id: str) -> BotInfo:
    for b in _discover():
        if b.id == bot_id:
            return b
    raise KeyError(f"unknown bot {bot_id!r}")


# --------------------------------------------------------------------------- #
# the protocol state for act(state, player) bots
# --------------------------------------------------------------------------- #
def bot_state(sim: Sim, min_spawn_dist: int = 17):
    """`server.bot_protocol.BotState` for the Sim's current tick: (h, w) grids
    plus counts and the config dict, so `to_genmod_state` / `env_config` work
    for the JAX-based bots and `Sim.from_state` for the sim36 ones."""
    from server.bot_protocol import BotState

    b = sim.b
    h, w = b.h, b.w
    owner = np.asarray(sim.owner, np.int8)
    army = np.asarray(sim.army, np.int32)
    general = np.zeros(b.n, bool)
    for p, g in enumerate(sim.gen):
        if g >= 0 and sim.owner[g] == p:
            general[g] = True
    land, tot = sim.totals()
    return BotState(
        h=h, w=w, n=2, tick=int(sim.step),
        owner=owner.reshape(h, w), army=army.reshape(h, w),
        mountain=np.asarray(b.mountain, bool).reshape(h, w),
        castle=np.asarray(b.castle, bool).reshape(h, w),
        general=general.reshape(h, w),
        general_idx=np.asarray(sim.gen, np.int32),
        alive=np.asarray([g >= 0 for g in sim.gen], bool),
        land=np.asarray(land, np.int32), army_total=np.asarray(tot, np.int32),
        config={"generalGrowthInterval": sim.gil, "landGrowthInterval": sim.land_int,
                "maxTicks": sim.max_steps, "minSpawnDist": min_spawn_dist, "tickMs": 350},
    )


# --------------------------------------------------------------------------- #
# agents
# --------------------------------------------------------------------------- #
_uid = itertools.count()


def load_module(path: str, tag: str):
    """Execute a bot file under a unique module name (private globals)."""
    name = f"_arena_{tag}_{next(_uid)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses / pickling look modules up by name
    spec.loader.exec_module(mod)
    return mod


class RandomBot:
    def __init__(self, p, rng=None):
        self.p = p
        self.rng = rng or random.Random()

    def act(self, sim):
        p, step_to = self.p, sim.b.step_to
        acts = [c * 9 + 1 + d + h
                for c in range(sim.b.n) if sim.owner[c] == p and sim.army[c] >= 2
                for d in range(4) if step_to[c][d] >= 0 for h in (0, 4)]
        return self.rng.choice(acts) if acts else 0


class Agent:
    """One bot in one seat. `decide(sim)` -> legal action or 0, never raises.

    Loading (module import, JAX for the old bots) happens on the first
    `decide` unless `load()` is called first, so a GameSession can do it on
    its worker thread. Counters for the bench: calls, errors, illegal.
    """

    def __init__(self, info: BotInfo, seat: int, board: Board):
        self.info, self.seat, self.board = info, seat, board
        self.impl = None            # Planner / script object, or the act function
        self.plan: list[int] = []   # queued actions returned by act() bots
        self.load_error: str | None = None
        self.error: str | None = None      # first decide() exception
        self.calls = self.errors = self.illegal = 0

    # -- loading ----------------------------------------------------------
    def load(self) -> bool:
        if self.impl is not None or self.load_error is not None:
            return self.load_error is None
        try:
            self.impl = self._build()
        except BaseException as exc:            # a bot file must not take the app down
            self.load_error = f"{type(exc).__name__}: {exc}"
            print(f"[arena.bots] {self.info.id}: load failed: {self.load_error}", file=sys.stderr)
        return self.load_error is None

    def _build(self):
        import core36
        info, p = self.info, self.seat
        if info.kind == "builtin":
            import core37
            ctx = core36.Ctx(self.board)
            return {"expander": lambda: core37.Expand(ctx, p),
                    "rusher": lambda: core36.Rusher(ctx, p),
                    "castle_rusher": lambda: core36.CastleRusher(ctx, p),
                    "econ": lambda: core36.Econ(ctx, p),
                    "random": lambda: RandomBot(p)}[info.id]()
        mod = load_module(info.path, info.id)
        if info.kind == "planner":
            return mod.Planner(core36.Ctx(self.board), p)
        return mod.act

    # -- playing ----------------------------------------------------------
    def decide(self, sim: Sim) -> int:
        self.calls += 1
        if not self.load():
            return 0
        try:
            if self.info.kind == "act":
                a = self._act_bot(sim)
            else:
                a = self.impl.act(sim)
            a = int(a) if a else 0
        except Exception as exc:
            self.errors += 1
            if self.error is None:      # log once per agent
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"[arena.bots] {self.info.id} raised at tick {sim.step}: {self.error}\n"
                      + traceback.format_exc(limit=4), file=sys.stderr)
            return 0
        if a > 0 and not sim.legal(self.seat, a):
            self.illegal += 1
            return 0
        return max(a, 0)

    def _act_bot(self, sim: Sim) -> int:
        """act(state, player) bots; replay a returned path like local_server."""
        if not self.plan:
            from server.bot_protocol import normalise
            state = bot_state(sim)
            self.plan = [int(a) for a in normalise(self.impl(state, self.seat), state)[:32]]
        while self.plan:
            a = self.plan.pop(0)
            if a <= 0:
                return 0
            if sim.legal(self.seat, a):
                return a
            self.plan.clear()        # a stale path is meaningless once broken
        return 0


def make_agent(bot_id: str, seat: int, board: Board, sim: Sim | None = None,
               lazy: bool = False) -> Agent:
    """Agent for `bot_id` in `seat`. `sim` is accepted for symmetry (bots read
    the live sim in decide). lazy=True defers the import to the first decide."""
    agent = Agent(get_bot(bot_id), seat, board)
    if not lazy:
        agent.load()
    return agent


# --------------------------------------------------------------------------- #
# bench: python -m arena.bots --bench
# --------------------------------------------------------------------------- #
BENCH_TICKS = 40
BENCH_TIMEOUT_S = 240.0       # per job (one bot, one map) incl. imports
BENCH_SOFT_S = 150.0          # stop playing after this and report what we have


def _bench_job(bot_id, seed, seat, ticks, out_q):
    """Play `ticks` ticks of bot vs expander in a fresh process; report timings."""
    res = {"bot": bot_id, "seed": seed, "seat": seat, "ok": False}
    try:
        from arena.maps import load_map
        _seed, m, c, o, a, gi = load_map(seed)
        board = Board(m, c, 23, 23)
        sim = Sim(board, o, a, gi, 0, 600)
        t0 = time.perf_counter()
        agent = make_agent(bot_id, seat, board, sim)
        res["load_s"] = time.perf_counter() - t0
        opp = make_agent("expander", 1 - seat, board, sim)
        times, moves = [], 0
        t_start = time.perf_counter()
        while not sim.done and sim.step < ticks and time.perf_counter() - t_start < BENCH_SOFT_S:
            t = time.perf_counter()
            x = agent.decide(sim)
            times.append(1000 * (time.perf_counter() - t))
            y = opp.decide(sim)
            moves += x > 0
            sim.step2(*((x, y) if seat == 0 else (y, x)))
        res.update(ok=True, ticks=sim.step, times=times, moves=moves, calls=agent.calls,
                   errors=agent.errors, illegal=agent.illegal, load_error=agent.load_error,
                   error=agent.error)
    except BaseException as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
    out_q.put(res)


def bench(workers: int = 12, ticks: int = BENCH_TICKS, seeds=(3, 11), only=None) -> dict:
    """Run every bot (incl. broken ones) on `seeds`, seat 0 then seat 1; one
    process per job so a hanging bot can be killed at BENCH_TIMEOUT_S."""
    bots = [b for b in list_bots(include_broken=True) if not only or b.id in only]
    jobs = [(b.id, s, k % 2) for b in bots for k, s in enumerate(seeds)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    running, results = {}, []
    t0 = time.perf_counter()
    while jobs or running:
        while jobs and len(running) < workers:
            job = jobs.pop(0)
            proc = ctx.Process(target=_bench_job, args=(*job, ticks, q), daemon=True)
            proc.start()
            running[job] = (proc, time.perf_counter())
        try:
            r = q.get(timeout=0.5)
            results.append(r)
            proc, _ = running.pop((r["bot"], r["seed"], r["seat"]))
            proc.join(5)
            print(f"  {r['bot']:<38} seed {r['seed']:<3} done "
                  f"({time.perf_counter() - t0:.0f}s, {len(jobs)} queued)", flush=True)
        except Exception:      # queue.Empty: check for hung or dead jobs
            pass
        now = time.perf_counter()
        for job, (proc, started) in list(running.items()):
            if now - started > BENCH_TIMEOUT_S or (not proc.is_alive() and proc.exitcode not in (0, None)):
                proc.kill()
                proc.join(5)
                running.pop(job)
                why = "timeout" if now - started > BENCH_TIMEOUT_S else f"exit {proc.exitcode}"
                results.append({"bot": job[0], "seed": job[1], "seat": job[2], "ok": False, "error": why})
                print(f"  {job[0]:<38} seed {job[1]:<3} {why}", flush=True)

    rows = {}
    for b in bots:
        rs = [r for r in results if r["bot"] == b.id]
        times = [t for r in rs if r.get("ok") for t in r["times"][1:]]   # first call = warm-up
        first = [r["times"][0] for r in rs if r.get("ok") and r["times"]]
        calls = sum(r.get("calls", 0) for r in rs)
        errors = sum(r.get("errors", 0) for r in rs)
        illegal = sum(r.get("illegal", 0) for r in rs)
        failed = [r.get("load_error") or r.get("error") for r in rs
                  if not r.get("ok") or r.get("load_error")]
        ms_mean = sum(times) / len(times) if times else None
        if failed or not times or errors * 2 >= calls or (illegal and illegal >= calls - errors):
            tier = "broken"
        elif ms_mean < FAST_MS:
            tier = "fast"
        elif ms_mean < OK_MS:
            tier = "ok"
        else:
            tier = "slow"
        rows[b.id] = {
            "kind": b.kind, "tier": tier,
            "ms_mean": round(ms_mean, 2) if ms_mean is not None else None,
            "ms_max": round(max(times), 1) if times else None,
            "ms_first": round(max(first), 1) if first else None,
            "load_s": round(max((r.get("load_s", 0) for r in rs), default=0), 2),
            "ticks": sum(r.get("ticks", 0) for r in rs), "calls": calls,
            "errors": errors, "illegal": illegal,
            "moves": sum(r.get("moves", 0) for r in rs),
            "error": next((e for e in failed if e), None)
                     or next((r.get("error") for r in rs if r.get("error")), None),
        }
    index = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "ticks": ticks,
             "seeds": list(seeds), "opponent": "expander", "bots": rows}
    if only and INDEX.exists():                 # partial re-bench: merge
        old = json.loads(INDEX.read_text())
        old["bots"].update(rows)
        rows, index["bots"] = old["bots"], old["bots"]
    INDEX.write_text(json.dumps(index, indent=1))
    return rows


def print_table(rows: dict) -> None:
    print(f"{'bot':<38} {'kind':<8} {'tier':<7} {'mean ms':>8} {'max ms':>8} {'1st ms':>8} "
          f"{'load s':>7} {'err':>4} {'ill':>4}  note")
    order = {b.id: i for i, b in enumerate(list_bots(include_broken=True))}
    for bot_id, r in sorted(rows.items(), key=lambda kv: order.get(kv[0], 999)):
        f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
        print(f"{bot_id:<38} {r['kind']:<8} {r['tier']:<7} {f(r['ms_mean']):>8} {f(r['ms_max']):>8} "
              f"{f(r['ms_first']):>8} {f(r['load_s'], 2):>7} {r['errors']:>4} {r['illegal']:>4}  "
              f"{(r.get('error') or '')[:60]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true", help="benchmark every bot -> arena/bot_index.json")
    ap.add_argument("--only", nargs="*", help="bench just these bot ids (merged into the index)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ticks", type=int, default=BENCH_TICKS)
    a = ap.parse_args()
    if a.bench:
        print_table(bench(a.workers, a.ticks, only=set(a.only) if a.only else None))
    else:
        for b in list_bots(include_broken=True):
            ms = "-" if b.ms_mean is None else f"{b.ms_mean:.1f}ms"
            print(f"{b.id:<38} {b.kind:<8} {b.tier:<7} {ms:>8}  {b.name}  --  {b.summary}")
