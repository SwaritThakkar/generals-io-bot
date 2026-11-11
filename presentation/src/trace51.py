import pathlib
import sys, json, importlib.util, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"models"), str(ROOT/"models/tools36")]
def load(f, n):
    s = importlib.util.spec_from_file_location(n, ROOT/f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
import adversaries38
from arena36 import start
from core36 import top_stacks
out = {}
for name, f in (("v46", "models/v46_opponent_learner.py"), ("v51", "models/v51_fast_defence.py")):
    bot = load(f, name + "_tr")
    ctx, sim = start(93003, 600)
    me = bot.Planner(ctx, 0); op = adversaries38.GatherRusher(ctx, 1, t0=200)
    rows = []
    while not sim.done:
        a = me.act(sim)
        g = sim.gen[0]; dg = ctx.dist(g)
        st = [s for s in top_stacks(sim, 1, 3, 5) if s != sim.gen[1]]
        att = max(st, key=lambda s: sim.army[s]) if st else None
        r = {"t": sim.step, "G": sim.army[g], "att": sim.army[att] if att is not None else 0,
             "d": dg[att] if att is not None else None, "why": me.why,
             "tot": sum(sim.army[c] for c in range(len(sim.owner)) if sim.owner[c] == 0)}
        if name == "v51":
            r["inc"] = me.incoming is not None
            r["tp"] = {k: round(v, 3) for k, v in me.type_p.items()}
        rows.append(r)
        b = op.act(sim)
        sim.step2(a or 0, b if b and b > 0 else 0)
    out[name] = {"rows": rows, "winner": sim.winner, "end": sim.step}
    print(name, "winner", sim.winner, "end", sim.step)
json.dump(out, open(sys.argv[1], "w"))
