import pathlib
import sys, json, pickle, importlib.util, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"models"), str(ROOT/"models/tools36")]
s = importlib.util.spec_from_file_location("v51rec", ROOT/"models/v51_fast_defence.py"); v51 = importlib.util.module_from_spec(s); s.loader.exec_module(v51)
import adversaries38
from arena36 import start
ctx, sim = start(93003, 600)
me = v51.Planner51(ctx, 0); op = adversaries38.GatherRusher(ctx, 1, t0=200)
_v42 = v51._v42
frames = []
while not sim.done:
    pre = (list(sim.owner), list(sim.army), list(sim.gen))
    a = me.act(sim)
    thr, m0 = me.econ.margin_of(sim)
    g = sim.gen[0]
    them = _v42._items_them(ctx, sim, 1, g) if g >= 0 else []
    us = _v42._items_us(ctx, sim, 0, g) if g >= 0 else []
    b = op.act(sim); b = b if b and b > 0 else 0
    f = {"t": sim.step, "owner": pre[0], "army": pre[1], "gen": pre[2], "a": int(a or 0), "b": int(b),
         "why": me.why, "m0": None if m0 >= 10**8 else m0, "incoming": me.incoming,
         "def_moves": list(me.def_moves), "tp": dict(me.type_p),
         "weights": dict(me.weights), "gates": me.gates(), "arm": me.arm,
         "macro": me.macro_name, "them": [(d, x, src, sorted(ps)) for d, x, src, ps in them],
         "us": [(d, x, src, sorted(ps)) for d, x, src, ps in us],
         "castles": [sum(1 for c in ctx.castles if sim.owner[c] == q) for q in (0, 1)],
         "totals": sim.totals()}
    frames.append(f)
    sim.step2(a or 0, b)
meta = {"mountain": ctx.b.mountain, "castle": ctx.b.castle, "h": ctx.b.h, "w": ctx.b.w, "castles": ctx.castles,
        "winner": sim.winner, "end": sim.step}
pickle.dump({"frames": frames, "meta": meta}, open(sys.argv[1], "wb"))
print("winner", sim.winner, "end", sim.step, "frames", len(frames))
import collections
c = collections.Counter(f["why"].split(":",1)[1] if ":" in f["why"] else f["why"] for f in frames)
print(c.most_common(15))
first_castle = next((f["t"] for f in frames if f["castles"][0] > 0), None)
print("first castle", first_castle)
for f in frames:
    if f["t"] in (40, 100, 150, 180, 190, 200, 210, 215, 219, 222, 226, 230, 234, 236, 237, 238, 300, 370, 400, 406, 407):
        print(f["t"], f["why"], f["m0"], f["incoming"], len(f["def_moves"]), f["castles"], max(f["tp"], key=f["tp"].get), round(max(f["tp"].values()),2), f["macro"])
