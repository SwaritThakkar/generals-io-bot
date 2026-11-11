import pathlib
import sys, pathlib, pickle
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2]), str(pathlib.Path(__file__).resolve().parents[2] / "models")]
from arena.maps import load_map
from arena.bots import make_agent
from sim36 import Board, Sim
OPP = "general_rusher"
def play(bot, seed, seat, cap=400, record=False):
    _s, m, c, o, a, gi = load_map(seed)
    board = Board(m, c, 23, 23)
    sim = Sim(board, o, a, gi, 0, cap)
    me = make_agent(bot, seat, board, sim); op = make_agent(OPP, 1 - seat, board, sim)
    frames = []
    while not sim.done:
        x = me.decide(sim); y = op.decide(sim)
        if record:
            frames.append({"t": sim.step, "owner": list(sim.owner), "army": list(sim.army), "gen": list(sim.gen), "a": x, "b": y})
        sim.step2(*((x, y) if seat == 0 else (y, x)))
    return sim.winner, sim.step, frames, (m, c)
if sys.argv[1] == "scan":
    for seed in range(1, 16):
        for seat in (0, 1):
            w, t, _, _ = play("v11_launch_intercept", seed, seat)
            if w == 1 - seat and t < 400:
                w2, t2, _, _ = play("v36_scripted_guard", seed, seat)
                print("seed", seed, "seat", seat, "v11 lost at", t, "| v36:", "won" if w2 == seat else ("lost" if w2 == 1 - seat else "draw"), t2, flush=True)
else:
    seed, seat = int(sys.argv[2]), int(sys.argv[3])
    for bot in ("v11_launch_intercept", "v36_scripted_guard"):
        w, t, fr, mc = play(bot, seed, seat, record=True)
        pickle.dump({"frames": fr, "mountain": mc[0], "castle": mc[1], "winner": w, "end": t, "seat": seat},
                    open(f"{sys.argv[4]}_{bot[:3]}.pkl", "wb"))
        print(bot, w, t)
