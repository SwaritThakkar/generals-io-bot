"""Assemble the deck. One slide per idea, bullets only, figures from media/."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from build_deck import Deck, INK, MUTED, FAINT, ACCENT, RED, BLUE, GREEN, RULE, SERIF, SANS, MONO
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT = pathlib.Path(__file__).resolve().parents[1]
d = Deck()
I = Inches
L, CW, FW = I(0.62), I(5.65), I(12.1)          # left margin, column width, full width
RX, RW = I(6.68), I(6.04)                       # right column for figures
BODY = I(1.78)

# ---------------------------------------------------------------- title
s = d.p.slides.add_slide(d.blank)
sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, I(0.22), I(7.5))
sh.fill.solid(); sh.fill.fore_color.rgb = ACCENT; sh.line.fill.background(); sh.shadow.inherit = False
tf = d._tb(s, I(1.3), I(2.25), I(10.6), I(0.3))
r = tf.paragraphs[0].add_run(); r.text = "DAC-203   ·   PROJECT NOTES"
f = r.font; f.name = SANS; f.size = Pt(12); f.bold = True; f.color.rgb = ACCENT
tf = d._tb(s, I(1.3), I(2.72), I(10.6), I(1.9))
p = tf.paragraphs[0]
r = p.add_run(); r.text = "A Search-Only Agent for\n"
f = r.font; f.name = SERIF; f.size = Pt(42); f.color.rgb = INK
r = p.add_run(); r.text = "Perfect-Information Generals.io"
f = r.font; f.name = SERIF; f.size = Pt(42); f.bold = True; f.color.rgb = INK
d._line(s, I(1.3), I(4.62), I(3.2), color=INK, pt=1.4)
tf = d._tb(s, I(1.3), I(4.92), I(10.0), I(1.0))
for i, t in enumerate(["Real-time strategy on a 23x23 grid.  b between 40 and 400.  500 ms per decision.",
                       "Five papers, one per chapter of Russell & Norvig 3-6.  No learning of any kind."]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(6)
    r = p.add_run(); r.text = t
    f = r.font; f.name = SANS; f.size = Pt(14); f.color.rgb = MUTED
d.n = 1

def divider(num, title, sub):
    s = d.p.slides.add_slide(d.blank); d.n += 1
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, I(13.333), I(7.5))
    sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    sh.line.fill.background(); sh.shadow.inherit = False
    tf = d._tb(s, I(1.3), I(2.9), I(2.0), I(1.4))
    r = tf.paragraphs[0].add_run(); r.text = num
    f = r.font; f.name = SERIF; f.size = Pt(64); f.bold = True; f.color.rgb = ACCENT
    tf = d._tb(s, I(2.8), I(3.05), I(9.0), I(0.8))
    r = tf.paragraphs[0].add_run(); r.text = title
    f = r.font; f.name = SERIF; f.size = Pt(36); f.bold = True
    f.color.rgb = RGBColor(0xF4, 0xF2, 0xEE)
    tf = d._tb(s, I(2.82), I(3.86), I(9.0), I(0.5))
    r = tf.paragraphs[0].add_run(); r.text = sub
    f = r.font; f.name = SANS; f.size = Pt(14); f.color.rgb = RGBColor(0x9A, 0x94, 0x88)
    return s

# ================================================================ 1. THE GAME
divider("1", "The Game", "Rules, production, and the three things that make it hard.")

s = d.slide("1. the game", "The board")
d.bullets(s, [
    "23 x 23 grid of square tiles",
    ("plain tiles - anyone may walk on them and own them", 1),
    ("mountains - impassable, permanent walls", 1),
    ("castles - produce army for whoever holds them", 1),
    ("generals - one per player; a base, and the thing you must defend", 1),
    "",
    "Each player starts owning exactly one tile: their general",
    "Every owned tile carries a number - the army units standing on it",
    ("mine red, opponent blue, unowned grey", 1),
], width=I(4.5))
d.figure(s, "d01-board", I(5.3), I(1.62), I(7.4), I(5.0))

s = d.slide("1. the game", "Moving and fighting")
d.bullets(s, [
    "One move per tick: pick a tile you own, pick a direction",
    ("send all but one unit, or exactly half", 1),
    ("or pass", 1),
    "That is the entire action set",
    "",
    ("Combat is subtraction", 0, RED),
    "The two stacks cancel one for one; the larger survives with the difference",
], width=I(5.0))
d.equation(s, r"\mathrm{survivor} = |a - b|,\quad \mathrm{owner} = \arg\max(a,b)",
           "combat", I(0.62), I(5.02), I(0.42))
d.note(s, "Ties keep the tile with its previous owner.", I(0.62), I(5.82), I(5.0),
       color=MUTED, bold=False)
d.figure(s, "d02-combat", I(5.9), I(1.72), I(6.8), I(4.6))

s = d.slide("1. the game", "A stack of size a captures a - 1 tiles")
d.figure(s, "d03-stack-walks", I(0.62), I(1.75), I(12.1), I(3.55))
d.bullets(s, [
    "Neutral tiles cost almost nothing - walk on with any units at all and it is yours",
    "One unit is left behind on every tile you leave, so the stack shrinks by one per step",
    "This is the mental model for the entire expansion phase",
], top=I(5.62), width=I(12.1))

s = d.slide("1. the game", "Where army comes from")
d.bullets(s, [
    "Your general produces 1 unit every 2 ticks",
    "Each castle you own produces 1 unit every 2 ticks",
    "Every tile you own produces 1 unit every 50 ticks",
    "",
    "Neutral castles sit on the map holding 40-50 units - beat the garrison, keep it forever",
], width=I(5.7))
d.equation(s, r"\rho \;=\; \frac{1}{2}\,P \;+\; \frac{1}{50}\,L"
              r"\qquad\Longrightarrow\qquad \frac{0.50}{0.02} \;=\; 25",
           "rho", I(0.62), I(4.28), I(0.50))
d.note(s, "One castle is worth exactly twenty-five plain tiles. "
          "A fact about the rules, not an opinion about strategy.",
       I(0.62), I(5.18), I(5.7))
d.figure(s, "d21-castle-value", I(6.5), I(2.05), I(6.2), I(4.3))

s = d.slide("1. the game", "Therefore: a compounding game")
d.bullets(s, [
    "Territory produces army",
    "Army takes territory",
    "Small early advantages snowball",
    "",
    ("Ten tiles ahead at turn 50", 0, RED),
    ("is usually much further ahead by turn 200", 1),
    "",
    "The curves diverge and never come back together",
], width=I(4.2))
d.figure(s, "d05-compounding", I(4.9), I(1.62), I(7.9), I(5.0))

s = d.slide("1. the game", "How you win")
d.bullets(s, [
    "Move an army onto the enemy general and win the fight there",
    "",
    "That is it",
    ("no points system", 1),
    ("no other victory condition", 1),
], width=I(4.6))
d.figure(s, "d04-how-you-win", I(5.2), I(2.45), I(7.5), I(4.0))

s = d.slide("1. the game", "Both players see everything")
d.bullets(s, [
    "The board is fully observable - nothing on it is hidden",
    "This is what makes the algorithms in this course apply at all",
    "",
    ("Classical adversarial search assumes you can evaluate the position "
     "in front of you. Here you can.", 0, RED),
    "",
    "Evaluation is exact, not expected",
    "\"Can my opponent reach my general within six ticks?\" has a definite "
    "yes-or-no answer, not a probability",
], width=I(5.0))
d.figure(s, "d06-visible", I(5.9), I(1.68), I(6.8), I(4.9))

s = d.slide("1. the game", "What makes it hard")
d.bullets(s, [
    ("ONE   ·   Time", 0, RED),
    ("A real-time strategy game. Both players move together, "
     "so tempo is a first-class resource.", 1),
    "",
    ("TWO   ·   Branching factor", 0, RED),
    ("Every owned tile with two or more units has 4 directions x 2 move types. "
     "Average b lands between 40 and 400.", 1),
    "",
    ("THREE   ·   Sparse, terminal reward", 0, RED),
    ("You win or lose 500 ticks later. Nothing tells you at tick 137 "
     "whether that move was good.", 1),
    "",
    ("AND   ·   Simultaneous commitment", 0, ACCENT),
    ("The one genuine piece of hidden information - and the reason plain "
     "minimax is not quite right.", 1),
], width=I(12.1), size=15, gap=3)

s = d.slide("1. the game  ·  hard #2", "Depth is unaffordable")
d.figure(s, "fig2-branching-factor", I(0.62), I(1.72), I(12.1), I(4.6))
d.note(s, "b = 204, measured from a real position. Chess is 35.",
       I(0.62), I(6.50), I(11.4))

s = d.slide("1. the game  ·  hard #3", "The reward is sparse and terminal")
d.figure(s, "d07-sparse-reward", I(0.62), I(1.72), I(12.1), I(3.9))
d.bullets(s, [
    "No signal for 500 ticks, then +1 or -1",
    "So the evaluation function is doing all of the work - it is the hard part, "
    "not the search",
], top=I(5.88), width=I(12.1))

s = d.slide("1. the game", "Simultaneous moves")
d.bullets(s, [
    "Both players commit their move for the same tick without seeing the other's",
    "",
    ("Minimax quietly assumes you move second", 0, RED),
    ("that you get to see what the opponent did", 1),
    "",
    "A node here is a matrix game, not a max or a min",
    "",
    "Dealing with this properly is an open question in this project",
], width=I(4.7))
d.figure(s, "d08-simultaneous", I(5.4), I(1.66), I(7.3), I(4.9))

# ================================================= 2. WHY SEARCH
divider("2", "Why Search", "And why it is not a fallback.")

s = d.slide("2. why search", "Search is a serious approach here")
d.bullets(s, [
    "The constraint: search and CSP only - Russell & Norvig, chapters 3 to 6",
    ("no reinforcement learning, no neural networks, no machine learning", 1),
    "",
    ("The strongest non-learning agent on the public ladder is built entirely "
     "out of classical optimisation", 0, RED),
    ("graph algorithms and dynamic programming - nothing else", 1),
    ("it was the best bot in the world at this game until an RL agent trained "
     "on four H200 GPUs beat it", 1),
    "",
    "So this is not a consolation prize. It is the approach that held the top "
    "of the ladder.",
], width=I(12.1))

s = d.slide("2. why search", "The sub-problems are genuinely NP-hard")
d.bullets(s, [
    ("Gathering scattered army to one point within a turn budget", 0, INK),
    ("rooted prize-collecting Steiner tree", 1, RED),
    "",
    ("Deciding what to spend a limited number of turns on", 0, INK),
    ("multiple-choice knapsack", 1, RED),
    "",
    "There is no closed form here",
    "",
    ("When a problem has no analytic solution and a combinatorial "
     "search space, search is what you use.", 0, ACCENT),
], width=I(4.9))
d.figure(s, "d09-nphard", I(5.6), I(1.66), I(7.1), I(4.9))

# ================================================= 3. THE PAPERS
divider("3", "The Papers", "Five papers, one per topic in the course.")

s = d.slide("3. the papers", "One paper per topic")
rows = [("3.1", "ARA*: Anytime A* with Provable Bounds on Sub-Optimality",
         "Likhachev, Gordon, Thrun  ·  NIPS 2003", "an answer at every moment"),
        ("3.2", "Real-Time Adaptive A*  (RTAA*)",
         "Koenig, Likhachev  ·  AAMAS 2006", "learn more per unit of search"),
        ("3.3", "Depression Avoidance in Real-Time Heuristic Search",
         "Hernandez, Baier  ·  AAAI 2012", "one bit per state, no scrubbing"),
        ("3.4", "ProbCut: Forward Pruning in Alpha-Beta",
         "Buro", "prune what is probably irrelevant"),
        ("3.5", "Metareasoning: deciding how much to think",
         "O'Ceallaigh, Ruml  ·  SoCS 2015   /   Russell, Wefald  ·  UAI 1990",
         "should I be searching at all?")]
y = I(1.85)
for num, title, cite, gist in rows:
    tf = d._tb(s, I(0.62), y, I(0.7), I(0.3))
    r = tf.paragraphs[0].add_run(); r.text = num
    f = r.font; f.name = MONO; f.size = Pt(15); f.bold = True; f.color.rgb = ACCENT
    tf = d._tb(s, I(1.38), y - I(0.02), I(7.4), I(0.34))
    r = tf.paragraphs[0].add_run(); r.text = title
    f = r.font; f.name = SERIF; f.size = Pt(16); f.bold = True; f.color.rgb = INK
    tf = d._tb(s, I(1.38), y + I(0.30), I(7.4), I(0.28))
    r = tf.paragraphs[0].add_run(); r.text = cite
    f = r.font; f.name = SANS; f.size = Pt(10.5); f.color.rgb = FAINT
    tf = d._tb(s, I(8.95), y + I(0.05), I(3.8), I(0.3))
    r = tf.paragraphs[0].add_run(); r.text = gist
    f = r.font; f.name = SANS; f.size = Pt(12); f.italic = True; f.color.rgb = MUTED
    y = y + I(1.0)
    if num != "3.5":
        d._line(s, I(0.62), y - I(0.22), I(12.1), color=RGBColor(0xE4, 0xE4, 0xE4))

# ---- 3.1 ARA*
s = d.slide("3.1  ARA*   ·   Likhachev, Gordon & Thrun, NIPS 2003",
            "A* gives you nothing until it finishes")
d.bullets(s, [
    "If my half second runs out mid-search I have no answer at all",
    "The tick fires regardless",
    "",
    ("Weighted A* is the classical fix", 0, RED),
    "Inflate the heuristic, and the search becomes greedy and much faster",
], width=I(12.1))
d.equation(s, r"f(n) \;=\; g(n) \;+\; \varepsilon\, h(n), \qquad \varepsilon > 1",
           "wastar", I(0.62), I(3.72), I(0.52))
d.note(s, "This comes with a theorem, not a hope:", I(0.62), I(4.62), I(11.4))
d.equation(s, r"\mathrm{cost}(\text{path found}) \;\leq\; \varepsilon \cdot "
              r"\mathrm{cost}(\text{optimal path})",
           "bound", I(0.96), I(5.10), I(0.52), color="#B4342C")
d.bullets(s, ["A dial between speed and quality - and a guarantee at every setting"],
          top=I(6.05), width=I(12.1))

s = d.slide("3.1  ARA*", "Weighted A* trades quality for speed")
d.figure(s, "d10-weighted-astar", I(0.62), I(1.72), I(12.1), I(4.5))
d.note(s, "Inflating h makes the search directional: 103 expansions at "
          "eps = 1 becomes 27 at eps = 3, and the path costs 24 instead of 26.",
       I(0.62), I(6.42), I(11.4), color=MUTED, bold=False)

s = d.slide("3.1  ARA*", "Anytime: run it repeatedly, epsilon decreasing")
d.bullets(s, [
    "Run with eps = 3, then 2, then 1.5, then 1.2",
    "Each run reuses the previous run's work instead of starting over",
    "",
    ("A rough answer almost immediately", 0, RED),
    ("It improves the longer you let it run", 0, RED),
    ("At every moment you know exactly how far from optimal you are", 0, RED),
    "",
    "Plain A* would still be returning nothing at the deadline",
], width=I(4.6))
d.figure(s, "d11-anytime", I(5.3), I(1.66), I(7.4), I(4.9))

s = d.slide("3.1  ARA*", "The reuse: a third list")
d.figure(s, "d12-three-lists", I(0.62), I(1.72), I(12.1), I(3.8))
d.bullets(s, [
    "A state whose g improves after it was closed goes to INCONS, not back to OPEN",
    "At the end of a pass, all of INCONS is poured back into OPEN for the next eps",
    ("that last arrow is the reuse", 1),
], top=I(5.68), width=I(12.1), size=14, gap=3)

# ---- 3.2 RTAA*
s = d.slide("3.2  RTAA*   ·   Koenig & Likhachev, AAMAS 2006",
            "Acting before you have finished thinking")
d.bullets(s, [
    "The setting: you must act now, you cannot search to the goal, and you will "
    "be back in a similar situation next step",
    "",
    ("LRTA*, the algorithm in the book", 0, INK),
    ("keeps a table of heuristic estimates", 1),
    ("raises the estimate of the state it stands in when neighbours turn out "
     "further from the goal than expected", 1),
    ("over repeated visits the table gets accurate and the agent stops "
     "repeating its mistakes", 1),
    "",
    ("The weakness: it updates essentially one state per move", 0, RED),
    ("on a large map the agent physically walks around correcting "
     "one number at a time", 1),
    "",
    ("RTAA*: bounded-depth A*, then update EVERY state that search expanded - "
     "one subtraction each", 0, ACCENT),
], width=I(12.1), size=14.5, gap=3)

s = d.slide("3.2  RTAA*", "Same lookahead, vastly more information")
d.figure(s, "d13-lrta-vs-rtaa", I(0.62), I(1.72), I(12.1), I(4.4))
d.note(s, "Same search region, same cost per move. This single comparison "
          "is the whole paper.", I(0.62), I(6.32), I(11.4))

s = d.slide("3.2  RTAA*", "Why the update is legal")
d.bullets(s, [
    "A* expands in order of increasing f",
    "So when it stops, the smallest f left on the frontier is a lower bound "
    "on the true distance to the goal",
    "Any path out of the searched region must cross that frontier",
], width=I(5.2))
d.equation(s, r"\mathrm{dist}(s,\, \mathrm{goal}) \;\geq\; f(\bar{s}) \;-\; g(s)",
           "rtaa_bound", I(0.62), I(3.52), I(0.46))
d.equation(s, r"h[s] \;\leftarrow\; g[\bar{s}] \;+\; h[\bar{s}] \;-\; g[s]",
           "rtaa_update", I(0.62), I(4.28), I(0.46), color="#B4342C")
d.bullets(s, [
    "It is a lower bound, so h stays admissible - everything A* guarantees "
    "still holds",
    "The paper also proves the update is monotonically non-decreasing: "
    "the table only ever becomes more informed",
], top=I(5.05), width=I(5.2), size=13.5)
d.figure(s, "d14-why-legal", I(6.1), I(1.66), I(6.6), I(4.9))

s = d.slide("3.2  RTAA*", "Two lines the paper leaves undefined")
d.figure(s, "d18-two-params", I(0.62), I(1.72), I(12.1), I(3.62))
d.bullets(s, [
    "The paper is honest about the trade-off: LRTA*'s heuristic ends up better "
    "informed for the same lookahead - RTAA* is simply so much cheaper per move "
    "that it reaches the goal sooner in wall-clock terms",
    ("which is precisely my situation: I am limited by time, not by "
     "number of moves", 1),
    ("and those two undefined lines are where a later paper lives", 1, ACCENT),
], top=I(5.50), width=I(12.1), size=13.5, gap=2)

# ---- 3.3 Depression
s = d.slide("3.3  Depression avoidance   ·   Hernandez & Baier, AAAI 2012",
            "A specific way real-time search fails")
d.bullets(s, [
    "A heuristic depression is a connected region where h badly underestimates "
    "true cost, compared with the values on the region's border",
    "",
    ("Picture h as a landscape where low means close to the goal. "
     "A depression is a bowl.", 0, RED),
    "",
    "A greedy agent rolls in - from outside, everything in the bowl looks close "
    "to the goal",
    "Then it cannot get out: each move raises one state's value slightly",
    "It wanders the bowl raising values one at a time until the whole thing is "
    "filled in",
    "",
    ("The literature calls this scrubbing. It can waste hundreds of moves.", 0, ACCENT),
], width=I(5.0), size=13, gap=2)
d.figure(s, "d15-depression", I(5.9), I(1.66), I(6.8), I(4.9))

s = d.slide("3.3  Depression avoidance", "The fix is one sentence, and detection is free")
d.bullets(s, [
    ("Prefer to move to states that are not known to be inside a depression.", 0, RED),
    "",
    "h starts admissible, and updates only ever raise it",
    "So the fact that you ever had to raise a state's value means the original "
    "estimate there was too optimistic",
    ("which is the definition of being in a depression", 1),
    "",
    ("Cost: one extra bit per state - remember h0, mark the state when h > h0", 0, ACCENT),
], width=I(5.2), size=14)
d.figure(s, "d16-detection", I(6.1), I(1.66), I(6.6), I(4.9))

s = d.slide("3.3  Depression avoidance", "Two ways to use the mark")
d.bullets(s, [
    ("MARK-AND-AVOID", 0, RED),
    ("prefer unmarked states when choosing where to head", 1),
    ("fall back to marked ones only if everything is marked", 1),
    "",
    ("MOVE-TO-BORDER", 0, BLUE),
    ("softer: if you are already deep in a depression and every neighbour "
     "is marked, aim for the states that look closest to the edge", 1),
    ("rather than refusing to be in the bowl at all", 1),
    "",
    ("There is almost no mathematics in this paper, which I take as a good sign "
     "- it is an observation rather than a construction.", 0, ACCENT),
], width=I(12.1), size=15, gap=4)

# ---- 3.4 ProbCut
s = d.slide("3.4  ProbCut   ·   Buro", "Prune what is probably irrelevant")
d.bullets(s, [
    "Alpha-beta only prunes when it can prove irrelevance - and proof requires "
    "searching",
    ("so depth stays low, and depth is what makes a program strong", 1),
    "",
    "ProbCut: before searching a subtree to full depth, run a much shallower "
    "search on it. If the shallow result lands far enough outside the current "
    "window, skip the deep search.",
], width=I(12.1), size=15)
d.equation(s, r"v \;=\; a\,v' \;+\; b \;+\; e, \qquad e \sim \mathcal{N}(0,\,\sigma^2)",
           "probcut", I(0.62), I(3.72), I(0.50))
d.bullets(s, [
    "v is the deep value, v' the shallow one - strongly correlated, because a "
    "position winning after two ply is usually winning after six",
    "a, b and sigma are fitted by ordinary least squares over a few thousand "
    "positions, offline",
    ("that is regression, not machine learning in the sense the course excludes", 1, ACCENT),
], top=I(4.55), width=I(12.1), size=14, gap=4)
d.equation(s, r"\text{prune if}\quad v' \;\geq\; \frac{T\sigma + \beta - b}{a}",
           "probcut_rule", I(0.62), I(6.08), I(0.62), color="#B4342C")

s = d.slide("3.4  ProbCut", "Does shallow predict deep in this game?")
d.figure(s, "d17-shallow-deep", I(0.62), I(1.72), I(12.1), I(4.35))
d.note(s, "Measured in our own simulator: 1-ply against 2-ply over 138 self-play "
          "positions. Tight enough to prune - and a sign the evaluation needs more "
          "than land and army.", I(0.62), I(6.26), I(11.4), color=MUTED, bold=False)

# ---- 3.5 Metareasoning
s = d.slide("3.5  Metareasoning   ·   Russell & Wefald UAI 1990,  O'Ceallaigh & Ruml SoCS 2015",
            "Should I be searching at all right now?")
d.bullets(s, [
    "The four papers above all answer \"how do I search well?\". This one asks "
    "a different question.",
    "",
    "Treat each possible node expansion as an action in its own right:",
    ("COST - the time it takes", 1),
    ("BENEFIT - P(this changes which move I finally play) x how much better "
     "that move is", 1),
    "",
    "Expand the node with the highest value of computation; stop when the "
    "expected benefit drops below the cost",
], width=I(12.1), size=15)
d._line(s, I(0.62), I(5.32), I(12.1), color=ACCENT, pt=1.2)
tf = d._tb(s, I(0.62), I(5.58), I(12.1), I(0.9))
r = tf.paragraphs[0].add_run()
r.text = "A computation that cannot change your answer has zero value,\nno matter how deep it is."
f = r.font; f.name = SERIF; f.size = Pt(21); f.italic = True; f.color.rgb = ACCENT
tf = d._tb(s, I(0.62), I(6.34), I(12.1), I(0.5))
r = tf.paragraphs[0].add_run()
r.text = "Obvious written down - and almost no game-playing program acts on it. The standard design searches to a fixed depth regardless of whether the decision was ever in doubt."
f = r.font; f.name = SANS; f.size = Pt(12); f.color.rgb = MUTED

s = d.slide("3.5  Metareasoning", "Value of computation")
d.figure(s, "d19-voc", I(0.62), I(1.72), I(12.1), I(4.5))
d.note(s, "Wildly different children: expanding further cannot change the answer. "
          "Nearly tied children: one more expansion could flip it.",
       I(0.62), I(6.42), I(11.4))

s = d.slide("3.5  Metareasoning", "Two decision points - the two undefined lines")
d.bullets(s, [
    ("DECISION 1   ·   the identity action", 0, RED),
    ("an action that keeps you in the same state while you think longer", 1),
    ("act now, or stand still and deliberate?", 1),
    ("standard real-time search never asks - it assumes acting is always right", 1),
    ("but if the best move is only marginally better than the second best, "
     "another round of lookahead might turn up something much better", 1),
    ("measured in goal achievement time = planning time + execution time", 1),
    "",
    ("DECISION 2   ·   prefix commitment", 0, BLUE),
    ("after lookahead you hold a path - how much of it do you execute "
     "before replanning?", 1),
    ("LRTA* commits to exactly one action; later variants commit to the "
     "whole path. Neither adapts.", 1),
    ("one action means replanning constantly, which is expensive", 1),
    ("the whole path means acting on stale information, which is dangerous", 1),
    ("the paper chooses dynamically - further when the plan is robust, "
     "less when it is fragile", 1),
], width=I(12.1), size=14, gap=1)

s = d.slide("3.5  Metareasoning", "Both mechanics come free in this game")
d.figure(s, "d20-mechanics", I(0.62), I(1.72), I(12.1), I(3.6))
d.bullets(s, [
    "Index 0 on every tile is PASS - the agent can literally stand still and "
    "think, and the rules permit it",
    "The move queue lets you queue a path in advance; the server plays one per "
    "tick - prefix commitment, implemented for us",
    ("this literature usually has to simulate both. That is why I think my own "
     "contribution sits here.", 1, ACCENT),
], top=I(5.48), width=I(12.1), size=13.5, gap=2)

# ================================================= 5. THE IDEAS
divider("5", "The Ideas I Want to Test", "Hypotheses rather than claims.")

s = d.slide("idea 1", "The value of computation is measurable, not estimated")
d.bullets(s, [
    "Russell & Wefald ask you to weigh a computation's cost against the "
    "probability it changes your action",
    ("the hard part has always been estimating that probability "
     "before doing the computation", 1),
    "",
    ("Here I do not have to estimate it", 0, RED),
    ("the batched evaluator gives the full one-ply score distribution "
     "in under a millisecond", 1),
    ("the gap between best and second-best move is a computed proxy for "
     "whether more thinking could change the answer", 1),
    "",
    ("The cost side is quantifiable too", 0, RED),
    ("passing costs exactly one tick of tempo", 1),
    ("the production rate is fixed by the rules, so a tick of thinking "
     "is priced in army units", 1),
    "",
    ("Both sides of the inequality are computable, which is unusual.", 0, ACCENT),
], width=I(5.6), size=13.5, gap=2)
d.figure(s, "fig4-value-of-computation", I(6.5), I(1.68), I(6.2), I(4.9))

s = d.slide("idea 2", "Speculative pre-computation")
d.bullets(s, [
    "Production is completely deterministic - after the map is generated "
    "there is no randomness at all",
    "So if nobody moved, the board ten ticks from now is exactly computable "
    "from the board today",
    "",
    "The only uncertainty about the near future is the handful of moves each "
    "side makes - and each move perturbs a tiny region of a 500-tile board",
    "",
    ("So start searching next tick's problem during this tick, and patch in "
     "the few changed tiles when the real moves arrive", 0, RED),
    "",
    ("Done properly this roughly doubles the effective compute budget "
     "without making anything faster", 0, ACCENT),
    "",
    "It depends entirely on the game being deterministic and fully observable: "
    "I can only extrapolate a board I can actually see",
], width=I(5.4), size=13.5, gap=3)
d.figure(s, "fig5-speculative-precomputation", I(6.3), I(1.68), I(6.4), I(4.9))

s = d.slide("idea 3", "The move queue is a compute budget")
d.bullets(s, [
    "Normally the queue is treated as a user-interface convenience",
    "",
    ("Every move I queue is a tick of thinking time I have bought", 0, RED),
    ("during a queued move I owe the game no decision", 1),
    ("queue a twelve-step march home through my own territory and I have "
     "bought six seconds of uninterrupted computation", 1),
    "",
    "The commit decision then stops being about confidence and becomes a trade:",
    ("how much thinking time do I want", 1),
    ("against how long am I willing to act on stale information", 1),
    "",
    ("This is prefix commitment from the metareasoning literature - but with an "
     "interpretation those papers do not have, because their domains have "
     "no queue.", 0, ACCENT),
], width=I(5.4), size=13.5, gap=2)
d.figure(s, "fig6-queue-as-budget", I(6.3), I(1.68), I(6.4), I(4.9))

s = d.slide("architecture", "The backbone")
d.figure(s, "fig3-tree-regions", I(0.62), I(1.72), I(12.1), I(4.5))
d.note(s, "Region decomposition over the tree - the structural backbone the "
          "five papers plug into.", I(0.62), I(6.42), I(11.4))

out = ROOT / "Search-Only-Generals-AI.pptx"
d.save(out)
print(f"{d.n} slides -> {out}")
