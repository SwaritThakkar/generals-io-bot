"""The five-minute cut. Ten slides, same figures and same voice as the full deck."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from build_deck import Deck, INK, MUTED, FAINT, ACCENT, RED, BLUE, SERIF, SANS, MONO
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT = pathlib.Path(__file__).resolve().parents[1]
d = Deck()
I = Inches

# ------------------------------------------------------------------ 1. title
s = d.p.slides.add_slide(d.blank)
sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, I(0.22), I(7.5))
sh.fill.solid(); sh.fill.fore_color.rgb = ACCENT
sh.line.fill.background(); sh.shadow.inherit = False
tf = d._tb(s, I(1.3), I(2.4), I(10.6), I(0.3))
r = tf.paragraphs[0].add_run(); r.text = "DAC-203   ·   FIVE-MINUTE VERSION"
f = r.font; f.name = SANS; f.size = Pt(12); f.bold = True; f.color.rgb = ACCENT
tf = d._tb(s, I(1.3), I(2.87), I(10.6), I(1.9))
p = tf.paragraphs[0]
r = p.add_run(); r.text = "A Search-Only Agent for\n"
f = r.font; f.name = SERIF; f.size = Pt(42); f.color.rgb = INK
r = p.add_run(); r.text = "Perfect-Information Generals.io"
f = r.font; f.name = SERIF; f.size = Pt(42); f.bold = True; f.color.rgb = INK
d._line(s, I(1.3), I(4.77), I(3.2), color=INK, pt=1.4)
tf = d._tb(s, I(1.3), I(5.07), I(10.0), I(0.6))
r = tf.paragraphs[0].add_run()
r.text = "Five papers, one per chapter of Russell & Norvig 3-6.  No learning of any kind."
f = r.font; f.name = SANS; f.size = Pt(14); f.color.rgb = MUTED
d.n = 1

# ------------------------------------------------------------------ 2. the game
s = d.slide("the game", "The board, and the whole action set")
d.bullets(s, [
    "A 23 x 23 grid: plain tiles, mountains, castles, one general each",
    "Every tile I own carries a number - the army standing on it",
    "",
    "One move per tick: a tile, a direction, and all but one unit or half",
    ("that is the entire action set", 1),
    "",
    ("Combat is subtraction - the larger stack survives with the difference",
     0, RED),
    "",
    "I win by taking the enemy general. That is the only victory condition.",
    "",
    ("And the board is fully observable - which is what makes this course's "
     "algorithms apply at all.", 0, ACCENT),
], width=I(5.5), size=14, gap=4)
d.figure(s, "d01-board", I(6.2), I(1.72), I(6.5), I(4.9))

# ------------------------------------------------------------------ 3. compounding
s = d.slide("the game", "Which means it compounds")
d.bullets(s, [
    "Generals and castles produce one unit every two ticks",
    "Every tile I own produces one unit every fifty ticks",
    "",
    ("So one castle is worth exactly twenty-five plain tiles - a fact about the "
     "rules, not an opinion about strategy", 0, RED),
    "",
    "Territory produces army, army takes territory",
    "Ten tiles ahead at turn fifty is usually much further ahead by turn "
    "two hundred",
    "",
    ("The curves diverge and never come back together.", 0, ACCENT),
], width=I(4.6), size=14.5, gap=5)
d.figure(s, "d05-compounding", I(5.3), I(1.66), I(7.4), I(4.95))

# ------------------------------------------------------------------ 4. hard
s = d.slide("the game", "What makes it hard")
d.figure(s, "fig2-branching-factor", I(0.62), I(1.72), I(12.1), I(4.05))
d.bullets(s, [
    ("It is real time. Both players move together, so tempo is a first-class "
     "resource - I get half a second per decision.", 0, INK),
    ("b = 204, measured from a real position. Chess is 35. Depth is "
     "unaffordable.", 0, RED),
    ("And the reward is sparse: I win or lose 500 ticks later. Nothing tells me "
     "at tick 137 whether that move was good.", 0, INK),
], top=I(5.92), width=I(12.1), size=13.5, gap=3)

# ------------------------------------------------------------------ 5. why search
s = d.slide("why search", "Search is not the fallback here")
d.bullets(s, [
    "Search and CSP only - chapters three to six. No learning of any kind.",
    "",
    ("That is not a consolation prize.", 0, RED),
    ("the strongest non-learning bot on the public ladder is pure classical "
     "optimisation", 1),
    ("it held the top of that ladder until an RL agent on four H200s beat it", 1),
    "",
    "And the sub-problems here are genuinely NP-hard:",
    ("gathering army - a rooted prize-collecting Steiner tree", 1),
    ("spending turns - multiple-choice knapsack", 1),
    "",
    ("No analytic solution, combinatorial search space. Search is what "
     "you use.", 0, ACCENT),
], width=I(5.4), size=13.5, gap=3)
d.figure(s, "d09-nphard", I(6.3), I(1.66), I(6.4), I(4.95))

# ------------------------------------------------------------------ 6. the papers
s = d.slide("the papers", "One paper per topic in the course")
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
y = I(1.92)
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
    y = y + I(0.98)
    if num != "3.5":
        d._line(s, I(0.62), y - I(0.22), I(12.1), color=RGBColor(0xE4, 0xE4, 0xE4))

# ------------------------------------------------------------------ 7. ARA*
s = d.slide("3.1  ARA*", "An answer at every moment")
d.bullets(s, [
    "Plain A* gives me nothing until it finishes. If my half second runs out "
    "mid-search I have no answer at all, and the tick fires regardless.",
    "",
    "Weighted A* inflates the heuristic - greedier, and much faster",
], width=I(5.2), size=14, gap=4)
d.equation(s, r"f(n) \;=\; g(n) \;+\; \varepsilon\, h(n)",
           "s_wastar", I(0.62), I(3.66), I(0.44))
d.bullets(s, [
    ("And it comes with a theorem, not a hope:", 0, RED),
], top=I(4.32), width=I(5.2), size=14)
d.equation(s, r"\mathrm{cost}(\text{found}) \;\leq\; \varepsilon \cdot "
              r"\mathrm{cost}(\text{optimal})",
           "s_bound", I(0.90), I(4.72), I(0.44), color="#B4342C")
d.bullets(s, [
    "So I run it repeatedly with epsilon decreasing, and each run reuses the "
    "previous run's work",
    ("a rough answer almost immediately, better the longer I let it run, and at "
     "every moment I know exactly how far from optimal I am", 1),
], top=I(5.42), width=I(5.2), size=13.5, gap=3)
d.figure(s, "d11-anytime", I(6.2), I(1.70), I(6.5), I(4.9))

# ------------------------------------------------------------------ 8. RTAA*
s = d.slide("3.2  RTAA*  +  3.3  depression avoidance", "Learning while I act")
d.bullets(s, [
    "I cannot search to the goal, I must act now, and I will be back in a "
    "similar situation next step",
    "",
    "LRTA*, the algorithm in the book, updates essentially one state per move",
    ("RTAA* runs a bounded-depth A* and then updates EVERY state that search "
     "expanded - one subtraction each", 0, RED),
], width=I(5.3), size=14, gap=4)
d.equation(s, r"h[s] \;\leftarrow\; g[\bar{s}] \;+\; h[\bar{s}] \;-\; g[s]",
           "s_rtaa", I(0.62), I(4.02), I(0.44), color="#B4342C")
d.bullets(s, [
    "Same lookahead, same cost per move, vastly more heuristic learned",
    "",
    ("Depression avoidance then costs one extra bit per state: because updates "
     "only ever raise h, a raised value IS the detector for a region I should "
     "route around.", 0, ACCENT),
], top=I(4.72), width=I(5.3), size=13.5, gap=4)
d.figure(s, "d13-lrta-vs-rtaa", I(6.3), I(1.70), I(6.4), I(4.9))

# ------------------------------------------------------------------ 9. metareasoning
s = d.slide("3.5  metareasoning", "Should I be searching at all right now?")
d.bullets(s, [
    "The other four papers ask how to search well. This one asks a different "
    "question.",
    "",
    ("A computation that cannot change my answer has zero value, no matter how "
     "deep it is.", 0, RED),
    "",
    "The 2015 paper names two decisions: act now or think longer, and how far "
    "to commit before replanning",
    ("which are exactly the two lines RTAA* left undefined", 1),
    "",
    ("And this game hands me both for free - index 0 is PASS, and the move queue "
     "plays one queued move per tick. That is where I think my contribution "
     "sits.", 0, ACCENT),
], width=I(5.3), size=13.5, gap=4)
d.figure(s, "d20-mechanics", I(6.3), I(2.05), I(6.4), I(4.2))

# ------------------------------------------------------------------ 10. the test
s = d.slide("what I want to test", "Here the value of computation is measurable")
d.bullets(s, [
    "Will this computation change my move? Normally you have to guess.",
    ("here the evaluator hands me the full one-ply distribution in under a "
     "millisecond", 1),
    "",
    ("Measured over a full game - how many moves tie at the best one-ply "
     "score:", 0, INK),
    ("38% of ticks: almost nothing ties. One ply decided it - play now.", 1, RED),
    ("57%: most moves tie. One ply cannot separate them - spend the budget "
     "here.", 1, RED),
    "",
    "Bimodal, so a single threshold splits the two cases cleanly",
    "",
    ("The cost side is priced by the rules too. Both sides of the inequality "
     "are computable, which is unusual.", 0, ACCENT),
], width=I(5.5), size=13.5, gap=3)
d.figure(s, "fig4-value-of-computation", I(6.4), I(1.70), I(6.3), I(4.9))

out = ROOT / "shorten-search-only-generals-ai.pptx"
d.save(out)
print(f"{d.n} slides -> {out.name}")
