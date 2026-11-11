"""Final DAC-203 deck. Theme: Graphite Paper (custom, derived from theme-factory's
Modern Minimalist): charcoal/slate/light grey on white, Georgia headings,
Calibri body, Computer Modern equations; red and blue only for us and them."""
import pathlib

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

HERE = pathlib.Path(__file__).parent
FIG, EQ = HERE / "fig", HERE / "eq"
OUT = pathlib.Path(__file__).resolve().parents[2] / "presentation" / "Generals-AI-Final.pptx"
OUT.parent.mkdir(exist_ok=True)

INK, CHAR, SLATE, RULE, PANEL = "222222", "36454F", "708090", "D3D3D3", "F4F4F2"
US, THEM = "B03A2E", "2C5F8A"
HEAD, BODY = "Georgia", "Calibri"
FOOT = "DAC-203 · Search agents for Generals.io"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
N = [0]


def rgb(h):
    return RGBColor.from_string(h)


def box(s, x, y, w, h, fill, name=None):
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid()
    r.fill.fore_color.rgb = rgb(fill)
    r.line.fill.background()
    r.shadow.inherit = False
    if name:
        r.name = name
    return r


def text(s, x, y, w, h, paras, size=17, color=INK, font=BODY, bold=False, italic=False, align=PP_ALIGN.LEFT,
         after=10, anchor=MSO_ANCHOR.TOP, spacing=1.08):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = paras if isinstance(paras, list) else [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(after)
        p.line_spacing = spacing
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.name = o.get("font", font)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def picture(s, path, x, y, w=None, h=None):
    iw, ih = Image.open(path).size
    if w is not None and h is not None:
        k = min(w / iw, h / ih)
        pw, ph = iw * k, ih * k
        return s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    if w is not None:
        return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(w * ih / iw))
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(h * iw / ih), Inches(h))


def eq(s, name, x, y, h=0.42, maxw=11.6):
    path = EQ / f"{name}.png"
    iw, ih = Image.open(path).size
    # scale so a one-line equation is h inches tall; taller ones keep the same type size
    scale = h / 120.0
    pw, ph = iw * scale, ih * scale
    if pw > maxw:
        pw, ph = maxw, maxw * ih / iw
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(pw), Inches(ph))


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


def content(label, title, note=None):
    s = prs.slides.add_slide(BLANK)
    N[0] += 1
    text(s, 0.7, 0.45, 11.9, 0.3, label.upper(), size=10.5, color=SLATE, bold=True, after=0)
    text(s, 0.7, 0.72, 11.9, 0.9, title, size=28, color=CHAR, font=HEAD, after=0)
    text(s, 0.7, 7.0, 9, 0.25, FOOT, size=9, color=SLATE, after=0)
    text(s, 11.9, 7.0, 0.7, 0.25, str(N[0]), size=9, color=SLATE, align=PP_ALIGN.RIGHT, after=0)
    if note:
        notes(s, note)
    return s


def section(num, title, sub, aima, note=None):
    s = prs.slides.add_slide(BLANK)
    N[0] += 1
    text(s, 0.7, 2.35, 2.0, 1.2, str(num), size=66, color=RULE, font=HEAD, after=0)
    text(s, 2.2, 2.55, 10.4, 0.8, title, size=34, color=CHAR, font=HEAD, after=0)
    text(s, 2.22, 3.4, 10.0, 0.9, sub, size=16, color=SLATE, after=0)
    text(s, 2.22, 4.35, 10.0, 0.4, aima, size=12, color=SLATE, italic=True, after=0)
    text(s, 11.9, 7.0, 0.7, 0.25, str(N[0]), size=9, color=SLATE, align=PP_ALIGN.RIGHT, after=0)
    if note:
        notes(s, note)
    return s


def table(s, x, y, w, rows, widths, size=12.5, row_h=0.42, first_bold=True):
    shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shp.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    for j, wd in enumerate(widths):
        tbl.columns[j].width = Inches(wd)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.background()
            c.margin_left = Inches(0.06 if j == 0 else 0.1)
            c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = val
            r.font.name = BODY
            r.font.size = Pt(size - 1 if i == 0 else size)
            r.font.bold = i == 0 or (first_bold and j == 0)
            r.font.color.rgb = rgb(SLATE if i == 0 else (CHAR if j == 0 else INK))
            tcPr = c._tc.get_or_add_tcPr()
            for edge in ("lnL", "lnR", "lnT", "lnB"):
                old = tcPr.find(qn("a:" + edge))
                if old is not None:
                    tcPr.remove(old)
            # schema order: borders come before the cell fill
            for k, edge in enumerate(("lnL", "lnR", "lnT", "lnB")):
                top_rule = edge == "lnT" and i == 0
                bottom = edge == "lnB"
                ln = etree.Element(qn("a:" + edge), w="12700" if (top_rule or (bottom and i == 0)) else ("6350" if bottom else "0"))
                if bottom or top_rule:
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    etree.SubElement(sf, qn("a:srgbClr"), val=(CHAR if (i == 0 or top_rule) else RULE))
                else:
                    etree.SubElement(ln, qn("a:noFill"))
                tcPr.insert(k, ln)
    return shp


def bars(s, x, y, w, h, cats, vals, colors, maxv=1.0, fmt='0%', horizontal=False):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("v", vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED,
                            Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.name = BODY
    ch.font.size = Pt(11)
    ch.font.color.rgb = rgb(SLATE)
    va = ch.value_axis
    va.maximum_scale, va.minimum_scale = maxv, 0
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E6E6E3")
    va.format.line.fill.background()
    va.tick_labels.number_format = fmt
    va.tick_labels.number_format_is_linked = False
    va.tick_labels.font.size = Pt(10)
    ca = ch.category_axis
    ca.format.line.color.rgb = rgb(RULE)
    ca.tick_labels.font.size = Pt(10.5)
    plot = ch.plots[0]
    plot.gap_width = 70
    ser = plot.series[0]
    for k, c in enumerate(colors):
        pt = ser.points[k]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.font.size = Pt(10)
    dl.font.color.rgb = rgb(CHAR)
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    return gf


GREY = "B9BEC2"

# =========================================================================== #
# title, outline
# =========================================================================== #
s = prs.slides.add_slide(BLANK)
N[0] += 1
text(s, 0.9, 1.7, 11.5, 0.35, "DAC-203 ARTIFICIAL INTELLIGENCE · IIT ROORKEE · COURSE PROJECT", size=11, color=SLATE, bold=True)
text(s, 0.9, 2.15, 11.5, 2.0, ["Search agents for perfect-information Generals.io",
                               [("From five papers to v51", {"size": 24, "color": SLATE})]],
     size=40, color=CHAR, font=HEAD, after=6)
box(s, 0.92, 4.25, 1.2, 0.025, CHAR)
text(s, 0.9, 4.5, 11.5, 1.2, ["Fifty-two bots, what each one assumed, where it broke, and the model we ended with.",
                              "Built on Russell & Norvig, Artificial Intelligence: A Modern Approach (4th ed.)."],
     size=16, color=SLATE, after=4)
notes(s, "This is the final presentation for the DAC-203 project. Part I proposed five search papers for a fully "
         "observable, simultaneous-move version of Generals.io. This deck reports what happened when we built them, why "
         "most of them did not help, and how the final bot, v51, works. Every result quoted comes from games run in our "
         "simulator or from recorded games against people.")

s = content("Outline", "Nine parts", "The deck follows the order in which the bots were built. Each part ends with what "
            "the next part had to fix. Part 8 is the detailed walk through v51.")
parts = [("1", "The problem in AIMA's terms", "environment, game, branching"),
         ("2", "The five papers, as built", "ARA*, RTAA*, depression avoidance, ProbCut, metareasoning"),
         ("3", "Searching the joint move", "matrix games, ABCD, scripts"),
         ("4", "Three learned components", "why each one did not pay"),
         ("5", "Search from the rules", "the invariant, rollouts, plans"),
         ("6", "The collection race", "D(X,T), reserve, castles"),
         ("7", "Learning the opponent", "Bayes, bandits, safe exploitation"),
         ("8", "v51 in depth", "fast defence and the opponent classifier"),
         ("9", "What the course material bought us", "chapter by chapter")]
for i, (n_, t_, d_) in enumerate(parts):
    col, row = divmod(i, 5)
    x, y = 0.7 + col * 6.2, 1.85 + row * 0.98
    text(s, x, y, 0.6, 0.5, n_, size=24, color=RULE, font=HEAD)
    text(s, x + 0.65, y + 0.02, 5.3, 0.4, t_, size=17, color=CHAR, font=HEAD, after=2)
    text(s, x + 0.65, y + 0.4, 5.3, 0.35, d_, size=12.5, color=SLATE)

# =========================================================================== #
# 1 problem
# =========================================================================== #
section(1, "The problem in AIMA's terms", "A two-player, zero-sum game on a 23 x 23 grid, both players moving at once.",
        "AIMA ch. 2 (task environments), ch. 6 (adversarial search), ch. 18 (multiagent decisions)")

s = content("1 · The problem", "Task environment", "AIMA chapter 2 classifies environments along seven properties. "
            "Ours is unusual in one respect: it is fully observable, which removes the fog-of-war machinery that most "
            "Generals.io bots carry, but it keeps simultaneous moves, which is why plain minimax does not apply directly.")
table(s, 0.7, 1.85, 7.4, [["Property (AIMA ch. 2)", "Our game"],
                         ["Observable", "Fully: every tile, army and castle is visible"],
                         ["Agents", "Two, competitive, zero-sum"],
                         ["Deterministic", "Yes, after the map is drawn"],
                         ["Episodic / sequential", "Sequential, 500 to 1,000 ticks"],
                         ["Static / dynamic", "Dynamic: one decision per tick, 0.5 s"],
                         ["Discrete", "Yes: 529 cells, 9 actions per cell"],
                         ["Known rules", "Yes; we wrote an exact simulator"]], [2.6, 4.8], size=13.5, row_h=0.5)
text(s, 8.7, 1.95, 4.0, 4.5, ["Production: the general and each castle add 1 every 2 ticks; every owned tile adds 1 every "
                              "50 ticks.",
                              "A move sends all but one unit, or half, to a neighbour. Combat subtracts.",
                              "The game ends when a general is taken, or at the tick cap on land, then army."],
     size=15, color=INK, after=12)

s = content("1 · The problem", "The game as a formal object", "We model the game as a deterministic simultaneous-move "
            "game. The joint action is what makes it hard: the state changes only after both players have committed.")
eq(s, "game", 0.7, 1.9, h=0.5)
eq(s, "branch", 0.7, 2.9, h=0.5)
text(s, 0.7, 4.05, 11.6, 2.6, ["Measured from a real mid-game position: b = 204 for one player, so about 41,600 joint "
                               "actions per tick. Chess has about 35.",
                               "The reward is terminal and sparse: +1 or −1 after hundreds of ticks. Everything in "
                               "between is judged by an evaluation function, so most of the work is in that function.",
                               "Simultaneous moves mean a node is a matrix game, not a max node or a min node "
                               "(AIMA ch. 18). Plain minimax assumes we see the reply first."],
     size=16, after=12)

# =========================================================================== #
# 2 five papers
# =========================================================================== #
section(2, "The five papers, as built", "Part I's plan: one paper per search chapter. v1 to v21 test them.",
        "AIMA ch. 3 (informed search), ch. 4 (online search), ch. 6 (pruning), plus Russell & Wefald on metareasoning",
        "Part I chose five papers. This part shows the formula behind each, the bot that implemented it, the result, "
        "and the assumption that did not hold in our game.")

s = content("2 · The five papers", "ARA*: an answer at every moment",
            "Weighted A* inflates the heuristic by epsilon and returns a path at most epsilon times optimal. ARA* "
            "repeats it with decreasing epsilon and reuses the previous search. We used it to route stacks in v1. It "
            "found routes quickly and correctly. Routes were never what lost games: v33 to v35 replaced it with "
            "time-dependent A* that also models garrisons and growth, cut node expansions from 31.6 to 22.6, and still "
            "won only 3 of 6 games against the direct rusher.")
text(s, 0.7, 1.75, 11.9, 0.4, "Likhachev, Gordon & Thrun, NIPS 2003", size=12.5, color=SLATE, italic=True)
eq(s, "wastar", 0.7, 2.25, h=0.5)
table(s, 0.7, 3.3, 11.9, [["Built in", "What happened", "Why it did not decide games"],
                         ["v1; v33–v35", "Routes found fast; time-aware A* cut expansions 31.6 → 22.6",
                          "Pathfinding was solved early; losses came from timing and home safety"],
                         ["Rusher test", "v33, v34, v35: 3 of 6 wins each", "A better route for the wrong plan is still the wrong plan"]],
      [2.2, 4.9, 4.8], size=13.5, row_h=0.62)

s = content("2 · The five papers", "RTAA* and depression avoidance: not built",
            "RTAA* updates the heuristic of every state a bounded A* expanded, so an agent that must act before "
            "finishing its search learns quickly. Depression avoidance then marks states whose heuristic was raised and "
            "routes around them. Both papers solve the problem of an agent that does not know the true distances. In "
            "our game the map is static and fully visible, and a breadth-first search over 529 cells takes well under "
            "a millisecond, so every bot uses exact distances. There was no heuristic left to learn.")
text(s, 0.7, 1.75, 11.9, 0.4, "Koenig & Likhachev, AAMAS 2006 · Hernández & Baier, AAAI 2012", size=12.5, color=SLATE, italic=True)
eq(s, "rtaa", 0.7, 2.3, h=0.5)
eq(s, "depr", 0.7, 3.15, h=0.5)
text(s, 0.7, 4.3, 11.9, 2.2, ["Assumption in the papers: true distances are unknown, so h must be learned while acting.",
                              "Our game: distances are known exactly and recomputed from the board every tick. "
                              "The setting these algorithms target does not arise."], size=16, after=12)

s = content("2 · The five papers", "ProbCut: excellent statistics, no extra wins",
            "ProbCut fits the deep search value as a linear function of a shallow one and skips the deep search when the "
            "shallow value is far enough outside the window. Our fit was very good. Deeper search bought with it did not "
            "change results: v16 and v17 split their matches with v11 and ran much slower, and the learned version v20 "
            "lowered prediction error and then lost 7 of 8 games when paired with castle-first play.")
text(s, 0.7, 1.75, 11.9, 0.4, "Buro, ProbCut (ICCA Journal 1995) · AIMA ch. 6, forward pruning", size=12.5, color=SLATE, italic=True)
eq(s, "probcut", 0.7, 2.3, h=0.5)
table(s, 0.7, 3.55, 7.0, [["Measurement", "Value"],
                         ["Fit on producer-wait positions", "a = 0.955, b = 2.04, σ = 2.77"],
                         ["Correlation shallow vs deep", "r = 0.974 (105 positions)"],
                         ["v17 (4-ply) vs v11, seed 68000", "1–1, much slower"],
                         ["v20 learned correction, held-out error", "2.02 → 1.79"],
                         ["v20 with castle-first play vs v11", "lost 7 of 8"]], [3.9, 3.1], size=13, row_h=0.47)
text(s, 8.2, 3.6, 4.4, 3.0, ["Why: pruning made the tactical tree deeper. The games were decided by economy and by "
                             "when the general was left thin, over hundreds of ticks. A four-ply tree cannot see either."],
     size=15, color=INK)

s = content("2 · The five papers", "Metareasoning and the move queue",
            "Russell and Wefald value a computation by how much it can change the chosen action, minus its cost. Part I "
            "proposed using the queue of safe moves as thinking time and pre-computing the next tick. v18 did both: it "
            "queued up to six safe moves and searched the predicted endpoint in a background worker. It lost both games "
            "of its seed. Forecasts were invalidated whenever stacks met, which is exactly when decisions matter, and the "
            "bot was never short of time in the first place. Later bots decide in about 10 ms.")
text(s, 0.7, 1.75, 11.9, 0.4, "Russell & Wefald, UAI 1990 · O'Ceallaigh & Ruml, SoCS 2015 · AIMA ch. 16, value of information",
     size=12.5, color=SLATE, italic=True)
eq(s, "voc", 0.7, 2.3, h=0.5)
text(s, 0.7, 3.4, 11.9, 3.0, ["The value of computation is high only where the best and second-best moves are close. We "
                              "measured that 57% of ticks had many tied moves at one ply, so there was room for the idea.",
                              "The problem was elsewhere: extra thinking time improved decisions inside a model that was "
                              "missing the thing that lost games. A bot that decides in 10 ms has nothing to gain from "
                              "buying more time."], size=16, after=12)

s = content("2 · The five papers", "Why the paper bots were weak",
            "This slide is the summary the rest of the deck answers. Each paper solved its own problem well. None of "
            "those problems was ours. Our first nineteen bots were search engines around an evaluation that did not "
            "know when the general was about to fall.")
table(s, 0.7, 1.85, 11.9, [["Paper", "What it assumes", "What our game is like", "Outcome"],
                          ["ARA*", "Finding a good path is the bottleneck", "Paths are easy; plans and timing are not", "Kept for routing"],
                          ["RTAA*", "Distances are unknown and learned online", "Distances are exact and cheap", "Not built"],
                          ["Depression avoidance", "Heuristic basins trap the agent", "No learned heuristic, no basins", "Not built"],
                          ["ProbCut", "Depth wins games", "Economy and home safety win games", "No gain, slower"],
                          ["Metareasoning", "Time is scarce", "10 ms per decision was enough", "No gain"]],
      [2.3, 3.4, 3.8, 2.4], size=13.5, row_h=0.62)
text(s, 0.7, 5.8, 11.9, 0.9, "Common failure: all five improve search. The early bots searched a model that did not "
     "represent the event that lost games, a general left too thin when an attack arrived.", size=15, color=CHAR, italic=True)

# =========================================================================== #
# 3 joint move
# =========================================================================== #
section(3, "Searching the joint move", "v3 to v35: matrix games, serialization, scripts, rollouts.",
        "AIMA ch. 6 (alpha-beta, MCTS), ch. 18 (normal-form games, maximin, Nash)")

s = content("3 · Joint move", "A node is a matrix game",
            "With simultaneous moves, the value of a node is the value of a zero-sum matrix game between our options and "
            "theirs. Von Neumann's theorem says mixed strategies achieve the maximin value. v3 solved a 3 x 3 matrix at "
            "every state; it was slow and weaker than plain search. v4 solved it only when candidate moves could collide, "
            "and v11 added a launch-intercept option. v11 stayed the final bot until v37.")
eq(s, "maximin", 0.7, 1.95, h=0.5)
table(s, 0.7, 3.1, 11.9, [["Bot", "Idea", "Result"],
                         ["v3", "Matrix game at every state", "Slower and weaker than plain search"],
                         ["v4, v5", "Matrix only when moves can collide; short safe queues", "Became the baseline"],
                         ["v10", "Two-ply matrix", "Slow; lost its 4-game match against v9"],
                         ["v11", "Adds capture of the enemy's launch source", "Final bot until v37"]],
      [1.5, 5.6, 4.8], size=13.5, row_h=0.55)
text(s, 0.7, 6.05, 11.9, 0.6, "Lesson: two stacks interact on a small fraction of ticks. The matrix matters at contact "
     "and nowhere else.", size=15, color=CHAR, italic=True)

s = content("3 · Joint move", "Serializing simultaneous moves: ABCD",
            "ABCD (Churchill, Saffidine and Buro) turns a simultaneous game into alternating alpha-beta by deciding who "
            "moves first at each step. Letting us move first gives the pessimistic maximin bound; letting them move first "
            "gives the optimistic one. Alternating the order evens out the bias. This is the search Puppet Search runs "
            "over scripts, and the one v50 implements.")
text(s, 0.7, 1.75, 11.9, 0.4, "Churchill, Saffidine & Buro, AIIDE 2012 · AIMA ch. 6, alpha-beta", size=12.5, color=SLATE, italic=True)
eq(s, "abcd", 0.7, 2.3, h=0.5)
text(s, 0.7, 3.45, 11.9, 3.0, ["The serialized tree is ordinary alpha-beta, so its pruning applies unchanged.",
                               "The cost is the bias: the first mover sees nothing, the second mover sees everything. "
                               "With an alternating policy the two errors partly cancel across a search.",
                               "We used it at depth 2 over plans (v50). Depth 4 with 40-tick rollouts was too slow "
                               "in Python: about 576 rollouts per decision before pruning."], size=16, after=12)

s = content("3 · Joint move", "Scripts instead of moves: Puppet Search",
            "Puppet Search keeps a small library of scripts with choice points and searches over those choices. The "
            "branching factor drops from thousands of moves to a handful of plans. v22 to v26 simulated hold, castle, "
            "charge and expand scripts for 96 exact ticks per pair. v23 split 6-6 with v11. v24's long waits oscillated "
            "because the target was re-chosen at every replan. The idea was right; it came back in v37 behind a safety "
            "layer.")
text(s, 0.7, 1.75, 11.9, 0.4, "Barriga, Stanescu & Buro, AIIDE 2015, Algorithm 1", size=12.5, color=SLATE, italic=True)
table(s, 0.7, 2.35, 11.9, [["Bot", "Script search", "Result"],
                          ["v22", "Hold / castle / charge, 96 exact ticks", "First temporal abstraction"],
                          ["v23", "Adds expansion; learned macro ProbCut (error 33.3 vs 36.5)", "6–6 vs v11 over 12 games"],
                          ["v24, v25", "Long producer waits; several castle waits", "Waits oscillated between targets"],
                          ["v33, v34", "Scenario rollouts, 112 and 224 ticks", "1 win, 1 draw, 2 losses each"]],
      [1.5, 6.2, 4.2], size=13.5, row_h=0.55)
text(s, 0.7, 5.35, 11.9, 1.2, "Missing piece: nothing in these searches guaranteed that the general survived while a "
     "plan ran. A plan that wins in rollouts against four scripts can still empty the general against a fifth.",
     size=15, color=CHAR, italic=True)

s = content("3 · Joint move", "More search did not move the rusher result",
            "Win rate against the direct general rusher, the opponent that kept beating us. Each grey bar added depth, a "
            "learned cut, a better route or longer rollouts. The red bar is v36, which changed what is checked rather "
            "than how deep. The v36 bar comes from a larger held-out suite.")
bars(s, 0.6, 1.7, 8.3, 5.1, ["v27\nlive matrix", "v32\nlearned prune", "v32\nbuckets only", "v33\ntime A*",
                             "v34\nA* ε=1.5", "v35\npayback A*", "v33\nrollouts", "v34\nrollouts", "v36\ninvariant"],
     [4 / 6, 5 / 6, 5 / 6, .5, .5, .5, .25, .25, 1.0], [GREY] * 8 + [US])
text(s, 9.2, 1.9, 3.6, 4.6, ["v29's second reply ply took 0.56 s on one node and still lost.",
                             "v30 lost both rusher seats; v31 took 124 s for one game and lost.",
                             "v36: 20 of 20 against the rusher and 20 of 20 against the castle rusher."],
     size=14.5, after=12)

s = content("3 · Joint move", "Diagnosis: the arithmetic of a rush",
            "This one picture explains why depth did not help. An attacker walking d steps leaves one unit per tile, so "
            "it loses about d units. The defending general gains about d over 2 in the same time. Two players who keep "
            "army at home cannot rush each other. A general falls only when it has been drained, or when several stacks "
            "arrive in sequence. Our early bots lost exactly that way, and none of their evaluations measured it.")
eq(s, "walk", 0.7, 1.85, h=0.5)
picture(s, FIG / "invariant.png", 0.7, 2.75, w=8.6)
text(s, 9.6, 2.9, 3.1, 3.6, ["Early wins against the rusher were land tiebreaks at short caps, not captures.",
                             "After v36 every result is reported with seats swapped and captures counted separately."],
     size=14.5, after=12)

# =========================================================================== #
# 4 learning
# =========================================================================== #
section(4, "Three learned components", "Each one learned something real. None of them won more games.",
        "AIMA ch. 19 (learning from examples), ch. 21 (deep learning)")

s = content("4 · Learning", "Why the learned parts did not pay",
            "Three times we trained a model to make search cheaper or better. Each model was accurate on its own test. "
            "None changed match results, for reasons that are worth stating precisely. The GPU was used for the CNN "
            "ranker; the other two were small fits.")
table(s, 0.7, 1.85, 11.9, [["Bot", "What was learned", "Accuracy", "Match result", "Reason"],
                          ["v20", "Nonlinear shallow → deep correction (ProbCut)", "Error 2.02 → 1.79", "Lost 7 of 8 vs v11",
                           "Better pruning of a tree that missed the cause of losses"],
                          ["v32", "MLP that rejects optional branches", "92.3% precision", "5 of 6 vs rusher",
                           "Bucket-only ablation also 5 of 6: the gain came from the buckets"],
                          ["v40", "CNN plan ranker, 8,071 regret-matched positions", "Top-2 regret below random", "4–16 vs v37",
                           "Noisy teacher; ranking errors compound over hundreds of ticks"]],
      [0.8, 3.3, 2.0, 1.9, 3.9], size=12.5, row_h=0.85)
text(s, 0.7, 5.6, 11.9, 1.0, "AIMA's warning about learned evaluation applies: a model is only as useful as the target it "
     "imitates. All three targets were outputs of searches that did not model home safety.", size=15, color=CHAR, italic=True)

# =========================================================================== #
# 5 rules
# =========================================================================== #
section(5, "Search from the rules", "v36 and v37: an exact simulator, one invariant, then plans on top.",
        "AIMA ch. 4 (AND-OR search), ch. 16 (decisions under uncertainty), ch. 19 (logistic regression)")

s = content("5 · From the rules", "An exact simulator first",
            "sim36 is a pure Python copy of the game step that updates only the cells a tick changes. It was checked "
            "against the JAX environment on thousands of random joint actions, and it reproduced all 780 recorded moves "
            "of two human games. A v36 decision with its safety rollouts costs 1 to 2 ms. Twenty seat-swapped games at "
            "cap 500 take about 20 seconds on 12 cores, which is what made honest testing possible from here on.")
table(s, 0.7, 1.85, 7.0, [["Measurement", "Value"], ["Agreement with JAX step", "identical boards on random actions"],
                         ["Recorded human moves reproduced", "780 of 780"], ["v36 decision time", "1–2 ms"],
                         ["20 games, cap 500, 12 cores", "about 20 s"]], [3.6, 3.4], size=14, row_h=0.55)
text(s, 8.2, 1.95, 4.4, 4.0, ["Without this, every rollout in the rest of the deck would have been too slow to run "
                              "inside a decision.",
                              "With it, a decision can afford a few hundred simulated ticks."], size=15, after=12)

s = content("5 · From the rules", "The invariant: the general must hold alone",
            "v36 is built on one rule. For every hostile all-in walk, our general alone, plus what it grows while the "
            "walk is under way, must beat what arrives, with a pad that grows with the biggest enemy stack. Moves that do "
            "not lower this margin are always allowed, so expansion never freezes. Any move that does lower it goes to an "
            "exact rollout check.")
eq(s, "invariant", 0.7, 1.9, h=0.5)
table(s, 0.7, 3.0, 6.4, [["v36 vs", "Result"], ["Direct rusher", "20 / 20"], ["Castle rusher", "20 / 20"],
                        ["v11 (old final bot)", "12 / 12"], ["Expander", "16 / 20, losses are tiebreaks"]],
      [3.0, 3.4], size=14, row_h=0.55)
text(s, 7.6, 3.05, 5.0, 3.5, ["Our own stacks are not walls: an enemy walk through our land assumes each tile holds at "
                              "most 3, because stacks can step aside.",
                              "The general never relies on being refilled. Otherwise it can step out, look safe because "
                              "a stack will come back, and oscillate."], size=14.5, after=12)

s = content("5 · From the rules", "Rollout checks as AND-OR search",
            "AIMA chapter 4 handles nondeterministic actions with AND-OR trees: our choice is an OR node, every possible "
            "outcome is an AND node, and a plan is valid only if it succeeds under all outcomes. Our two checks have "
            "exactly this shape. survives() plays the candidate move and then every plausible enemy launch against a "
            "fixed defender. verify() plays a planned attack against four scripted replies and accepts it only if it wins "
            "against all four.")
table(s, 0.7, 1.85, 11.9, [["Check", "OR node (ours)", "AND node (theirs)", "Accept if"],
                          ["survives(move)", "the candidate move, then the Defender script",
                           "top 3 enemy stacks × shortest and cheapest route, on the old and new board", "no rollout loses the general"],
                          ["verify(plan)", "the attack plan", "Rusher, CastleRusher, Pass, turtle Econ", "it wins against all four within d + 4 ticks"]],
      [2.0, 3.2, 4.3, 2.4], size=13, row_h=0.8)
text(s, 0.7, 4.6, 11.9, 1.8, ["The checks are deliberately incomplete: the enemy policy set is small. That is the price of "
                              "running them inside a 10 ms decision. Section 6 shows what this missed."],
     size=15.5, color=INK)

s = content("5 · From the rules", "Plans, not moves (v37)",
            "v37 brings back Puppet Search's idea behind the v36 safety layer. Each decision chooses among about six macro "
            "plans. Each plan is rolled out 24, 40 or 60 ticks against four opponent scripts, and the final board is "
            "scored by a nine-feature logistic model. The plan score blends the expected value under our belief about "
            "the opponent with the worst case. This is the Hurwicz criterion with alpha one half: neither pure "
            "expectation, which trusts the belief, nor pure maximin, which ignores it.")
eq(s, "hurwicz", 0.7, 1.9, h=0.5)
eq(s, "logistic", 0.7, 2.85, h=0.5)
table(s, 0.7, 3.85, 6.3, [["Measurement", "Value"], ["Plans after merging identical rollouts", "6.1 → 4.6"],
                         ["Value model log-loss, opening", "0.473 (land-only 0.522)"],
                         ["Value model log-loss, mid-game", "0.372 (land-only 0.359)"],
                         ["Expander", "19 / 20 (v36: 16 / 20)"], ["vs v36 / vs v11", "12–8 / 20–0"]],
      [3.4, 2.9], size=13, row_h=0.47)
text(s, 7.5, 3.9, 5.1, 2.8, ["The value model did not beat a land-only baseline in the mid-game. We kept it because it "
                             "helped in the opening and the end, and said so.",
                             "Phases come from the board, not the clock: opening while no enemy walk reaches us and "
                             "35% of land is neutral; end when the cap is near or one side leads 2 to 1."],
     size=14, after=10)

# =========================================================================== #
# 6 collection race
# =========================================================================== #
section(6, "The collection race", "v42 and v44: what people did to v37, and the margin that sees it.",
        "Prize-collecting Steiner tree and knapsack (NP-hard), greedy approximation; AIMA ch. 5, constraints")

s = content("6 · Collection race", "How people beat v37",
            "A recorded 887-tick game. The human spread out, took four castles while v37 took none, kept the general "
            "light, then gathered everything into one stack in the last stretch and broke a general that held about 90% "
            "of v37's army. v37's threat model looked at one walking stack at a time. A spread army never looked "
            "dangerous until it had been collected. Across recorded games longer than 150 ticks, humans won 5, v37 won 6, "
            "and 2 were unfinished.")
picture(s, FIG / "human.png", 0.6, 1.7, w=7.4)
text(s, 8.4, 1.9, 4.3, 4.6, ["Threat model of v37: the strongest single walk.",
                             "Human strategy: many walks, collected in sequence, each one cheap.",
                             "One move per tick means a collection takes as many ticks as it has walks, which gives the "
                             "defender time, if the defender is counting."], size=14.5, after=12)

s = content("6 · Collection race", "D(X, T): what a side can deliver in T moves",
            "We count how much army a side can walk onto a cell X using walks whose lengths add up to at most T, because "
            "each tick allows one move. Choosing the walks is a knapsack over walks, and choosing the gathering tree is a "
            "rooted prize-collecting Steiner tree, which is how the strongest non-learning Generals.io bot, Human.exe, "
            "gathers army (Straka et al. 2026). Both are NP-hard. We use a greedy approximation: biggest walk first, each "
            "tile counted once.")
eq(s, "dxt", 0.7, 1.85, h=0.5)
picture(s, FIG / "collect.png", 0.6, 3.0, w=5.3)
text(s, 6.7, 2.85, 6.0, 3.8, ["One walk at a time: the biggest arrives with 21 against a general of 40. Safe.",
                              "Three walks in sequence, 25 moves: 21 + 16 + 15 = 52 against 40 + 12 growth. Not safe.",
                              "Greedy is not optimal, but it errs toward the attacker for threats and toward "
                              "caution for our own walks home."], size=15, after=12)

s = content("6 · Collection race", "The race margin",
            "For every horizon T we compare our general plus growth plus what our own stacks can bring home, against what "
            "the enemy can collect onto it. Our walks get a lead of 2 plus T over 10 ticks, because defence must start "
            "early. The margin is the minimum over T, and every move must keep it above the pad. The chart is a real "
            "position at tick 300: the two curves meet at T = 52, so the bot defended.")
eq(s, "race", 0.7, 1.85, h=0.5)
picture(s, FIG / "race.png", 0.6, 2.7, w=6.6)
text(s, 7.6, 2.85, 5.1, 3.8, ["T ranges over every enemy walk length and every sum of two, up to 60.",
                              "Enemy walks treat our tiles as holding at most 3. Our walks home count only through our "
                              "own land, since a walk through theirs can be cut.",
                              "v42 vs v37: 19–1 on the real harness. Promoted."], size=14.5, after=12)

s = content("6 · Collection race", "Reserve and castles (v44)",
            "A person beat v42 by making it spend its general on castles and a strike, then walking their own general out "
            "and collecting 84 into 137. v44 adds three rules. Tiles are counted once in D. A move out of the general "
            "must leave a reserve against every enemy walk, now and after 30 and 60 ticks of their growth. A castle is "
            "taken only if it pays back before the cap and the race margin holds on the board projected to the end of "
            "the walk.")
eq(s, "reserve", 0.7, 1.85, h=0.5)
eq(s, "payback", 0.7, 2.75, h=0.5)
table(s, 0.7, 3.75, 6.6, [["v44 vs", "Result"], ["v42", "13–7"], ["v37", "19–1"],
                         ["Five scripted bots", "82 / 100 (v42: 75)"]], [3.4, 3.2], size=14, row_h=0.55)
text(s, 7.8, 3.8, 4.8, 2.8, ["A castle produces as much as 25 plain tiles, so castles decide long games.",
                             "The same walk that buys a castle drains the general. The projection to the end of the "
                             "walk is what makes both true at once."], size=14.5, after=12)

# =========================================================================== #
# 7 opponent
# =========================================================================== #
section(7, "Learning the opponent", "v45 to v50: belief over types, gated behaviour, safe exploitation, bandits.",
        "AIMA ch. 12 (Bayes' rule), ch. 14 (filtering), ch. 17 (bandits), ch. 18 (game theory)")

s = content("7 · Opponent", "A posterior over opponent types",
            "v46 keeps a belief over four opponent scripts. Every tick each script predicts the opponent's last move; "
            "scripts that predicted it gain weight. The forgetting factor 0.95 makes this a filter rather than a fixed "
            "estimate, so the belief can follow an opponent that changes style, and the additive floor keeps every type "
            "possible. This follows the Bayesian opening prediction of Synnaeve and Bessière with a much simpler "
            "likelihood.")
eq(s, "posterior", 0.7, 1.85, h=0.5)
picture(s, FIG / "posterior46.png", 0.6, 2.75, w=7.4)
text(s, 8.4, 2.9, 4.3, 3.6, ["Against a bot that expands to tick 120 and then sweeps its land into one charge: the "
                             "belief reads expander for the whole expansion and moves to the two rusher types during "
                             "the sweep.",
                             "AIMA ch. 14 calls this filtering: predict, observe, update."], size=14, after=12)

s = content("7 · Opponent", "Gates, safe exploitation, bandits",
            "The belief switches two gates. The castle gate is an income race: castles are bought while the projected lead "
            "at the cap is not yet safe. The threat gate opens on observed attacks or when the rush types dominate the "
            "belief, which raises the reserve from 0.4 to 0.6 and enables counter-strikes. Ganzfried and Sandholm's safe "
            "exploitation says to deviate toward the opponent only by as much as you have already gained; v50 implements "
            "that rule exactly. Exp3 chooses parameter settings across games.")
eq(s, "income", 0.7, 1.85, h=0.5)
eq(s, "ses", 0.7, 2.75, h=0.5)
eq(s, "exp3", 0.7, 3.65, h=0.5)
text(s, 0.7, 4.75, 11.9, 1.9, ["The gates can make the bot greedier or more careful. They cannot switch off the safety "
                               "layer, so a wrong belief costs efficiency and never the general.",
                               "v46 vs v37 17–3, vs v42 14–6, vs v44 11–9; 81 / 100 against five scripted bots. "
                               "Promoted, and hardest of our bots for people to beat at the time."], size=15, after=10)

s = content("7 · Opponent", "v50: the papers' actual algorithms",
            "v46 cited its papers for ideas. v50 implemented their algorithms and switched each on alone against v46, "
            "20 games each. Results at this size are noisy; the full v50 played 60 games. Human.exe's min-cost-flow "
            "expansion was clearly worse than v46's greedy one-tile-per-move expansion, even after three fixes, so it is "
            "off by default.")
table(s, 0.7, 1.8, 11.9, [["Algorithm (paper)", "Alone vs v46"],
                         ["Rooted prize-collecting Steiner tree gather, bisection on λ (Straka et al. 2026)", "13–7"],
                         ["Min-cost flow expansion + multiple-choice knapsack (Straka et al. 2026)", "5–15"],
                         ["Local matrix game solved by regret matching (Lisý et al. 2013)", "11–9"],
                         ["Puppet ABCD + SES (Barriga 2015; Ganzfried & Sandholm 2015)", "10–10"],
                         ["Bayesian timing + kNN typing (Synnaeve & Bessière 2011; Weber & Mateas 2009)", "13–7"],
                         ["Castle build order by branch and bound (Churchill & Buro 2011)", "9–11"],
                         ["Nash arm over the meta-game (Tavares et al. 2016)", "10–10"],
                         ["Full v50, flow expansion off, 60 games", "35–25"]], [9.6, 2.3], size=13, row_h=0.5)

# =========================================================================== #
# 8 v51
# =========================================================================== #
section(8, "v51 in depth", "v46 plus a committed fast defence and a learned 10-type opponent classifier.",
        "AIMA ch. 4 (search), ch. 14 (filtering), ch. 21 (neural networks); Straka et al. 2026 (gathering)",
        "This part explains the final bot completely: the loss that motivated it, the three formulas of the fast defence, "
        "the classifier, one full tick, and the results.")

s = content("8 · v51", "Architecture",
            "v51 adds one layer to v46. Read the stack from the bottom. Each layer proposes moves and the layers below can "
            "refuse them. The new layer sits on top because it pre-empts the planner when an attack is coming, but its "
            "gathering moves only ever move army toward our own general, so they cannot weaken the safety margin.")
picture(s, FIG / "layers.png", 0.6, 1.7, w=7.0)
text(s, 8.1, 1.9, 4.6, 4.6, ["New in v51: an approach detector, a defence-need estimate, a Steiner-tree gather sized "
                             "to the attacker's arrival, and a merge rule for stacks next to the general.",
                             "Also new: a neural opponent classifier over 10 types, trained on the Mac GPU.",
                             "Everything below the top layer is v46 unchanged."], size=14.5, after=12)

s = content("8 · v51", "The loss that motivated it",
            "The game that v46 lost against a bot that plays like our human testers: expand until tick 200, sweep into one stack, "
            "charge. A 103-army stack walked from 22 tiles away to the general over about 20 ticks. v46 kept expanding "
            "with two-army tiles, defended for single ticks whenever its margin dipped, and only started gathering at "
            "distance 4, one stack per tick. It had 400 army on the board and lost its 77-army general to 82. Against "
            "this bot v46 lost 8 of 20 games by capture.")
picture(s, FIG / "trace_pair.png", 1.2, 1.7, w=10.6)
text(s, 0.7, 6.5, 11.9, 0.4, "Same map, same seat, same opponent. Left: v46. Right: v51, whose general holds 124 when the "
     "attacker arrives with 90.", size=14, color=SLATE, italic=True)

s = content("8 · v51", "Step 1: detect the approach",
            "We follow each large enemy stack from tick to tick and watch its distance to our general. A stack is "
            "incoming when it has closed in by two or more tiles within five ticks and is big enough to matter. The "
            "estimated time of arrival is its current distance. Against collector types, which the classifier "
            "recognises, the size threshold drops so that a gathered stack is noticed earlier.")
eq(s, "approach", 0.7, 1.95, h=0.5)
text(s, 0.7, 3.1, 11.9, 3.4, ["Tracking: a stack keeps its history if it is on the same cell or a neighbour of last "
                              "tick's cell.",
                              "Commitment: once incoming, the state lasts six ticks past the last approaching "
                              "observation. v46's failure was the opposite, deciding afresh every tick and drifting back "
                              "to expansion.",
                              "Threshold: 15 army or 20% of our general; 12 or 15% against collectors."],
     size=16, after=12)

s = content("8 · v51", "Step 2: how short will we be?",
            "Given the arrival time, we ask the race question for horizons up to ETA plus 5: what can they collect onto "
            "our general, minus what the general holds and grows. A margin of 15% of the attacker covers estimation "
            "error. If the need is positive, defence takes every move until it is not.")
eq(s, "need", 0.7, 1.95, h=0.5)
text(s, 0.7, 3.1, 11.9, 3.4, ["D_them is the same collection estimate as in section 6, with each tile counted once.",
                              "The need counts only the general. Stacks parked next to it do not count, because the "
                              "attacker can land on the general before they move in.",
                              "The rule that follows: no stack waits beside the general."],
     size=16, after=12)

s = content("8 · v51", "Step 3: gather into the general with a Steiner tree",
            "We need the most army moved into the general within ETA minus 2 moves. That is a rooted prize-collecting "
            "Steiner tree: each own tile is a prize worth its army minus one, each tree edge is one move. We solve it "
            "greedily for a cost per move lambda, then bisect on lambda so the tree fits the move budget, as Human.exe "
            "does. Moves are played leaf-first so army accumulates on the way in.")
eq(s, "pcst", 0.7, 1.85, h=0.48)
picture(s, FIG / "pcst.png", 0.6, 2.65, w=6.2)
text(s, 7.3, 2.8, 5.4, 3.8, ["Greedy step: from the current tree, run one BFS over our land and add the path whose "
                             "collected army minus lambda times its length is largest.",
                             "Bisection: 7 steps between 0 and the largest prize.",
                             "Merge rule: any stack of 10 or more next to the general steps in at once; within 6 ticks of "
                             "impact, any stack of 2 or more."], size=14, after=12)

s = content("8 · v51", "Opponent classifier: features",
            "The classifier sees only the board, from our side. Fourteen numbers per tick describe the opponent: size, "
            "how concentrated its army is, where its biggest stack is and whether it is coming closer, how fast its land "
            "and top stack grow, and how often its general has been drained. They are cheap to compute and do not depend "
            "on knowing the opponent's moves.")
table(s, 0.7, 1.8, 11.9, [["#", "Feature", "#", "Feature"],
                         ["1", "tick / 500", "8", "change in that distance over 5 ticks"],
                         ["2", "land / 100", "9", "land gained over 20 ticks"],
                         ["3", "total army / 300", "10", "top stack growth over 5 ticks"],
                         ["4", "share of army in their general", "11", "cells with 5 or more army"],
                         ["5", "share of army in their top stack", "12", "times their general was drained"],
                         ["6", "castles held / 3", "13", "army gained over 20 ticks"],
                         ["7", "distance of top stack to our general", "14", "top stack within 10 tiles"]],
      [0.5, 5.4, 0.6, 5.4], size=13.5, row_h=0.52, first_bold=False)

s = content("8 · v51", "Opponent classifier: model and training",
            "Ten opponent types, chosen to cover what we had seen: direct and castle rushers, an expander, a turtle, "
            "gather-rushers that sweep at tick 60, 120 and 200, a spread-and-collect bot, an economy bot and a planner. "
            "We played v46 against each on 33 maps, both seats, and recorded the features every second tick. Training "
            "ran on the Mac GPU with PyTorch MPS; evaluation is on 8 maps never seen in training.")
eq(s, "mlp", 0.7, 1.9, h=0.48)
table(s, 0.7, 2.95, 6.6, [["Item", "Value"], ["Network", "14 → 64 → 64 → 10, ReLU"],
                         ["Training rows", "99,437 (25 maps)"], ["Held-out rows", "33,318 (8 maps)"],
                         ["Optimiser", "Adam, lr 0.002, 60 epochs"], ["Device", "Apple GPU (MPS)"]],
      [2.8, 3.8], size=13.5, row_h=0.5)
text(s, 7.8, 3.0, 4.9, 3.5, ["AIMA ch. 21: a small feed-forward network trained by cross-entropy.",
                             "Held-out per-tick accuracy by phase: 25% before tick 100, 53% at 100–200, 70% at "
                             "200–300, 67% after. Chance is 10%."], size=14.5, after=12)

s = content("8 · v51", "What the classifier confuses, and why",
            "Rows are the true type, columns the prediction, on held-out maps. The large off-diagonal blocks come from the "
            "game itself: some types play identically for long stretches. A rusher banks at home before it charges, so before the charge it is a "
            "turtle. A late gather-rusher expands until tick 200, so before the sweep it is an expander. The economy bot "
            "and the planner play almost the same game. Gather-early is recognised 92% of the time because it sweeps "
            "almost at once.")
picture(s, FIG / "confusion.png", 0.6, 1.65, h=5.2)
text(s, 7.3, 1.9, 5.4, 4.6, ["rusher ↔ turtle: both keep army home until the rusher launches.",
                             "gather late ↔ expander: identical before tick 200.",
                             "econ ↔ planner: 43% of planner ticks read as econ.",
                             "The defence does not depend on these labels: the approach detector reacts to a stack "
                             "that is actually walking at the general."], size=14.5, after=12)

s = content("8 · v51", "From per-tick guesses to a belief",
            "A single board is weak evidence early in the game, so v51 accumulates log-probabilities with forgetting. "
            "This is a filter over a static type with a decay, the same structure as v46's posterior, with the network's "
            "output as the observation likelihood. On held-out maps the accumulated belief is right 39% of the time at "
            "tick 100, 66% at tick 200 and 70% at tick 300. The chart is the game shown at the start of this part: the belief locks on late "
            "gather-rusher at tick 215, just as the sweep finishes.")
eq(s, "online", 0.7, 1.85, h=0.5)
picture(s, FIG / "typep.png", 0.6, 2.7, w=7.4)
text(s, 8.4, 2.85, 4.3, 3.7, ["The 10 types map onto v46's four lookahead scripts, so the planner uses them directly.",
                              "Aggressive types open the threat gate from tick 60 instead of 150.",
                              "Collector types lower the approach threshold."], size=14.5, after=12)

s = content("8 · v51", "One tick of v51",
            "The order of checks in one decision. The first two steps are new in v51. Every step from the counter-strike "
            "down is v46. A move from the fast defence skips the castle and plan layers for that tick, and the commitment "
            "keeps it in control until the attack is over.")
code = ["1  classify:  x_t ← features(board);  ℓ ← 0.95 ℓ + 0.4 log P(y | x_t)",
        "2  defend:    if a stack is incoming and need > 0:",
        "                  merge a big stack next to the general, else next PCST move (budget ETA − 2)",
        "3  capture:   a stack next to their general beats it → take it",
        "4  counter:   home loses the race → strike first if we land sooner (verified)",
        "5  urgent:    v36: verified rush, or defend when margin ≤ pad",
        "6  castle:    gated castle walk, checked past its end",
        "7  plan:      committed macro plan, replanned on events; every move passes safe_after()",
        "8  fallback:  Defender script"]
box(s, 0.7, 1.8, 11.9, 4.55, PANEL)
text(s, 0.95, 2.0, 11.4, 4.2, code, size=14, font="Menlo", color=INK, after=6)
text(s, 0.7, 6.45, 11.9, 0.4, "Average decision time about 10 ms in the simulator.", size=13, color=SLATE, italic=True)

s = content("8 · v51", "Results",
            "Simulator games, seats swapped on every map. The first two rows are v51 head to head with v46: level, which is "
            "expected, because v46 rarely builds one big stack and walks it at the general, so the fast defence seldom "
            "fires. The collector rows are where the fast defence matters, and these were measured with the same "
            "defence and no classifier (v52); the v51 rows with the classifier against these bots were not run.")
table(s, 0.7, 1.8, 11.9, [["Opponent", "Setting", "v51 / v52", "v46"],
                         ["v46 head to head", "60 games, cap 500", "32–28 (v51)", "–"],
                         ["", "captures in those 60 games", "2 won, 2 lost", "–"],
                         ["Gather-rusher, sweep at 200", "20 games, cap 600", "19–1 (v52)", "12–8"],
                         ["Gather-rusher, sweep at 120", "20 games, cap 600", "20–0 (v52)", "18–2"],
                         ["Gather-rusher, sweep at 60", "20 games, cap 600", "19–1 (v52)", "16–4"],
                         ["Spread collector", "20 games, cap 600", "13–7 (v52)", "14–6"],
                         ["Spread collector, late", "20 games, cap 600", "12–8 (v52)", "14–6"],
                         ["Direct and castle rushers", "40 games, cap 600", "40 / 40", "–"]],
      [3.8, 3.4, 2.5, 2.2], size=13.5, row_h=0.5)

s = content("8 · v51", "Limits we know about",
            "Five limits, each with the evidence behind it. None of them is hypothetical: each came out of a game or a "
            "measurement.")
text(s, 0.7, 1.9, 11.9, 4.8, ["Spread collectors are slightly harder for v51 than for v46 (13–7 and 12–8 against 14–6). "
                              "They take many castles, about 4 a game, and the committed defence gives them time to do it.",
                              "The classifier has not been shown to win games: v51 and v52 both scored 32–28 against v46.",
                              "Early in the game the classifier is weak, 39% at tick 100, because early boards look alike.",
                              "Against a careful opponent v51 attacks late. A verified attack exists only when the enemy "
                              "general is drained; v49 tried a guaranteed-kill search and gained little against v46.",
                              "All numbers are from 20 to 60 game samples. Differences of two or three games are noise."],
     size=16, after=12)

# =========================================================================== #
# 9 course
# =========================================================================== #
section(9, "What the course material bought us", "Chapter by chapter: what we used, where it lives in v51, whether it helped.",
        "Russell & Norvig, Artificial Intelligence: A Modern Approach, 4th edition")

s = content("9 · Course material", "AIMA concepts in the final bot",
            "This is the mapping the course asked for. The left column is the textbook idea, the middle is where it "
            "appears in v51, and the right column is our measured verdict on whether it earned its place.")
table(s, 0.7, 1.75, 11.9, [["AIMA idea (chapter)", "Where in v51", "Verdict"],
                          ["Task environments (2)", "Fully observable, simultaneous, dynamic: framed every design choice", "essential"],
                          ["Informed search, A* (3)", "Routing; exact BFS distances", "solved early"],
                          ["AND-OR search (4)", "survives() and verify() rollouts", "the safety layer"],
                          ["Constraints (5)", "Safety margin as a hard constraint on every move", "essential"],
                          ["Alpha-beta, pruning (6)", "ABCD over plans (v50); ProbCut era (v12–v21)", "little gain"],
                          ["Bayes' rule, filtering (12, 14)", "Opponent posterior; classifier belief", "useful, not decisive"],
                          ["Decisions, VOI (16)", "Hurwicz plan score; metareasoning (v18)", "score kept, VOI not"],
                          ["Bandits (17)", "Exp3 across games", "untested in matches"],
                          ["Game theory (18)", "Matrix games, maximin, safe exploitation", "local fights only"],
                          ["Learning, networks (19, 21)", "Logistic value model; 10-type MLP", "modest"]],
      [3.4, 6.2, 2.3], size=12.5, row_h=0.45)

s = content("9 · Course material", "Five things we learned",
            "Each of these cost us at least one failed bot to learn.")
lessons = [("Measure the right thing.", "Swap seats, count captures apart from tiebreaks, and use caps long enough "
            "for captures to happen. Half our early gains were noise."),
           ("Model the event that loses games.", "Twenty bots of deeper and learned search moved nothing. One rule about "
            "the general took the rusher from 3 of 6 to 20 of 20."),
           ("Constraints before objectives.", "Safety as a hard constraint, preference as the objective, kept every later "
            "layer from undoing the earlier ones."),
           ("People are the best test set.", "Every big step after v37 came from replaying a game a person had won."),
           ("Learn the opponent inside the safety layer.", "Beliefs may change what v51 prefers. They never change what "
            "it is allowed to do.")]
for i, (h, b) in enumerate(lessons):
    y = 1.85 + i * 0.98
    text(s, 0.7, y, 0.5, 0.5, str(i + 1), size=22, color=RULE, font=HEAD)
    text(s, 1.3, y + 0.02, 11.3, 0.9, [[(h + " ", {"bold": True, "color": CHAR}), (b, {})]], size=15.5, after=0)

s = content("References", "Papers and book",
            "Every paper listed here is either implemented in one of the bots or is the source of a formula on a slide.")
refs = ["Russell, S. & Norvig, P. Artificial Intelligence: A Modern Approach, 4th ed. Pearson, 2020.",
        "Likhachev, M., Gordon, G. & Thrun, S. ARA*: Anytime A* with provable bounds on sub-optimality. NIPS 2003.",
        "Koenig, S. & Likhachev, M. Real-time adaptive A*. AAMAS 2006.",
        "Hernández, C. & Baier, J. Real-time adaptive A* with depression avoidance. AAAI 2012.",
        "Buro, M. ProbCut: an effective selective extension of the alpha-beta algorithm. ICCA Journal, 1995.",
        "Russell, S. & Wefald, E. Principles of metareasoning. Artificial Intelligence, 1991; UAI 1990.",
        "Churchill, D., Saffidine, A. & Buro, M. Fast heuristic search for RTS game combat scenarios (ABCD). AIIDE 2012.",
        "Barriga, N., Stanescu, M. & Buro, M. Puppet Search: enhancing scripted behavior by look-ahead search. AIIDE 2015.",
        "Straka, M., Lisý, V. & Schmid, M. Superhuman AI for Generals.io using self-play reinforcement learning. arXiv 2606.23348, 2026.",
        "Synnaeve, G. & Bessière, P. A Bayesian model for opening prediction in RTS games. AIIDE 2011.",
        "Weber, B. & Mateas, M. A data mining approach to strategy prediction. CIG 2009.",
        "Ganzfried, S. & Sandholm, T. Safe opponent exploitation. ACM TEAC, 2015.",
        "Churchill, D. & Buro, M. Build order optimization in StarCraft. AIIDE 2011.",
        "Tavares, A. et al. Rock, Paper, StarCraft: strategy selection in real-time strategy games. AIIDE 2016.",
        "Auer, P., Cesa-Bianchi, N., Freund, Y. & Schapire, R. The nonstochastic multiarmed bandit problem. SIAM J. Comput., 2002.",
        "Lisý, V., Kovařík, V., Lanctot, M. & Bošanský, B. Convergence of MCTS in simultaneous move games. NeurIPS 2013."]
text(s, 0.7, 1.75, 11.9, 5.1, refs, size=11.5, color=INK, after=3)

prs.core_properties.title = "Search agents for perfect-information Generals.io"
prs.core_properties.author = "DAC-203 project"
prs.save(str(OUT))
print("saved", OUT, N[0], "slides")
