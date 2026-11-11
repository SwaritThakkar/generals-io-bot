"""v51 talk deck. Theme "Arena Night" (custom, via theme-factory's create-your-own
route): the game's own palette, slate background, brass accent, red/blue players."""
import json
import pathlib

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = pathlib.Path(__file__).parent
SH, ART = HERE / "shots", HERE / "art"
OUT = pathlib.Path(__file__).resolve().parents[2] / "presentation" / "v51-talk.pptx"

BG, PANEL, PANEL2, LINE = "0B1012", "121A1D", "18232A", "243036"
INK, INK2, INK3, BRASS = "E2E9EA", "9DB0B4", "6B7D82", "D7A244"
RED, BLUE = "D1544A", "4A86C6"
HEAD, BODY = "Georgia", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
N = [0]


def rgb(h):
    return RGBColor.from_string(h)


def new(note=None):
    s = prs.slides.add_slide(BLANK)
    N[0] += 1
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(BG)
    if note:
        s.notes_slide.notes_text_frame.text = note
    return s


def rect(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        r.fill.solid()
        r.fill.fore_color.rgb = rgb(fill)
    else:
        r.fill.background()
    if line:
        r.line.color.rgb = rgb(line)
        r.line.width = Pt(1.25)
    else:
        r.line.fill.background()
    r.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        r.adjustments[0] = radius
    return r


def text(s, x, y, w, h, paras, size=17, color=INK, font=BODY, bold=False, italic=False, align=PP_ALIGN.LEFT,
         after=8, anchor=MSO_ANCHOR.TOP, spacing=1.1):
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
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.name = o.get("font", font)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def img(s, path, x, y, w=None, h=None):
    iw, ih = Image.open(path).size
    if w and h:
        k = min(w / iw, h / ih)
        pw, ph = iw * k, ih * k
        return s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    if w:
        return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(w * ih / iw))
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(h * iw / ih), Inches(h))


def eq(s, name, x, y, h=0.5, maxw=8.0):
    p = ART / f"eq_{name}.png"
    iw, ih = Image.open(p).size
    w = h / 120 * iw
    if w > maxw:
        w = maxw
    return s.shapes.add_picture(str(p), Inches(x), Inches(y), Inches(w), Inches(w * ih / iw))


def header(s, label, title):
    text(s, 0.7, 0.42, 9, 0.3, label.upper(), size=11, color=BRASS, bold=True, after=0)
    text(s, 0.7, 0.72, 12, 0.8, title, size=30, font=HEAD, after=0)
    text(s, 12.0, 7.02, 0.7, 0.25, str(N[0]), size=9, color=INK3, align=PP_ALIGN.RIGHT, after=0)


def chip(s, x, y, label, value, color=BRASS):
    w = 0.25 + 0.085 * (len(label) + len(value)) + 0.2
    rect(s, x, y, w, 0.36, PANEL2, color, MSO_SHAPE.ROUNDED_RECTANGLE, 0.5)
    text(s, x + 0.14, y + 0.07, w - 0.2, 0.25, [[(label + "  ", {"bold": True, "color": color}), (value, {"color": INK})]],
         size=11, after=0)
    return x + w + 0.15


def chips(s, y, book, paper=None, x=0.7, stack=False):
    x2 = chip(s, x, y, "BOOK", book, BRASS)
    if paper:
        if stack:
            chip(s, x, y + 0.45, "PAPER", paper, BLUE)
        else:
            chip(s, x2, y, "PAPER", paper, BLUE)


def stat(s, x, y, big, label, color=BRASS, w=2.4):
    text(s, x, y, w, 0.7, big, size=34, font=HEAD, color=color, after=0)
    text(s, x, y + 0.72, w, 0.5, label, size=12, color=INK2, after=0)


def panel(s, x, y, w, h, fill=PANEL):
    return rect(s, x, y, w, h, fill, None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.04)


def section(num, title, sub, shot=None, note=None):
    s = new(note)
    if shot:
        pic = img(s, SH / shot, 7.0, 0.0, h=7.5)
        rect(s, 6.9, 0, 0.6, 7.5, BG)
    text(s, 0.8, 2.3, 2, 1.1, num, size=60, font=HEAD, color=BRASS, after=0)
    text(s, 0.8, 3.35, 6.0, 1.2, title, size=36, font=HEAD, after=0)
    text(s, 0.82, 4.45, 5.8, 1.0, sub, size=16, color=INK2, after=0)
    return s


def arrow(s, x0, y0, x1, y1, color=INK3):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(2)
    ln = c.line._get_or_add_ln()
    from lxml import etree
    from pptx.oxml.ns import qn
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def walk_slide(t, title, lines, stats, shot, note, belief=None):
    s = new(note)
    img(s, SH / shot, 0.35, 0.35, h=6.8)
    x = 7.45
    text(s, x, 0.45, 5.4, 0.3, "ONE GAME, STEP BY STEP", size=11, color=BRASS, bold=True, after=0)
    text(s, x, 0.75, 5.4, 0.9, [[("tick ", {"size": 18, "color": INK2}), (str(t), {"size": 40, "color": BRASS})]],
         font=HEAD, after=0)
    text(s, x, 1.65, 5.4, 0.7, title, size=24, font=HEAD, after=0)
    text(s, x, 2.45, 5.4, 2.6, lines, size=15.5, color=INK, after=9)
    yy = 5.25 if not belief else 4.65
    for i, (big, lab, col) in enumerate(stats):
        stat(s, x + i * 1.85, yy, big, lab, col, w=1.8)
    if belief:
        img(s, ART / belief, x, 5.95, h=1.25)
    text(s, 12.0, 7.02, 0.7, 0.25, str(N[0]), size=9, color=INK3, align=PP_ALIGN.RIGHT, after=0)
    return s


# =========================================================================== #
# 1 opening
# =========================================================================== #
s = new("We were asked to build a game-playing agent with the methods from the course. This talk explains the agent we "
        "ended with, v51: how it decides a move, which chapter of the book and which paper each part comes from, and "
        "one real game played step by step. At the end the professor plays it live.")
img(s, SH / "t235.png", 6.9, 0, h=7.5)
rect(s, 6.75, 0, 0.5, 7.5, BG)
text(s, 0.8, 1.75, 6, 0.35, "DAC-203 · ARTIFICIAL INTELLIGENCE · IIT ROORKEE", size=12, color=BRASS, bold=True)
text(s, 0.8, 2.25, 5.9, 2.2, ["Building a game agent", "from the textbook"], size=42, font=HEAD, after=0, spacing=1.0)
text(s, 0.8, 4.15, 5.8, 1.5, ["v51: a search-and-belief agent for Generals.io, built from Russell & Norvig and six papers."],
     size=17, color=INK2)
text(s, 0.8, 6.3, 5.8, 0.4, "Right: tick 235 of a real game. v51 (red) has pulled 124 army into its general, one tile "
     "from a 90-army attack.", size=11.5, color=INK3, italic=True)

s = new("Three parts. First the game and the agent design in the book's terms. Then how v51 thinks, one idea at a time, "
        "each idea with its chapter and its paper. Then one game step by step, the results, and the live game.")
header(s, "This talk", "Three parts, then a live game")
for i, (n_, t_, d_, shot) in enumerate([("1", "The task", "The game and the agent design, in AIMA's terms", "t120.png"),
                                         ("2", "How v51 thinks", "Eight ideas, each tied to a chapter and a paper", "t219_plan.png"),
                                         ("3", "One game, then you play", "A replay with v51's reasoning drawn on the board; results; live game", "t407.png")]):
    x = 0.7 + i * 4.1
    panel(s, x, 1.8, 3.8, 4.9)
    img(s, SH / shot, x + 0.25, 2.05, w=3.3)
    text(s, x + 0.3, 5.5, 3.3, 0.5, [[(n_ + "  ", {"color": BRASS, "size": 22, "font": HEAD}), (t_, {"size": 19, "font": HEAD})]], after=2)
    text(s, x + 0.3, 6.0, 3.3, 0.6, d_, size=12.5, color=INK2)

# =========================================================================== #
# part 1
# =========================================================================== #
section("1", "The task", "A game, an agent, and the book's way of describing both.", "t120.png",
        "Part 1 sets up the problem in the vocabulary of chapter 2.")

s = new("The game is a variant of Generals.io with no fog of war. Walk through the screenshot: red is us, blue the "
        "opponent, the crown is a general, circles are castles. Then the four rules on the right. The rule that matters "
        "most later is the last one: army grows on its own, and generals and castles grow fast.")
header(s, "1 · The task", "The game in one minute")
img(s, SH / "t120.png", 0.7, 1.55, h=5.3)
text(s, 6.4, 1.7, 6.3, 5.0, [[("The board. ", {"bold": True, "color": BRASS}), ("23 × 23 tiles. The number on a tile is the "
                                                                               "army standing on it. Each player has one general (the crown). Grey tiles are mountains; circles are castles.", {})],
                             [("One move per tick. ", {"bold": True, "color": BRASS}), ("Pick a tile with 2 or more army, push all but one of it (or half) to a "
                                                                                       "neighbour. Both players move at the same moment.", {})],
                             [("Fights subtract. ", {"bold": True, "color": BRASS}), ("10 army walking onto 7 enemy army leaves 3 and takes the tile.", {})],
                             [("Army grows on its own. ", {"bold": True, "color": BRASS}), ("Generals and castles add 1 every 2 ticks; every tile adds 1 every 50 ticks.", {})],
                             [("Winning. ", {"bold": True, "color": BRASS}), ("Take the enemy general. Otherwise, at the tick limit, more land wins.", {})]],
     size=16, after=14)

s = new("Chapter 2 designs any agent by stating PEAS and the environment's properties. Go through the four boxes, then "
        "the conclusion: the board is fully visible, so the only uncertainty is the opponent's intention; results come "
        "hundreds of ticks later, so the agent must estimate how good a position is; both players move at once, so each "
        "decision is a small game in the sense of chapter 18.")
header(s, "1 · The task", "The agent, in AIMA's terms")
chips(s, 1.5, "ch. 2 Intelligent agents · ch. 18 Multiagent decisions")
for i, (k, v) in enumerate([("Performance", "take the enemy general; otherwise more land at the limit"),
                            ("Environment", "23 × 23 board, one opponent"),
                            ("Actuators", "one move per tick: a tile, a direction, all-but-one or half"),
                            ("Sensors", "the whole board, every tile's owner and army")]):
    y = 2.15 + i * 0.95
    panel(s, 0.7, y, 6.0, 0.8)
    text(s, 0.95, y + 0.12, 1.8, 0.5, k, size=16, font=HEAD, color=BRASS)
    text(s, 2.75, y + 0.15, 3.8, 0.6, v, size=14.5)
text(s, 7.2, 2.15, 5.5, 0.4, "Environment properties", size=16, font=HEAD, color=BRASS)
text(s, 7.2, 2.6, 5.5, 1.9, ["Fully observable · deterministic · sequential", "Dynamic: a decision every 0.5 s",
                             "Discrete · two competing agents, moving simultaneously"], size=15, color=INK2, after=6)
panel(s, 7.2, 4.35, 5.5, 2.25, PANEL2)
text(s, 7.45, 4.5, 5.1, 2.0, [[("So we need ", {}), ("a utility-based agent", {"bold": True, "color": BRASS}),
                               (" that keeps a belief about the opponent (the board itself is never in doubt), judges positions with an "
                                "estimate because the reward arrives hundreds of ticks later, and treats each tick as a simultaneous game.", {})]],
     size=15)

s = new("This is the fact the whole design rests on. Show the strip: 30 army walks 8 tiles and leaves one behind on "
        "each, so 22 reach the general's door and 21 hit. In those 8 ticks the general grew from 12 to 16. So one stack "
        "against a general that keeps its army at home usually fails. A general falls when it has been emptied, or when "
        "several stacks arrive one after another.")
header(s, "1 · The task", "Walking costs army, waiting earns it")
text(s, 0.7, 1.6, 4, 0.3, "start", size=12, color=INK2)
img(s, SH / "strip0.png", 0.7, 1.95, w=11.9)
text(s, 0.7, 3.25, 4, 0.3, "8 ticks later", size=12, color=INK2)
img(s, SH / "strip8.png", 0.7, 3.6, w=11.9)
stat(s, 0.7, 4.95, "30 → 22", "the attacker leaves 1 army on every tile it crosses", RED, w=3.6)
stat(s, 4.6, 4.95, "12 → 16", "the general grows by 1 every 2 ticks", BLUE, w=3.6)
panel(s, 8.4, 4.85, 4.3, 1.75, PANEL2)
text(s, 8.6, 5.0, 3.95, 1.5, ["A general falls only when it has been emptied, or when several stacks arrive one after another."],
     size=15.5, color=INK)

# =========================================================================== #
# part 2
# =========================================================================== #
section("2", "How v51 thinks", "Eight ideas. Each one: what it does, the chapter it comes from, the paper we borrowed from.",
        "t219_plan.png", "Part 2 goes through v51's decision loop one idea at a time.")

s = new("The agent loop, left to right. Every tick v51 looks at the board, updates its belief about the opponent, asks "
        "whether an attack is coming, and either defends fast or chooses a plan by looking ahead. Every move it makes, "
        "whatever produced it, passes a safety test first. The next slides take each box in turn.")
header(s, "2 · How v51 thinks", "One tick of v51")
steps = [("Look", "read the whole board", "ch. 2"), ("Believe", "update the guess about the opponent's style", "ch. 12, 14, 21"),
         ("Check danger", "how short would my general be?", "ch. 3"), ("Defend or plan", "gather home fast, or pick a plan by looking ahead", "ch. 6, 16"),
         ("Safety test", "play the move out before making it", "ch. 4"), ("Move", "one move, every 0.5 s", "")]
for i, (h, d, c) in enumerate(steps):
    x = 0.7 + i * 2.08
    panel(s, x, 2.3, 1.85, 2.6, PANEL if i not in (3, 4) else PANEL2)
    text(s, x + 0.15, 2.45, 1.6, 0.5, f"{i + 1}", size=26, font=HEAD, color=BRASS)
    text(s, x + 0.15, 3.05, 1.6, 0.5, h, size=17, font=HEAD)
    text(s, x + 0.15, 3.55, 1.6, 1.0, d, size=12.5, color=INK2)
    if c:
        text(s, x + 0.15, 4.5, 1.6, 0.3, c, size=11, color=BRASS)
    if i < len(steps) - 1:
        arrow(s, x + 1.86, 3.6, x + 2.07, 3.6, INK3)
text(s, 0.7, 5.45, 11.9, 1.0, "Two things are fixed in this loop: a move that fails the safety test is never played, and "
     "when an attack is coming the defence takes every move until it is over.", size=15.5, color=INK2)

# --- idea 1: belief ------------------------------------------------------------
s = new("Idea 1. v51 cannot see the opponent's plan, so it keeps a probability for each of ten playing styles. Each "
        "tick it measures fourteen things about the opponent from the board, and a small neural network turns those "
        "numbers into a probability per style. Look at the two bars: at tick 205 the opponent looks like an expander. "
        "Ten ticks later, after it empties its general into one stack, it looks like a late gather-rusher. The papers: "
        "Synnaeve and Bessière update a Bayesian belief over strategies from what they observe; Weber and Mateas train "
        "a classifier on game features to name the strategy early. We used a classifier as the observation model "
        "inside a Bayesian filter.")
header(s, "Idea 1 · Knowing the opponent", "A belief about the opponent's style")
chips(s, 1.5, "ch. 12 Bayes · ch. 14 filtering · ch. 21 networks", "Synnaeve & Bessière 2011 · Weber & Mateas 2009")
text(s, 0.7, 2.15, 6.0, 3.0, ["The board shows everything except the opponent's intention. v51 keeps a probability for "
                              "each of ten playing styles: rushers, an expander, a turtle, three gather-rushers, a "
                              "spread-and-collect player, and two economy bots.",
                              "Every tick it measures 14 things about the opponent (land, army, how much sits in its "
                              "general, how far its biggest stack is and whether it is getting closer) and a small "
                              "neural network turns them into a probability per style."], size=15.5, after=10)
panel(s, 7.2, 2.15, 5.5, 4.6)
text(s, 7.45, 2.3, 5, 0.3, "tick 205", size=13, color=INK2, bold=True)
img(s, ART / "belief205.png", 7.45, 2.6, h=1.75)
text(s, 7.45, 4.45, 5, 0.3, "tick 215, after the opponent empties its general", size=13, color=INK2, bold=True)
img(s, ART / "belief215.png", 7.45, 4.75, h=1.75)
text(s, 0.7, 5.45, 6.0, 1.3, "Trained on our own games against all ten styles, on the Mac GPU. On maps it never saw, "
     "it names the right style 66% of the time by tick 200 and 70% by tick 300. Guessing would give 10%.",
     size=13.5, color=INK2)

s = new("The formula, after the words. Today's belief is mostly yesterday's belief, nudged toward what today's board "
        "suggests. l is the accumulated evidence for a style, P of y given x is the network's opinion of today's board. "
        "The 0.95 lets old evidence fade, so the belief can follow an opponent that changes style; that is the "
        "predict-observe-update filter of chapter 14. The chart is the whole game: expander until the sweep, then late "
        "gather-rusher.")
header(s, "Idea 1 · Knowing the opponent", "How the belief moves over a game")
text(s, 0.7, 1.6, 6.3, 0.5, "In words: today's belief is mostly yesterday's belief, nudged toward what today's board suggests.",
     size=16, color=INK)
panel(s, 0.7, 2.45, 6.3, 1.0, PANEL2)
eq(s, "belief", 0.95, 2.67, h=0.42, maxw=5.9)
text(s, 0.7, 3.65, 6.3, 2.5, [[("ℓ(y)", {"bold": True, "color": BRASS}), ("  the evidence collected so far for style y", {})],
                              [("P(y | x)", {"bold": True, "color": BRASS}), ("  the network's opinion of today's board x", {})],
                              [("0.95", {"bold": True, "color": BRASS}), ("  old evidence fades, so the belief can follow an opponent that changes style", {})],
                              [("0.4", {"bold": True, "color": BRASS}), ("  how much one board counts", {})]], size=14.5, after=8)
img(s, ART / "belief_game.png", 7.25, 1.6, w=5.5)
text(s, 7.25, 4.4, 5.4, 2.0, "In this game the belief reads expander until tick 200, because before its sweep this "
     "bot plays exactly like an expander. It switches to late gather-rusher at tick 215, as the sweep finishes.",
     size=13.5, color=INK2)

# --- idea 2: safety score -------------------------------------------------------
s = new("Idea 2. The danger from part 1 is several enemy stacks arriving one after another. v51 measures it every tick. "
        "For each number of ticks T it compares two amounts: how much enemy army could reach the general in T ticks if "
        "the enemy walks its stacks in one after another, and how much the general will hold plus what our own stacks "
        "can bring home in time. The blue line on the board is one such enemy walk at tick 219.")
header(s, "Idea 2 · Is my general safe?", "The safety score")
chips(s, 1.5, "ch. 3 search (BFS distances), greedy approximation", "Straka, Lisý & Schmid 2026 (Human.exe)", stack=True)
img(s, SH / "t219_walks.png", 7.3, 1.45, h=5.4)
text(s, 0.7, 2.6, 6.2, 4.3, ["For every horizon of T ticks, v51 compares two amounts:",
                              [("Threat: ", {"bold": True, "color": BLUE}), ("the enemy army that could reach our general in T ticks, walking its stacks in one after another.", {})],
                              [("Defence: ", {"bold": True, "color": RED}), ("what our general will hold by then, plus what our own stacks can walk home in time.", {})],
                              "The safety score is defence minus threat at the worst T. Small means an attack is possible soon.",
                              [("Right: ", {"color": INK2}), ("tick 219. The light-blue path is the enemy's walk onto our general (gold box): 111 army, 18 tiles away.", {"color": INK2})]],
     size=15.5, after=10)

s = new("The formula, with every symbol in words. G is our general's army, growth of T what it gains in T ticks, D us "
        "and D them the army each side can deliver onto our general. Our walks get a small lead because defence has to "
        "start early. Choosing which stacks to walk under a move budget is NP-hard; Human.exe solves its gathering as a "
        "prize-collecting Steiner tree, and we use a greedy version. What we extrapolated: Human.exe uses this to collect "
        "its own army; we run it on the enemy's army to measure the threat as well.")
header(s, "Idea 2 · Is my general safe?", "Measuring it, and watching it in a game")
text(s, 0.7, 1.6, 12, 0.4, "In words: what my general and my nearby stacks can have there, minus what the enemy can bring, at the worst moment.",
     size=16)
panel(s, 0.7, 2.15, 11.9, 1.0, PANEL2)
eq(s, "safety", 0.95, 2.37, h=0.42, maxw=11.4)
text(s, 0.7, 3.35, 6.4, 3.0, [[("G", {"bold": True, "color": BRASS}), ("  our general's army now;   ", {}), ("growth(T)", {"bold": True, "color": BRASS}), ("  what it gains in T ticks", {})],
                              [("D_us, D_them", {"bold": True, "color": BRASS}), ("  army each side can deliver to our general in T moves, one walk after another, each tile counted once", {})],
                              [("lead", {"bold": True, "color": BRASS}), ("  2 ticks plus a tenth of T: our walks must arrive early", {})],
                              [("From the paper: ", {"bold": True, "color": BLUE}), ("Human.exe gathers its own army this way. We run it on the enemy's army too, to measure the threat.", {})]],
     size=14, after=8)
img(s, ART / "safety_game.png", 7.35, 3.35, w=5.3)
text(s, 7.35, 5.85, 5.3, 1.0, "The score falls from 26 to 12 when the opponent empties its general into one stack, and "
     "recovers to about 37 while v51 gathers.", size=13.5, color=INK2)

# --- idea 3: safety test --------------------------------------------------------
s = new("Idea 3, from chapter 4. Before making any move, v51 tries it on a copy of the board. If the safety score stays "
        "above a small cushion, fine. If not, it plays the move out: the enemy launches an all-out attack along each "
        "likely route and v51 defends. If the general falls in any of those imagined games, the move is thrown away. "
        "That is an AND-OR tree: our move is the OR choice, the enemy's routes are the AND branches, and the move is "
        "accepted only if every branch is safe.")
header(s, "Idea 3 · Never make an unsafe move", "Play the move out before making it")
chips(s, 1.5, "ch. 4 AND-OR search")
panel(s, 0.7, 2.15, 6.0, 4.6)
rect(s, 2.6, 2.4, 2.2, 0.75, PANEL2, BRASS, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
text(s, 2.6, 2.55, 2.2, 0.5, "our move  (OR)", size=14, align=PP_ALIGN.CENTER, color=BRASS)
for i, (lab, ok) in enumerate([("enemy route 1", True), ("enemy route 2", True), ("enemy route 3", False)]):
    x = 0.95 + i * 1.9
    arrow(s, 3.7, 3.15, x + 0.8, 3.85, INK3)
    rect(s, x, 3.85, 1.6, 0.7, PANEL2, BLUE, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
    text(s, x, 3.98, 1.6, 0.5, lab, size=12.5, align=PP_ALIGN.CENTER)
    arrow(s, x + 0.8, 4.55, x + 0.8, 5.15, INK3)
    rect(s, x, 5.15, 1.6, 0.7, RED if not ok else "2E4A3A", None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
    text(s, x, 5.3, 1.6, 0.5, "general holds" if ok else "general falls", size=12.5, align=PP_ALIGN.CENTER)
text(s, 0.95, 6.1, 5.5, 0.5, "AND: one fall is enough to reject the move.", size=13, color=INK2, align=PP_ALIGN.CENTER)
text(s, 7.2, 2.15, 5.5, 4.6, ["Step 1. Try the move on a copy of the board. If the safety score is still above a small "
                              "cushion, the move is allowed.",
                              "Step 2. Otherwise, play it out. The enemy attacks along each likely route with everything "
                              "it has; v51 defends. If the general falls in even one of these imagined games, the move "
                              "is thrown away.",
                              "The cushion grows with the enemy's biggest stack, so v51 is more careful when the enemy "
                              "can hit harder.",
                              "The same test, with four imagined enemy replies, decides whether an attack is allowed (idea 8)."],
     size=15, after=10)

# --- idea 4: fast defence --------------------------------------------------------
s = new("Idea 4, the new part of v51. When a large enemy stack is clearly coming, v51 stops everything and defends. "
        "Notice: a stack that has come at least two tiles closer within five ticks and is big enough to matter. Measure: "
        "how much the general alone will be short when it arrives; stacks parked next to the general do not count. "
        "Gather: move as much army as possible into the general before it arrives. The brass arrows on the board are "
        "the actual plan at tick 219: fourteen moves, all leading into the general.")
header(s, "Idea 4 · Defending fast", "Notice, measure, gather")
chips(s, 1.5, "ch. 3 problem formulation, greedy search, binary search", "Straka, Lisý & Schmid 2026 (Human.exe)", stack=True)
img(s, SH / "t219_plan.png", 7.3, 1.45, h=5.4)
for i, (h_, d_) in enumerate([("Notice", "A big enemy stack (15+ army, or a fifth of our general) has come at least 2 tiles closer in 5 ticks. Its distance is the time left."),
                              ("Measure", "How short will the general alone be when it lands? Stacks beside the general do not count: the attacker can land first."),
                              ("Gather", "Spend every move pulling army into the general, using a tree of paths that fits the time left.")]):
    y = 2.55 + i * 1.38
    panel(s, 0.7, y, 6.3, 1.22)
    text(s, 0.95, y + 0.15, 0.6, 0.8, str(i + 1), size=30, font=HEAD, color=BRASS)
    text(s, 1.65, y + 0.14, 5.2, 0.4, h_, size=17, font=HEAD)
    text(s, 1.65, y + 0.55, 5.2, 0.8, d_, size=13.5, color=INK2)
text(s, 0.7, 6.65, 6.3, 0.4, "Right: tick 219, the 14-move gather plan (brass arrows) into the general (gold box).",
     size=12, color=INK3, italic=True)

s = new("How the gather tree is built. Think of each of our tiles as holding a prize, its army minus the one that must "
        "stay. Each step costs a move. We want the set of paths into the general that collects the most prize within the "
        "moves left: a rooted prize-collecting Steiner tree, NP-hard in general. Human.exe's trick, which we copied: put a "
        "price lambda on each move, grow the tree greedily, and binary-search the price until the tree fits the time. "
        "Then play the moves from the outer leaves inward so the army snowballs. Any stack of ten or more that is "
        "already next to the general steps in at once.")
header(s, "Idea 4 · Defending fast", "The gather tree")
text(s, 0.7, 1.6, 7.3, 0.5, "In words: collect the most army into the general using no more moves than there are ticks left.", size=16)
panel(s, 0.7, 2.45, 7.3, 1.0, PANEL2)
eq(s, "tree", 0.95, 2.7, h=0.40, maxw=6.9)
text(s, 0.7, 3.65, 6.3, 3.2, [[("a_v − 1", {"bold": True, "color": BRASS}), ("  the prize on tile v: its army, minus the one that must stay", {})],
                              [("λ", {"bold": True, "color": BRASS}), ("  a price per move. High price: a small tree near the general. Low price: a big tree that takes too long.", {})],
                              [("ETA − 2", {"bold": True, "color": BRASS}), ("  the moves available before the attacker lands, with 2 to spare", {})],
                              [("How: ", {"bold": True, "color": BLUE}), ("grow the tree greedily for a given λ, binary-search λ until it fits, then play the moves from the leaves inward so army snowballs.", {})]],
     size=14, after=8)
img(s, SH / "t230.png", 8.3, 1.55, h=5.0)
text(s, 8.3, 6.6, 4.4, 0.4, "Tick 230: three moves of the plan left.", size=12, color=INK3, italic=True)

# --- idea 5: plans ----------------------------------------------------------------
pt = json.loads((ART / "plan_table.json").read_text())
s = new("Idea 5. When nothing is urgent, v51 does not search individual moves; there are tens of thousands of joint "
        "moves per tick. It chooses between about six plans, plays each forward 40 ticks against four imagined "
        "opponents, and scores the end with a small learned model of the chance of winning. The table is real, from tick "
        "240 of this game: the castle plans score 0.69 and keeping on expanding scores 0.59. Puppet Search is the paper: "
        "search over the choices of scripts instead of raw moves. The score is half the average, weighted by the belief, "
        "and half the worst case.")
header(s, "Idea 5 · Choosing what to do", "Imagine the next 40 ticks, four times")
chips(s, 1.5, "ch. 6 adversarial search, playouts · ch. 16 decisions · ch. 19 logistic regression", "Barriga, Stanescu & Buro 2015 (Puppet Search)")
names = {"expand": "keep expanding", "hold": "hold", "castle241": "take castle A", "castle264": "take castle B",
         "castle354": "take castle C", "strike": "strike", "rush": "rush"}
show = [r for r in pt["rows"] if r["plan"] in ("expand", "castle241", "castle264", "castle354", "rush")]
best = max(r["score"] for r in show)
panel(s, 0.7, 2.15, 7.4, 4.55)
heads = ["plan", "vs rusher", "vs castle\nrusher", "vs expander", "vs passive", "score"]
xs = [0.95, 3.0, 4.05, 5.1, 6.15, 7.15]
for x_, h_ in zip(xs, heads):
    text(s, x_, 2.3, 1.1, 0.5, h_, size=11.5, color=INK2, bold=True)
for i, r in enumerate(show):
    y = 2.95 + i * 0.68
    win = abs(r["score"] - best) < 1e-6 or r["plan"] == "castle241"
    if win:
        rect(s, 0.8, y - 0.08, 7.2, 0.6, PANEL2, BRASS, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
    text(s, xs[0], y, 2.0, 0.4, names[r["plan"]], size=14, color=INK)
    for x_, o in zip(xs[1:5], ["rusher", "castle_rusher", "expander", "hold"]):
        text(s, x_, y, 1.0, 0.4, f"{r['vals'][o]:.2f}", size=14, color=INK2, font="Menlo")
    text(s, xs[5], y, 1.0, 0.4, f"{r['score']:.2f}", size=14, color=BRASS if win else INK, font="Menlo", bold=win)
text(s, 0.95, 6.35, 7.0, 0.3, "Real values at tick 240 of the game. Numbers are the estimated chance of winning after 40 ticks.",
     size=11, color=INK3, italic=True)
text(s, 8.5, 2.15, 4.2, 1.3, "In words: prefer plans that do well against the opponent we think we face, but not plans "
     "that would be a disaster if we are wrong.", size=15)
panel(s, 8.5, 3.55, 4.2, 0.95, PANEL2)
eq(s, "plan", 8.65, 3.75, h=0.36, maxw=3.9)
text(s, 8.5, 4.7, 4.2, 2.0, [[("V(a,b)", {"bold": True, "color": BRASS}), ("  chance of winning after plan a against imagined opponent b", {})],
                             [("P(b)", {"bold": True, "color": BRASS}), ("  the belief from idea 1", {})],
                             [("½ average + ½ worst", {"bold": True, "color": BRASS}), ("  the Hurwicz criterion from decision theory", {})]],
     size=13, after=6)

# --- idea 6: castles ----------------------------------------------------------------
s = new("Idea 6. A castle produces as much as 25 plain tiles, but taking one drains the stack that walks there. v51 takes "
        "a castle only if it pays back before the game ends and the safety score still holds on the board as it will "
        "be after the walk. Whether castles are worth buying at all comes from an income race, the idea in Churchill and "
        "Buro's build-order work: project both players' army to the end of the game at their current income.")
header(s, "Idea 6 · Castles and the economy", "Buy a castle only when it pays and home stays safe")
chips(s, 1.5, "ch. 16 utility, expected value", "Churchill & Buro 2011 (build-order planning)", stack=True)
img(s, SH / "t298.png", 7.3, 1.45, h=5.4)
text(s, 0.7, 2.6, 6.3, 4.3, [[("Worth it? ", {"bold": True, "color": BRASS}), ("A castle makes 1 army every 2 ticks, as much as 25 tiles. It must earn back "
                                                                              "its garrison plus the walk before the game ends.", {})],
                              [("Safe? ", {"bold": True, "color": BRASS}), ("v51 moves the board forward to the end of the walk, with the enemy moved "
                                                                          "forward too, and checks the safety score there. The walk spends the stack, and that is when home is weakest.", {})],
                              [("Needed? ", {"bold": True, "color": BRASS}), ("Project both players' army to the end of the game at current income. While v51 "
                                                                            "is not clearly ahead, castles are worth buying; once it is, only cheap ones.", {})],
                              [("Right: ", {"color": INK2}), ("tick 298, v51's first castle in this game, taken after the attack was beaten.", {"color": INK2})]],
     size=15, after=12)

# --- idea 7: gates ----------------------------------------------------------------
s = new("Idea 7. The belief from idea 1 changes how v51 plays. Against an aggressive style it keeps more army at home and "
        "may counter-attack; against a passive one it keeps less at home and expands. Ganzfried and Sandholm asked when it "
        "is safe to exploit a guessed opponent; their answer is to exploit only as far as you can afford to be wrong. We "
        "build that in: the belief can make v51 greedier or more careful, but it can never switch off the safety test. A "
        "wrong guess costs efficiency, never the general.")
header(s, "Idea 7 · Using the belief safely", "Exploit the opponent, but never past the safety test")
chips(s, 1.5, "ch. 18 game theory: best response, maximin", "Ganzfried & Sandholm 2015 (safe exploitation)")
for i, (h_, a_, b_) in enumerate([("Threat switch", "aggressive style seen, or an attack observed", "keep 60% of every threat at home; counter-attacks allowed"),
                                  ("", "quiet opponent for 150 ticks", "keep 40% at home; expand more"),
                                  ("Castle switch", "not clearly ahead, or the opponent takes castles", "full castle play, enemy castles first"),
                                  ("", "clearly ahead on income", "only cheap castles, never from the general")]):
    y = 2.2 + i * 1.0
    panel(s, 0.7, y, 7.6, 0.85, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.95, y + 0.25, 1.8, 0.4, h_, size=15, font=HEAD, color=BRASS)
    text(s, 2.75, y + 0.12, 2.6, 0.7, a_, size=13, color=INK2)
    arrow(s, 5.35, y + 0.42, 5.7, y + 0.42, INK3)
    text(s, 5.8, y + 0.12, 2.4, 0.7, b_, size=13)
panel(s, 8.7, 2.2, 4.0, 3.85, PANEL2)
text(s, 8.95, 2.4, 3.6, 3.5, ["The belief changes what v51 prefers.", "It never changes what v51 is allowed to do: every move "
                              "still passes the safety test of idea 3.", "A wrong guess costs some efficiency, never the general."],
     size=16, after=14)

# --- idea 8: attack ---------------------------------------------------------------
s = new("Idea 8. From part 1, attacking a full general rarely works, so v51 attacks only when it can prove the attack "
        "works. The attack must arrive bigger than the enemy general plus its growth plus the biggest stack the enemy "
        "could bring home; our own general must stay safe while the stack is away; and the attack must win in imagined "
        "games against four replies. In practice this fires when the enemy general has been emptied, as here at tick 405, "
        "where the enemy general holds 2.")
header(s, "Idea 8 · Attacking", "Attack only when the attack is proven")
chips(s, 1.5, "ch. 4 AND-OR search · ch. 6 games")
img(s, SH / "t405.png", 7.3, 1.45, h=5.4)
text(s, 0.7, 2.15, 6.3, 4.6, ["A stack may attack the enemy general only if:",
                              [("1  ", {"color": BRASS, "bold": True}), ("it arrives bigger than the enemy general plus its growth plus the biggest stack the enemy could bring home;", {})],
                              [("2  ", {"color": BRASS, "bold": True}), ("our own general stays safe while the stack is away;", {})],
                              [("3  ", {"color": BRASS, "bold": True}), ("it wins in imagined games against four replies: rush us, rush with a castle, do nothing, pull everything home.", {})],
                              "In practice this fires when the enemy general has been emptied. Against a careful opponent it rarely fires; that is v51's main weakness.",
                              [("Right: ", {"color": INK2}), ("tick 405. The enemy general (gold box) holds 2.", {"color": INK2})]],
     size=15, after=10)

# --- papers -----------------------------------------------------------------------
s = new("The six papers in one place. For each: its idea, what we took from it, and where it sits in v51. Two of them "
        "shape the most behaviour: Human.exe's gathering gives both the safety score and the fast defence, and Puppet "
        "Search gives the plan lookahead.")
header(s, "2 · How v51 thinks", "The six papers behind v51")
rows = [("Straka, Lisý & Schmid 2026", "Human.exe gathers army with a prize-collecting Steiner tree", "the same tree rooted at our general; the delivery estimate for both sides", "ideas 2, 4"),
        ("Barriga, Stanescu & Buro 2015", "Puppet Search: search over script choices, not raw moves", "six plans played 40 ticks ahead against four imagined opponents", "idea 5"),
        ("Synnaeve & Bessière 2011", "Bayesian prediction of the opponent's strategy", "a belief over ten styles, updated each tick with fading memory", "idea 1"),
        ("Weber & Mateas 2009", "a classifier on game features predicts strategy early", "a neural network on 14 board features as the observation model", "idea 1"),
        ("Ganzfried & Sandholm 2015", "exploit a guessed opponent only as far as is safe", "the belief changes preferences; the safety test stays on", "idea 7"),
        ("Churchill & Buro 2011", "plan the economy by projecting income forward", "the income race that decides whether castles are worth it", "idea 6")]
for x_, h_ in zip((0.7, 3.75, 7.05, 11.6), ("Paper", "Its idea", "What we took", "Where")):
    text(s, x_, 1.6, 3, 0.3, h_, size=12, color=BRASS, bold=True)
for i, (p_, idea, took, where) in enumerate(rows):
    y = 2.0 + i * 0.8
    panel(s, 0.6, y, 12.2, 0.7, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.75, y + 0.18, 2.9, 0.5, p_, size=13, font=HEAD)
    text(s, 3.75, y + 0.1, 3.2, 0.6, idea, size=12.5, color=INK2)
    text(s, 7.05, y + 0.1, 4.4, 0.6, took, size=12.5)
    text(s, 11.6, y + 0.18, 1.2, 0.5, where, size=12.5, color=BRASS)

# =========================================================================== #
# part 3: one game
# =========================================================================== #
section("3", "One game, step by step", "v51 (red) against a bot that plays the way people beat our earlier versions: "
        "grab land until tick 200, sweep everything into one stack, charge.", "t213.png",
        "Part 3 replays one game. Every screenshot is the real board, with v51's reasoning drawn on top.")

walk_slide(40, "Opening: one tile per move", ["Both sides expand one tile per move. Nothing threatens either general, "
                                              "so v51's plan is simply to expand.", "The belief is still unsure: early boards look alike for every style."],
           [("21", "v51 army", RED), ("21", "opponent army", BLUE)], "t40.png",
           "Tick 40. Nothing to defend; both sides expand at the same rate.")
walk_slide(205, "The sweep starts", ["The opponent's general is filling up: it is pulling its land in.",
                                     "To v51 this still looks like an expander, because before its sweep this bot is an expander."],
           [("61", "our general", RED), ("79", "their general", BLUE)], "t205.png",
           "Tick 205. The opponent starts collecting. The belief still says expander, which is reasonable from the board.",
           belief="belief205.png")
walk_slide(213, "It empties its general", ["Their general drops from 93 to 1, and a 94-army stack appears (gold boxes).",
                                           "The safety score falls to 12: an attack is now possible soon."],
           [("94", "their new stack", BLUE), ("12", "safety score", BRASS)], "t213.png",
           "Tick 213. The whole army is now one stack. The safety score reacts at once.")
walk_slide(215, "The belief changes", ["The board now matches the late gather-rusher. The belief jumps to 97%.",
                                       "v51 makes its first defensive move."],
           [("103", "attacker", BLUE), ("66", "our general", RED)], "t215.png",
           "Tick 215. Ten ticks after reading expander, the belief reads late gather-rusher.", belief="belief215.png")
walk_slide(219, "Approach confirmed", ["The stack has closed 2 tiles in a few ticks: it is incoming, "
                                                        "18 tiles away.", "v51 measures the shortfall and plans 14 moves into "
                                                        "its general (brass arrows). From now on every move is a gather move."],
           [("111", "attacker", BLUE), ("18", "tiles away", BRASS), ("68", "our general", RED)], "t219_plan.png",
           "Tick 219. The fast defence takes over: a 14-move Steiner-tree plan into the general.")
walk_slide(224, "Gathering, leaf first", ["Army flows inward from the far edges first, so it snowballs on the way in.",
                                          "The attacker is shrinking as it walks."],
           [("119", "attacker", BLUE), ("13", "tiles away", BRASS), ("71", "our general", RED)], "t224.png",
           "Tick 224. Nine moves of the plan left.")
walk_slide(235, "Merged", ["The big stack next to the general steps in. The general holds 124.",
                           "The attacker is next to the general with 90 (gold boxes)."],
           [("124", "our general", RED), ("90", "attacker", BLUE)], "t235.png",
           "Tick 235. The general went from 68 to 124 in 16 ticks.")
walk_slide(237, "The attack lands, and fails", ["The attack arrives with 88; the general keeps 38.",
                                                "The previous version, v46, lost this exact position at tick 237 with 77 against 82."],
           [("38", "general after the hit", RED), ("0", "attackers left", BLUE)], "t237.png",
           "Tick 237. Absorbed. The same position cost v46 the game.")
walk_slide(407, "The proven attack", ["v51 went back to castles (first one at tick 298) and pulled ahead in army.",
                                      "At tick 405 the enemy general held 2. The attack passed all four imagined replies; "
                                      "the general is captured at tick 407."],
           [("1,099", "v51 army", RED), ("772", "opponent army", BLUE)], "t407.png",
           "Tick 407. Capture.")

s = new("The whole game on one chart: the sweep, the absorbed attack, the first castle, the capture.")
header(s, "3 · One game", "The whole game")
img(s, ART / "army_game.png", 0.7, 1.6, w=8.0)
text(s, 9.1, 1.8, 3.6, 4.6, ["Before the sweep the two armies grow together.", "The attack costs both sides army; v51 "
                             "keeps its general.", "After the first castle at tick 298, v51's army pulls away until the capture."],
     size=15, after=12)

# =========================================================================== #
# part 4
# =========================================================================== #
section("4", "Results, then you play", "How v51 does against other bots, and what to look for in the live game.", "t237.png",
        "Part 4: results, then the live game.")

s = new("Results from simulator games with seats swapped on every map. The collector rows are the strategy people used "
        "against our earlier bots, and they are where v51 gains most. These rows were measured with v51's defence "
        "without the classifier (we called that build v52); the defence code is the same. Head to head with v46 it is "
        "level, 32 to 28 over 60 games, and it is slightly worse against the spread-and-collect player.")
header(s, "4 · Results", "Where v51 is stronger")
img(s, ART / "results.png", 0.6, 1.6, w=8.4)
text(s, 9.3, 1.75, 3.4, 4.9, [[("Collectors: ", {"bold": True, "color": BRASS}), ("the strategy people used to beat earlier versions. v51 gains most here.", {})],
                              [("Head to head with v46: ", {"bold": True, "color": BRASS}), ("32–28 over 60 games; level.", {})],
                              [("Weak spot: ", {"bold": True, "color": BRASS}), ("the spread-and-collect player, which takes castles while v51 defends.", {})],
                              [("20 games per bar, both seats. Defence measured without the classifier.", {"color": INK3, "size": 12})]],
     size=14.5, after=12)

s = new("The live game. Launch the arena, pick the champion. Tell the professor what to watch for: one tile per move in "
        "the opening; the moment a big stack heads for v51's general, v51 stops expanding and pulls its army in from the "
        "edges; it does not attack a full general; spreading out and taking castles is the game where it is weakest.")
header(s, "4 · Live game", "Play v51")
img(s, SH / "t120.png", 7.3, 1.45, h=5.4)
panel(s, 0.7, 1.65, 6.3, 0.75, PANEL2)
text(s, 0.95, 1.85, 6.0, 0.5, "python play_arena.py   then pick  Final · champion", size=13, font="Menlo", color=BRASS)
text(s, 0.7, 2.7, 6.3, 4.2, ["What to watch for:",
                             [("Opening. ", {"bold": True, "color": BRASS}), ("One tile per move; the general is left alone.", {})],
                             [("Send a big stack at its general. ", {"bold": True, "color": BRASS}), ("v51 stops expanding and pulls army in from the edges.", {})],
                             [("Keep your general full. ", {"bold": True, "color": BRASS}), ("v51 will not attack it; it waits for it to be emptied.", {})],
                             [("Spread out and take castles. ", {"bold": True, "color": BRASS}), ("This is the game where v51 is weakest.", {})]],
     size=15.5, after=12)

s = new("One paragraph of history, only if asked. v51 is the fifty-first version. The early versions searched deeper and "
        "did not get stronger, because they did not model the event that lost games. The pieces you saw came in this "
        "order.")
header(s, "Backup", "How we got here")
steps = [("v1–v35", "deeper search: weighted A*, ProbCut, matrix games, scripts", "no stronger"),
         ("v36", "the safety rule and the play-it-out test", "20 of 20 vs rushers"),
         ("v37", "plans with a 40-tick lookahead", "became the champion"),
         ("v42–v44", "safety score from collection; castles; reserve", "19–1 vs v37"),
         ("v46", "belief about the opponent", "17–3 vs v37"),
         ("v51", "fast defence and the trained classifier", "19–1 vs the human-style bot")]
for i, (v_, d_, r_) in enumerate(steps):
    y = 1.75 + i * 0.82
    panel(s, 0.7, y, 11.9, 0.7, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.95, y + 0.17, 1.6, 0.4, v_, size=16, font=HEAD, color=BRASS)
    text(s, 2.7, y + 0.2, 6.6, 0.4, d_, size=14)
    text(s, 9.5, y + 0.2, 3.0, 0.4, r_, size=13, color=INK2)

s = new("References.")
header(s, "References", "Book and papers")
text(s, 0.7, 1.7, 11.9, 5.0, ["Russell, S. & Norvig, P. Artificial Intelligence: A Modern Approach, 4th ed. Pearson, 2020.",
                              "Straka, M., Lisý, V. & Schmid, M. Superhuman AI for Generals.io using self-play reinforcement learning. arXiv 2606.23348, 2026.",
                              "Barriga, N., Stanescu, M. & Buro, M. Puppet Search: enhancing scripted behavior by look-ahead search with applications to RTS games. AIIDE 2015.",
                              "Synnaeve, G. & Bessière, P. A Bayesian model for opening prediction in RTS games with application to StarCraft. AIIDE 2011.",
                              "Weber, B. & Mateas, M. A data mining approach to strategy prediction. CIG 2009.",
                              "Ganzfried, S. & Sandholm, T. Safe opponent exploitation. ACM Transactions on Economics and Computation, 2015.",
                              "Churchill, D. & Buro, M. Build order optimization in StarCraft. AIIDE 2011."],
     size=14, color=INK2, after=10)

prs.core_properties.title = "Building a game agent from the textbook: v51"
prs.save(str(OUT))
print("saved", OUT, N[0])
