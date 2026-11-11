"""v51 talk, plain-language edition with full speaker notes. Reuses the helpers
(theme, layout functions) from build.py."""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
_src = (HERE / "build.py").read_text().split("# 1 opening")[0]
_src = _src.replace('("tick ", {"size": 18', '("turn ", {"size": 18').replace('"ONE GAME, STEP BY STEP"', '"ONE REAL GAME"')
exec(compile(_src, "build_helpers", "exec"))


def chips(s, y, book, paper=None, x=0.7, stack=False):  # noqa: F811
    """One short line each: what we used from the book, and from which paper."""
    x2 = chip(s, x, y, "FROM THE BOOK", book, BRASS)
    if paper:
        if stack:
            chip(s, x, y + 0.45, "FROM THE PAPER", paper, BLUE)
        else:
            chip(s, x2, y, "FROM THE PAPER", paper, BLUE)


def term(s, x, y, w, h, name, expl, say=None):
    """A box naming the technical term behind the idea, with a plain explanation;
    the same explanation is added to the speaker notes."""
    rect(s, x, y, w, h, PANEL2, BRASS, MSO_SHAPE.ROUNDED_RECTANGLE, 0.08)
    text(s, x + 0.2, y + 0.12, w - 0.4, h - 0.2, [[("TERM  ", {"bold": True, "color": BRASS, "size": 11}),
                                                   (name, {"bold": True, "color": INK, "size": 14})],
                                                  [(expl, {"color": INK2, "size": 12.5})]], after=3)
    tf = s.notes_slide.notes_text_frame
    tf.text = tf.text + "\n\nTERM ON THIS SLIDE, " + name.upper() + ": " + (say or expl)


def concept(n, title, idea, paper, maths, example, note, label=None):
    """The concept behind an idea, explained simply, with its paper and its maths,
    before the slides that show how v51 uses it."""
    s = new(note)
    header(s, label or f"Idea {n} · The concept", title)
    text(s, 0.7, 1.65, 5.7, 2.6, idea, size=16, after=10)
    if paper:
        name, did = paper[:2]
        label, col = (paper[2], BRASS) if len(paper) > 2 else ("THE PAPER", BLUE)
        panel(s, 0.7, 4.25, 5.7, 2.6, PANEL2)
        rect(s, 0.7, 4.25, 0.08, 2.6, col)
        text(s, 0.95, 4.4, 5.3, 0.3, label, size=11, color=col, bold=True)
        text(s, 0.95, 4.7, 5.3, 0.6, name, size=15, font=HEAD)
        text(s, 0.95, 5.3, 5.3, 1.5, did, size=13.5, color=INK2)
    y = 1.6
    for words, e in maths:
        text(s, 6.8, y, 5.9, 0.4, words, size=13.5, color=INK2)
        pic = eq(s, e, 6.85, y + 0.38, h=0.36, maxw=5.7)
        y = y + 0.38 + pic.height / 914400 + 0.22
    if example:
        words, e = example
        ey = max(y + 0.05, 5.05)
        panel(s, 6.7, ey, 6.0, 6.95 - ey, PANEL2)
        text(s, 6.9, ey + 0.12, 5.6, 0.6, [[("EXAMPLE  ", {"bold": True, "color": BRASS, "size": 11}), (words, {"color": INK2, "size": 13})]])
        eq(s, e, 6.95, ey + 0.82, h=0.34, maxw=5.5)
    return s



# =========================================================================== #
s = new("""SAY: Good morning. You've just watched the tutorial and seen the arena, so you already know the game.

In the next fifteen minutes I'll show you how our bot, v51, decides what to do, using one real game it played. Then the professor gets to play it.

The idea of the project was simple: take the methods from the textbook, search, probability, decision making and a bit of learning, and see how good a game player you can build with them.""")
img(s, SH / "t235.png", 6.9, 0, h=7.5)
rect(s, 6.75, 0, 0.5, 7.5, BG)
text(s, 0.8, 1.75, 6, 0.35, "DAC-203 · ARTIFICIAL INTELLIGENCE · IIT ROORKEE", size=12, color=BRASS, bold=True)
text(s, 0.8, 2.25, 5.9, 2.2, ["How our bot", "plays Generals"], size=44, font=HEAD, after=0, spacing=1.0)
text(s, 0.8, 4.15, 5.8, 1.5, ["v51: a game agent built from the textbook and six research papers."], size=18, color=INK2)
text(s, 0.8, 6.3, 5.8, 0.4, "Right: a real moment from a game. Our bot is red.", size=12, color=INK3, italic=True)

# --- game recap --------------------------------------------------------------
s = new("""SAY: A quick recap of what you just saw in the tutorial.

Red is our bot, blue is the opponent. The number on a square is how many soldiers stand there. The crown is the general. Lose it and you lose the game.

Each turn you move one stack of soldiers one square. Both players move at the same time, so you never get to react to the other player's move first.

And soldiers grow by themselves. The general and castles grow fast, ordinary land grows slowly. Keep that in mind, because the whole bot is built around it.""")
header(s, "Quick recap", "The game in thirty seconds")
img(s, SH / "t120.png", 0.7, 1.55, h=5.3)
for i, (h_, d_) in enumerate([("Numbers are soldiers", "Red is our bot, blue is the opponent."),
                              ("Lose your general, lose the game", "The crown marks each player's general."),
                              ("One move per turn, both at once", "Move one stack one square. Fights subtract."),
                              ("Soldiers grow on their own", "Generals and castles fast, land slowly.")]):
    y = 1.75 + i * 1.25
    panel(s, 6.5, y, 6.2, 1.05)
    text(s, 6.75, y + 0.16, 5.8, 0.4, h_, size=18, font=HEAD, color=BRASS)
    text(s, 6.75, y + 0.58, 5.8, 0.4, d_, size=14.5, color=INK2)

s = new("""SAY: This slide is the key to everything that follows.

A stack of 30 walks toward a general that has 12. Every square it walks over, it has to leave one soldier behind. So after 8 squares it has 22 left.

Meanwhile the general has been sitting still and growing. It now has 16.

So attacking a general that keeps its soldiers at home usually fails. A general only falls in two situations. One: its owner moved the soldiers away. Two: several stacks arrive one after another, and together they are too much.

Our bot watches for exactly these two things, both in attack and in defence.""")
header(s, "The key idea", "Walking costs soldiers, waiting earns them")
text(s, 0.7, 1.6, 4, 0.3, "start", size=13, color=INK2)
img(s, SH / "strip0.png", 0.7, 1.95, w=11.9)
text(s, 0.7, 3.25, 4, 0.3, "8 turns later", size=13, color=INK2)
img(s, SH / "strip8.png", 0.7, 3.6, w=11.9)
stat(s, 0.7, 4.95, "30 → 22", "the attacker leaves a soldier on every square", RED, w=3.6)
stat(s, 4.6, 4.95, "12 → 16", "the general grows while it waits", BLUE, w=3.6)
panel(s, 8.4, 4.85, 4.3, 1.75, PANEL2)
text(s, 8.6, 5.0, 3.95, 1.5, ["A general falls only when its soldiers have left, or when several stacks hit it one after another."],
     size=16, color=INK)

exec(compile((HERE / "pivot_slides.py").read_text(), "pivot_slides", "exec"))

# =========================================================================== #
section("2", "How v51 decides a move", "Eight ideas. For each: the concept and its paper first, then how v51 uses it.",
        "t219_plan.png", """SAY: Now, how does the bot decide what to do each turn? I'll go through eight ideas. For each one I'll say what it does, and where we got it from: the textbook, a research paper, or both.""")

s = new("""SAY: Every turn, half a second, the bot goes through the same steps.

First it looks at the board. Then it updates its guess about what kind of player it is facing.

Then it asks: is an attack coming at my general? If yes, it drops everything and defends. If not, it picks a plan by imagining how the next forty turns would go.

And whatever move it picks, it tests it first. If the test says the move could lose the general, the move is never played.

The next slides go through these steps one by one.""")
header(s, "2 · How v51 decides", "What happens every turn")
steps = [("Look", "read the whole board"), ("Guess", "what kind of player is the opponent?"),
         ("Check", "is an attack coming?"), ("Decide", "defend now, or pick a plan"),
         ("Test", "could this move lose the general?"), ("Move", "one move, every half second")]
for i, (h, d) in enumerate(steps):
    x = 0.7 + i * 2.08
    panel(s, x, 2.3, 1.85, 2.3, PANEL if i not in (3, 4) else PANEL2)
    text(s, x + 0.15, 2.45, 1.6, 0.5, f"{i + 1}", size=26, font=HEAD, color=BRASS)
    text(s, x + 0.15, 3.05, 1.6, 0.5, h, size=19, font=HEAD)
    text(s, x + 0.15, 3.6, 1.6, 1.0, d, size=13.5, color=INK2)
    if i < len(steps) - 1:
        arrow(s, x + 1.86, 3.45, x + 2.07, 3.45, INK3)
text(s, 0.7, 4.95, 11.9, 0.8, "Two rules never change: a move that fails the test is never played, and when an attack is "
     "coming, defence gets every move until it is over.", size=16, color=INK2)
term(s, 0.7, 5.85, 5.85, 1.05, "Utility-based agent (AIMA ch. 2)",
     "An agent that picks actions by estimating how good the result will be, instead of following fixed rules.",
     "In the book's terms this is a utility-based agent: it doesn't follow fixed if-then rules, it estimates how good each outcome would be and picks the best.")
term(s, 6.75, 5.85, 5.85, 1.05, "Simultaneous-move game (AIMA ch. 18)",
     "Both players commit before seeing the other's move, so each turn is closer to rock-paper-scissors than to chess.",
     "And it's a simultaneous-move game: both players commit at the same moment, so you can't wait and react. That's why plain chess-style minimax doesn't fit directly.")

# --- idea 1 ----------------------------------------------------------------------
concept(1, "Updating a belief with evidence",
        ["You hold a probability for each explanation. New evidence raises the explanations that predicted it and lowers the others. That is Bayes' rule.",
         "Do it again every turn, keeping what you believed before, and you get a filter: a belief that follows the game."],
        ("Synnaeve & Bessière, AIIDE 2011", "Predicted a StarCraft player's opening strategy with a Bayesian model updated from what the player had built so far. Weber & Mateas (CIG 2009) trained classifiers on the same kind of features."),
        [("Bayes' rule: belief in y after seeing evidence e", "bayes"),
         ("Repeated every turn: today's belief from yesterday's", "filter"),
         ("A classifier turns measurements x into probabilities", "softmax")],
        ("(illustrative) two styles, 50/50; the general empties: 90% likely for a builder, 5% for a land grabber", "bayes_ex"),
        """SAY: Before I show how the bot guesses, here's the idea underneath, in its simplest form.

You hold a probability for each explanation. When evidence arrives, the explanations that predicted it go up and the others go down. That's Bayes' rule, the first formula: belief after the evidence equals how well y predicted the evidence, times what we believed before, divided by a normaliser.

Example at the bottom: say we think the opponent is 50/50 a land grabber or a builder. Then its general suddenly empties. A builder does that 90 percent of the time, a land grabber 5 percent. Plug in: 95 percent builder after one observation.

Do that every turn, carrying the old belief forward, and you get a filter, the second formula, from chapter 14.

The third formula is how a classifier, our small neural network, produces those probabilities: it computes a score for each style and the softmax turns the scores into probabilities that add up to one.

The paper: Synnaeve and Bessière predicted StarCraft strategies exactly this way; Weber and Mateas trained classifiers on game features.""")

s = new("""SAY: Idea one: guess what kind of player you are facing.

The bot can see the whole board, but it can't see the opponent's plan. So it keeps a guess. It knows ten playing styles, for example: rushes early, builds up and then attacks, grabs land and castles.

Every turn it measures a few things about the opponent: how much land it has, how much of its army sits in its general, how big its biggest stack is, and whether that stack is coming closer. A small neural network turns those numbers into a probability for each style.

Look at the two bar charts. At turn 205 the opponent looks like a player that only grabs land. Ten turns later it has pulled everything into one big stack, and the bot now says 97 percent: this player builds up and then attacks.

On new maps it guesses right about 70 percent of the time by the middle of the game. Random guessing would be 10 percent.""")
header(s, "Idea 1", "Guess what kind of player you're facing")
chips(s, 1.5, "updating beliefs with probability", "Synnaeve & Bessière 2011; Weber & Mateas 2009")
text(s, 0.7, 2.2, 6.0, 3.6, ["The bot can see the whole board, but not the opponent's plan. So it keeps a guess over ten "
                             "playing styles, like \"rushes early\" or \"builds up, then attacks\".",
                             "Each turn it measures a few things about the opponent: its land, how much army sits in its "
                             "general, how big its biggest stack is, and whether that stack is getting closer.",
                             "A small neural network turns those measurements into a probability for each style."],
     size=15, after=10)
panel(s, 7.2, 2.15, 5.5, 4.6)
text(s, 7.45, 2.3, 5, 0.3, "turn 205", size=13, color=INK2, bold=True)
img(s, ART / "belief205.png", 7.45, 2.6, h=1.75)
text(s, 7.45, 4.45, 5, 0.3, "turn 215, after it pulls everything into one stack", size=13, color=INK2, bold=True)
img(s, ART / "belief215.png", 7.45, 4.75, h=1.75)
term(s, 0.7, 5.2, 6.0, 1.05, "Classifier (AIMA ch. 19, 21)",
     "A model that sorts an input into one of several categories. Here a small neural network sorts the opponent's stats into 10 styles.",
     "The network is a classifier: it takes some numbers and sorts them into one of a fixed set of categories, here ten playing styles. We trained it on games against each style.")
text(s, 0.7, 6.4, 6.0, 0.5, "Right about 70% of the time by mid-game on new maps. Random guessing: 10%.", size=13.5, color=BRASS)

s = new("""SAY: How does the guess change over time? One board on its own is weak evidence, so the bot doesn't throw away what it believed before.

In words: today's guess is mostly yesterday's guess, nudged a little toward what today's board suggests. That's the formula: 0.95 of the old evidence plus a bit of the new.

The 0.95 means old evidence slowly fades, so if the opponent changes style halfway, the bot notices.

In the chart you can see it: the bot thinks "land grabber" until turn 200, which is fair, because until then this opponent really is just grabbing land. The moment it pulls everything together, the guess flips.

This is the "update your belief as evidence arrives" idea from the probability chapters of the book.""")
header(s, "Idea 1", "The guess changes as the game goes on")
text(s, 0.7, 1.6, 6.3, 0.7, "In words: today's guess = mostly yesterday's guess, nudged toward what today's board suggests.",
     size=16)
panel(s, 0.7, 2.45, 6.3, 1.0, PANEL2)
eq(s, "belief", 0.95, 2.67, h=0.42, maxw=5.9)
text(s, 0.7, 3.65, 6.3, 2.5, [[("ℓ(y)", {"bold": True, "color": BRASS}), ("  how much evidence we have that the opponent plays style y", {})],
                              [("P(y | x)", {"bold": True, "color": BRASS}), ("  what the network thinks after seeing today's board x", {})],
                              [("0.95", {"bold": True, "color": BRASS}), ("  old evidence fades slowly, so a change of style is noticed", {})]],
     size=15, after=10)
term(s, 0.7, 5.3, 6.3, 1.2, "Bayesian filtering (AIMA ch. 14)",
     "Keep a probability for each possibility and update it with every new observation, using Bayes' rule. Here old evidence also fades slowly.",
     "This is Bayesian filtering from chapter 14: you hold a probability for each possibility, and every new observation updates it with Bayes' rule. Ours also forgets slowly, so it can follow an opponent that changes style.")
img(s, ART / "belief_game.png", 7.25, 1.6, w=5.5)
text(s, 7.25, 4.4, 5.4, 2.0, "Until turn 200 the guess is \"land grabber\", which is correct: before its attack, this "
     "opponent only grabs land. At turn 215 the guess flips to \"builds up, then attacks\".", size=14.5, color=INK2)

# --- idea 2 ----------------------------------------------------------------------
concept(2, "Packing the most into a limited budget",
        ["You have several items, each with a value and a cost, and a fixed budget. Which items do you take? This is the knapsack problem.",
         "Here an item is a stack walking to our general: its value is the soldiers it delivers, its cost is the moves it takes. The budget is the number of turns."],
        ("Straka, Lisý & Schmid, 2026 (Human.exe)", "Describes Human.exe, the strongest hand-written Generals bot. It decides how to gather its army by solving exactly this kind of budgeted collection problem."),
        [("The knapsack problem: take items i to maximise value within budget W", "knap"),
         ("Meanwhile the general grows by one every two turns", "growth")],
        ("three enemy stacks bring 21, 16 and 15 in 25 moves; our general has 40", "knap_ex"),
        """SAY: Idea two rests on a classic problem: the knapsack. You have items with a value and a cost, and a budget. Which ones do you take?

For us, an item is an enemy stack walking toward our general. Its value is how many soldiers it delivers, its cost is how many moves it takes, and the budget is the number of turns. The first formula says: choose the items, x_i is one if we take item i, to maximise total value without going over the budget.

The second formula is the defender's side: the general gains one soldier every two turns.

Example: three enemy stacks would deliver 21, 16 and 15 soldiers in 25 moves in total, 52. Our general has 40 and grows by 12 in 25 turns, also 52. That's a tie, which means danger.

The knapsack problem is NP-hard: no fast exact method is known. So we use a greedy rule, biggest item first.

The paper: Human.exe, the best hand-written Generals bot, plans its gathering as this kind of budgeted collection problem.""")

s = new("""SAY: Idea two: is my general safe right now?

Remember the danger: several stacks arriving one after another. So every turn the bot asks: if the enemy started walking its stacks at my general right now, how many soldiers could it get there? And how many would my general have by then, counting the stacks I could bring home in time?

The light-blue line on the board is the enemy's route to our general at turn 219. Its big stack has 111 soldiers and is 18 squares away.

The difference between the two numbers is a safety score. When it gets small, the bot knows an attack is possible soon.

We got this from a paper about Human.exe, the strongest hand-written Generals bot. It uses this counting to gather its own army. We use the same counting on the enemy's army, to measure the threat.""")
header(s, "Idea 2", "Is my general safe?")
chips(s, 1.5, "searching for shortest paths", "Straka, Lisý & Schmid 2026 (Human.exe)", stack=True)
img(s, SH / "t219_walks.png", 7.3, 1.45, h=5.4)
text(s, 0.7, 2.6, 6.2, 4.3, ["Every turn the bot compares two numbers:",
                             [("Threat: ", {"bold": True, "color": BLUE}), ("how many enemy soldiers could reach my general if the enemy started walking its stacks in now.", {})],
                             [("Defence: ", {"bold": True, "color": RED}), ("how many my general will have by then, plus what my own stacks can bring home in time.", {})],
                             "Defence minus threat is the safety score. Small means danger.",
                             [("Right: the enemy's route (light blue) to our general (gold box). 111 soldiers, 18 squares away.", {"color": INK2, "size": 14})]],
     size=16, after=12)

s = new("""SAY: Here is the same idea written as a formula, and what it looked like during the game.

In words: what my general and my nearby stacks can have there, minus what the enemy can bring, at the worst possible moment. "Worst possible moment" because the bot checks every time window, ten turns, twenty turns and so on, and keeps the worst one.

The chart shows the safety score during the game. It sits around 40, then at turn 213 it falls to 12. That's the moment the opponent emptied its general into one big stack. The grey band is where the bot switches to defending, and the score climbs back up.""")
header(s, "Idea 2", "The safety score during the game")
text(s, 0.7, 1.6, 12, 0.4, "In words: what my side can have at the general, minus what the enemy can bring, at the worst moment.",
     size=16)
panel(s, 0.7, 2.15, 11.9, 1.0, PANEL2)
eq(s, "safety", 0.95, 2.37, h=0.42, maxw=11.4)
text(s, 0.7, 3.4, 6.3, 3.0, [[("G", {"bold": True, "color": BRASS}), ("  soldiers on our general now", {})],
                             [("growth(T)", {"bold": True, "color": BRASS}), ("  what the general gains in T turns", {})],
                             [("D_us, D_them", {"bold": True, "color": BRASS}), ("  soldiers each side can walk to our general in T turns", {})],
                             [("lead", {"bold": True, "color": BRASS}), ("  our stacks must arrive a little early", {})]], size=15, after=9)
term(s, 0.7, 5.35, 6.3, 1.3, "NP-hard, solved greedily",
     "Picking the best stacks to walk in a limited number of moves is NP-hard: no fast exact method is known. A greedy algorithm takes the biggest stack first, then the next.",
     "Working out exactly how many soldiers can be gathered in T moves is NP-hard, a knapsack-style problem with no known fast exact solution. So we use a greedy algorithm: take the biggest stack first, then the next, until the moves run out. It's fast and close enough.")
img(s, ART / "safety_game.png", 7.35, 3.35, w=5.3)
text(s, 7.35, 5.85, 5.3, 1.0, "At turn 213 the opponent empties its general into one stack. The score drops from 26 to 12.",
     size=14, color=INK2)

# --- idea 3 ----------------------------------------------------------------------
concept(3, "Planning against every possible reply",
        ["Some problems have two kinds of choice. At an OR point we pick one option, and one good option is enough. At an AND point the world picks, and our plan must work for every option.",
         "A move is safe only if it survives every attack the enemy could make. The tree of choices with these two kinds of point is called an AND-OR tree."],
        ("Russell & Norvig, AIMA chapter 4", "Introduces AND-OR trees for planning when the outcome of an action is not under our control: a plan counts only if it works for every outcome.", "FROM THE BOOK"),
        [("A move is accepted only if the general survives every reply b", "andor"),
         ("Scoring the tree: we take the best at OR points, the worst at AND points", "orand")],
        ("three possible enemy attacks; the general survives two of them", "andor_ex"),
        """SAY: Idea three comes straight from the textbook, chapter 4.

Some problems have two kinds of choice. At an OR point, we choose, and we only need one option that works. At an AND point, the environment or the opponent chooses, and our plan has to work no matter what it picks.

The first formula is the rule the bot uses: accept a move only if, for all enemy replies b, the general survives.

The second formula is how such a tree is scored: take the best value at our choices, the worst value at the opponent's choices. That's the same idea as minimax in chess, but we only use it as a yes or no safety test.""")

s = new("""SAY: Idea three: never make a move that could lose the general.

Before every move, the bot tries it on a copy of the board. If the safety score stays healthy, fine.

If not, it plays the move out in its head. It imagines the enemy attacking along each likely route, with everything it has, and our bot defending. If the general falls in even one of those imagined games, the move is thrown away.

So the bot only needs one bad outcome to reject a move. In the textbook this is called an AND-OR search: we choose our move, but the move has to survive every one of the enemy's options.""")
header(s, "Idea 3", "Test every move before making it")
chips(s, 1.5, "planning against every possible reply (AND-OR search)")
panel(s, 0.7, 2.15, 6.0, 4.6)
rect(s, 2.6, 2.4, 2.2, 0.75, PANEL2, BRASS, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
text(s, 2.6, 2.58, 2.2, 0.5, "our move", size=15, align=PP_ALIGN.CENTER, color=BRASS)
for i, (lab, ok) in enumerate([("enemy attack 1", True), ("enemy attack 2", True), ("enemy attack 3", False)]):
    x = 0.95 + i * 1.9
    arrow(s, 3.7, 3.15, x + 0.8, 3.85, INK3)
    rect(s, x, 3.85, 1.6, 0.7, PANEL2, BLUE, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
    text(s, x, 3.98, 1.6, 0.5, lab, size=12.5, align=PP_ALIGN.CENTER)
    arrow(s, x + 0.8, 4.55, x + 0.8, 5.15, INK3)
    rect(s, x, 5.15, 1.6, 0.7, RED if not ok else "2E4A3A", None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
    text(s, x, 5.3, 1.6, 0.5, "general safe" if ok else "general lost", size=12.5, align=PP_ALIGN.CENTER)
text(s, 0.95, 6.1, 5.5, 0.5, "One loss is enough to throw the move away.", size=14, color=INK2, align=PP_ALIGN.CENTER)
text(s, 7.2, 2.2, 5.5, 4.6, ["1.  Try the move on a copy of the board. If the general is still safe, play it.",
                             "2.  If not, imagine the enemy attacking along each likely route, with our bot defending.",
                             "3.  If the general falls in any of these imagined games, never play the move."],
     size=16.5, after=16)
term(s, 7.2, 5.0, 5.5, 1.4, "AND-OR search (AIMA ch. 4)",
     "A search tree with two kinds of node. At OR nodes we choose one option; at AND nodes every option must work. A move is good only if it survives all enemy attacks.",
     "This is AND-OR search from chapter 4. Our move is an OR node: we just need one good choice. The enemy's possible attacks are an AND node: our move has to survive all of them, not just most.")

# --- idea 4 ----------------------------------------------------------------------
concept(4, "Collecting as much as possible with a price on effort",
        ["We want a tree of routes, all ending in our general, that collects the most soldiers. Every square on a route costs one move.",
         "Trick: put a price λ on each move. Maximise prize minus price. A high price gives a small, quick tree; a low price gives a big, slow one. Search for the price that makes the tree fit the time left."],
        ("Straka, Lisý & Schmid, 2026 (Human.exe)", "Human.exe gathers its army with this exact structure, a prize-collecting Steiner tree, and searches for the price per move so the plan fits its turn budget."),
        [("Prize-collecting Steiner tree: prize p_v on each square, price λ per move", "pcst"),
         ("Binary search on the price: halve the range each time", "bisect")],
        ("prices between 0 and 120, to within 1", "bis_ex"),
        """SAY: Idea four, the fast defence, uses a structure called a prize-collecting Steiner tree. Let me explain it simply.

Imagine every square we own holds a prize, its soldiers. We want a tree of routes, all ending at our general, that collects the most prize. But every square on a route costs a move, and we only have so many turns.

The trick: put a price, lambda, on each move. Then maximise prize minus price times the number of moves. That's the first formula. If the price is high, only stacks very close to the general are worth fetching. If it's low, the tree reaches far away and takes too long.

So we search for the right price with binary search, the second formula: try the middle of the range, see if the tree fits, keep the half that works. Each try halves the range, so seven tries are enough.

The paper: Human.exe gathers its army with exactly this method. We use it the other way round, to defend.""")

s = new("""SAY: Idea four is the new part in v51, and it's the reason it beats the strategy that people used against our earlier bots.

When a big enemy stack is clearly coming, the bot stops everything else and defends.

First it notices: a big stack that has moved at least two squares closer in the last few turns. How far away it is tells the bot how many turns it has.

Then it measures: when the attacker arrives, how many soldiers short will the general be?

Then it gathers: it uses every remaining move to pull soldiers into the general itself, not next to it.

The gold arrows on the board are the actual plan it made at turn 219: fourteen moves, all ending in the general.""")
header(s, "Idea 4", "When an attack is coming, defend fast")
chips(s, 1.5, "turning a goal into a search problem", "Straka, Lisý & Schmid 2026 (Human.exe)", stack=True)
img(s, SH / "t219_plan.png", 7.3, 1.45, h=5.4)
for i, (h_, d_) in enumerate([("Notice", "A big enemy stack moved 2+ squares closer in the last 5 turns. Its distance = turns left."),
                              ("Measure", "When it arrives, how many soldiers short will the general be?"),
                              ("Gather", "Use every remaining move to pull soldiers into the general.")]):
    y = 2.55 + i * 1.38
    panel(s, 0.7, y, 6.3, 1.22)
    text(s, 0.95, y + 0.15, 0.6, 0.8, str(i + 1), size=30, font=HEAD, color=BRASS)
    text(s, 1.65, y + 0.14, 5.2, 0.4, h_, size=18, font=HEAD)
    text(s, 1.65, y + 0.58, 5.2, 0.8, d_, size=14, color=INK2)
text(s, 0.7, 6.7, 6.3, 0.4, "Right: the 14-move plan at turn 219 (gold arrows), all ending in the general.",
     size=12.5, color=INK3, italic=True)

s = new("""SAY: How does it decide which stacks to pull in? Think of every square we own as holding a small prize: its soldiers. Every step costs one move. We want the set of routes into the general that collects the biggest prize, using no more moves than we have turns left.

That is a hard problem in general. The trick, which we borrowed from Human.exe, is to put a price on each move. With a high price, the bot only collects stacks close to the general. With a low price, it reaches far, maybe too far to finish in time. It tries prices until the plan just fits the time left.

Then it plays the moves from the far end first, so soldiers pile up on the way in, like a snowball. In the screenshot, three moves of the plan are left.""")
header(s, "Idea 4", "Which stacks to pull in, and in what order")
text(s, 0.7, 1.6, 7.3, 0.7, "In words: collect as many soldiers as possible into the general, using no more moves than the "
     "turns left.", size=16)
panel(s, 0.7, 2.45, 7.3, 1.0, PANEL2)
eq(s, "tree", 0.95, 2.7, h=0.40, maxw=6.9)
text(s, 0.7, 3.65, 6.3, 3.2, [[("a_v − 1", {"bold": True, "color": BRASS}), ("  soldiers we can take from square v (one must stay)", {})],
                              [("λ", {"bold": True, "color": BRASS}), ("  a price per move. Try prices until the plan fits the time left.", {})],
                              [("ETA − 2", {"bold": True, "color": BRASS}), ("  turns before the attacker arrives, minus a small safety margin", {})],
                              [("Order: ", {"bold": True, "color": BLUE}), ("start with the farthest stacks, so soldiers pile up on the way in.", {})]],
     size=14.5, after=8)
term(s, 0.7, 5.75, 7.3, 1.15, "Prize-collecting Steiner tree · binary search",
     "A tree of routes into one point that collects the most prize for its cost (NP-hard). Binary search finds the right price per move by halving the range each try: 7 tries.",
     "The formal name is a prize-collecting Steiner tree: a tree of routes into one point, the general, that collects the most prize for the cost. It's NP-hard, so we build it greedily for a given price per move, and find the right price with binary search: halve the range each time, seven tries.")
img(s, SH / "t230.png", 8.3, 1.55, h=5.0)
text(s, 8.3, 6.6, 4.4, 0.4, "Turn 230: three moves of the plan left.", size=12.5, color=INK3, italic=True)

# --- idea 5 ----------------------------------------------------------------------
pt = json.loads((ART / "plan_table.json").read_text())
concept(5, "Choosing when you don't know the opponent",
        ["Too many single moves to search, so search over a few plans instead, each played forward in simulation (a rollout).",
         "Then decide. Average over what we think the opponent is (expected value), or assume the worst (maximin), or mix the two (the Hurwicz criterion)."],
        ("Barriga, Stanescu & Buro, AIIDE 2015 (Puppet Search)", "Searches over the choices inside a few scripted plans instead of over raw moves, which makes look-ahead affordable in real-time strategy games."),
        [("Expected value: average over opponents, weighted by our belief", "ev"),
         ("Maximin: assume the worst opponent", "mm"),
         ("Hurwicz: a mix of the two; ours is half and half", "hurwicz")],
        ("turn 240 of the game", "hur_ex"),
        """SAY: Idea five is about deciding when you don't know who you're up against.

First, you can't search every move, there are tens of thousands. Puppet Search, the paper here, says: search over a few plans instead, and play each one forward in simulation. That forward simulation is called a rollout.

Then you have a table: each plan's chance of winning against each type of opponent. How do you pick?

Expected value, the first formula: average the row, weighted by how likely you think each opponent is.

Maximin, the second: look only at the worst case in each row.

Expected value trusts your guess. Maximin ignores it. The Hurwicz criterion, the third formula, mixes them; we use half and half.

Example from the real game: taking the castle scores 0.70 on average and 0.69 in the worst case, so 0.69. Expanding scores 0.59. The bot takes the castle.""")

s = new("""SAY: Idea five: when nothing urgent is happening, how does it choose what to do?

There are tens of thousands of possible moves each turn, far too many to search one by one. So instead the bot chooses between a handful of plans: keep expanding, take one of the nearby castles, attack, and so on.

For each plan, it imagines the next forty turns, four times, against four imagined opponents: one that rushes, one that takes castles first, one that grabs land, and one that waits. At the end it estimates its chance of winning.

This table is real, from turn 240 of the game. Taking a castle scores 0.69, keeping on expanding scores 0.59, so it takes a castle.

How does it combine the four columns? Half the score is the average, weighted by its guess about the opponent from idea one. The other half is the worst case. So it prefers plans that suit the opponent it thinks it faces, but never plans that would be a disaster if the guess is wrong.

The paper here is Puppet Search: search over a few plans instead of over every move.""")
header(s, "Idea 5", "Pick a plan by imagining the next 40 turns")
chips(s, 1.5, "looking ahead in games; choosing under uncertainty", "Barriga, Stanescu & Buro 2015 (Puppet Search)")
names = {"expand": "keep expanding", "castle241": "take castle A", "castle264": "take castle B",
         "castle354": "take castle C", "rush": "attack now"}
show = [r for r in pt["rows"] if r["plan"] in names]
panel(s, 0.7, 2.15, 7.4, 4.55)
heads = ["plan", "vs rusher", "vs castle\nrusher", "vs land\ngrabber", "vs waiter", "score"]
xs = [0.95, 3.0, 4.05, 5.1, 6.15, 7.15]
for x_, h_ in zip(xs, heads):
    text(s, x_, 2.3, 1.1, 0.5, h_, size=11.5, color=INK2, bold=True)
for i, r in enumerate(show):
    y = 2.95 + i * 0.68
    win = r["plan"] in ("castle241", "castle354")
    if win:
        rect(s, 0.8, y - 0.08, 7.2, 0.6, PANEL2, BRASS, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
    text(s, xs[0], y, 2.0, 0.4, names[r["plan"]], size=14.5)
    for x_, o in zip(xs[1:5], ["rusher", "castle_rusher", "expander", "hold"]):
        text(s, x_, y, 1.0, 0.4, f"{r['vals'][o]:.2f}", size=14, color=INK2, font="Menlo")
    text(s, xs[5], y, 1.0, 0.4, f"{r['score']:.2f}", size=14, color=BRASS if win else INK, font="Menlo", bold=win)
text(s, 0.95, 6.35, 7.0, 0.3, "Real numbers from turn 240: the learned win estimate (section 3) after 40 imagined turns.",
     size=11.5, color=INK3, italic=True)
text(s, 8.5, 2.15, 4.2, 1.4, "In words: half the average (weighted by our guess about the opponent), half the worst case.",
     size=16)
panel(s, 8.5, 3.6, 4.2, 0.95, PANEL2)
eq(s, "plan", 8.65, 3.8, h=0.36, maxw=3.9)
text(s, 8.5, 4.7, 4.2, 0.8, "Good against the opponent we expect, never a disaster if we guessed wrong.",
     size=14, color=INK2)
term(s, 8.5, 5.5, 4.2, 1.45, "Rollouts · Hurwicz criterion",
     "A rollout plays the game forward in simulation. The Hurwicz criterion scores a choice by mixing its average and its worst case; ours is 50/50.",
     "Two terms here. Imagining forty turns ahead is called a rollout: playing the game forward in simulation. And scoring by half average, half worst case is the Hurwicz criterion from decision theory, a middle ground between trusting your guess completely and assuming the worst.")

# --- idea 6 ----------------------------------------------------------------------
concept(6, "Is an investment worth it?",
        ["A castle is an investment: you pay soldiers now, and it produces soldiers until the game ends.",
         "Compare what it will produce with what it costs. And to know whether you need it at all, project both players' armies forward at their current income."],
        ("Churchill & Buro, AIIDE 2011", "Planned StarCraft build orders by simulating each player's economy forward in time, to see which investments pay off and when."),
        [("Income per turn: generals and castles give ½, each square of land 1/50", "income"),
         ("Projected army after L more turns", "project"),
         ("A castle pays if what it produces beats its cost (garrison g plus walk d)", "payback")],
        ("200 turns left, garrison 45, walk 10", "pay_ex"),
        """SAY: Idea six treats a castle as an investment.

You pay soldiers now, the garrison plus the soldiers lost walking there, and the castle produces half a soldier per turn until the game ends.

The first formula is income per turn: generals and castles give a half each, and every square of land gives one fiftieth.

The second projects an army forward: what you have now plus income times the turns left. Comparing both players' projections tells the bot whether it is clearly ahead or not.

The third is the payback rule: half the remaining turns, after the walk, must be more than the cost.

Example: 200 turns left, garrison 45, walk 10. It produces 95, it costs 55. Worth it.

The paper: Churchill and Buro planned StarCraft economies exactly by simulating income forward.""")

s = new("""SAY: Idea six: castles. A castle produces as many soldiers as 25 ordinary squares, so castles win long games. But taking one means sending a big stack away, and that leaves home weaker.

So the bot asks three questions. Is it worth it: will the castle pay back what it costs before the game ends? Is it safe: the bot fast-forwards the board to the moment the stack has been spent on the castle, moves the enemy forward too, and checks the general is still safe at that moment. And is it needed: if we are already clearly ahead, cheap castles only.

The last idea comes from a paper on planning the economy in StarCraft: project both players' income forward to the end of the game.

In the screenshot, turn 298, the bot takes its first castle, after it has dealt with the attack.""")
header(s, "Idea 6", "Castles: only when they pay and home stays safe")
chips(s, 1.5, "weighing costs against expected payoff", "Churchill & Buro 2011", stack=True)
img(s, SH / "t298.png", 7.3, 1.45, h=5.4)
text(s, 0.7, 2.6, 6.3, 4.3, [[("Worth it? ", {"bold": True, "color": BRASS}), ("A castle produces like 25 squares of land. It must pay back its cost before the game ends.", {})],
                             [("Safe? ", {"bold": True, "color": BRASS}), ("Fast-forward to when our stack has been spent on the castle. Is the general still safe then?", {})],
                             [("Needed? ", {"bold": True, "color": BRASS}), ("If we are already clearly ahead, take only cheap castles.", {})],
                             [("Right: turn 298, the bot's first castle in this game.", {"color": INK2, "size": 14})]],
     size=15.5, after=11)
term(s, 0.7, 5.95, 6.3, 0.95, "Expected utility (AIMA ch. 16)",
     "The value of a choice is what it is expected to gain minus what it costs, over the time it has to pay off.",
     "This is expected utility from chapter 16: a castle is worth taking only if what it's expected to produce before the game ends is more than what it costs.")

# --- idea 7 ----------------------------------------------------------------------
concept(7, "Exploiting a guess without betting the game on it",
        ["If you know your opponent's habits, you can play the best response to them. But if the guess is wrong, a best response can lose badly.",
         "Safe play guarantees the game's value whatever the opponent does. Safe exploitation sits between the two: exploit only as much as you can afford to lose."],
        ("Ganzfried & Sandholm, ACM TEAC 2015", "Showed when it is safe to deviate from safe play to exploit an opponent: only up to what you have already gained above the game's value."),
        [("The value of the game: the best guarantee against any opponent", "value"),
         ("Best response: the best play against a guessed opponent q", "br"),
         ("Safe exploitation: never risk falling more than k below the value", "safe")],
        ("rock-paper-scissors, opponent plays rock 50%: paper earns", "rps_ex"),
        """SAY: Idea seven is about how far to trust the guess.

Take rock-paper-scissors. If your opponent plays rock half the time, the best response is paper: on average you win a quarter of a point per round, the example at the bottom. But if they notice and switch to scissors, playing paper every time loses every round.

Safe play, the first formula, is the best you can guarantee against anyone. In rock-paper-scissors that's zero: play randomly.

The best response, second formula, is the best play against the opponent you think you face.

Ganzfried and Sandholm's answer, the third formula, is to exploit only as far as you can afford: never fall more than k below the safe value.

In our bot, the guess changes how greedy it is, but the safety test never switches off. That's our version of this rule.""")

s = new("""SAY: Idea seven: how much should the bot trust its guess about the opponent?

The guess from idea one changes how it plays. If the opponent looks aggressive, it keeps more soldiers at home, sixty percent of each threat, and it is allowed to counter-attack. If the opponent looks quiet, it keeps less at home and grabs more land.

But the guess is sometimes wrong. A paper by Ganzfried and Sandholm asks exactly this: how far can you exploit a guess about your opponent safely? Their answer is: only as far as you can afford to be wrong.

We build that in. The guess can make the bot greedier or more careful, but it can never switch off the safety test from idea three. A wrong guess costs a little efficiency. It never costs the general.""")
header(s, "Idea 7", "Use the guess, but never trust it with the general")
chips(s, 1.5, "game theory: best response vs playing safe", "Ganzfried & Sandholm 2015")
for i, (a_, b_) in enumerate([("Opponent looks aggressive", "keep 60% of each threat at home; counter-attacks allowed"),
                              ("Opponent looks quiet", "keep 40% at home; grab more land"),
                              ("We are not clearly ahead", "take castles, enemy castles first"),
                              ("We are clearly ahead", "only cheap castles")]):
    y = 2.2 + i * 0.9
    panel(s, 0.7, y, 7.6, 0.78, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.95, y + 0.22, 3.5, 0.5, a_, size=15.5, color=INK)
    arrow(s, 4.45, y + 0.39, 4.85, y + 0.39, INK3)
    text(s, 5.0, y + 0.1, 3.2, 0.7, b_, size=14, color=BRASS)
term(s, 0.7, 5.9, 12.0, 0.85, "Best response vs maximin (AIMA ch. 18)",
     "Best response: best play against the opponent you expect. Maximin: best play against the worst one. We play best response with a maximin safety floor.",
     "In game-theory terms, playing for the opponent you expect is a best response, and playing for the worst possible opponent is maximin. Best response wins more when the guess is right; maximin never loses badly. We take the best response, but keep the safety test as a maximin-style floor.")
panel(s, 8.7, 2.2, 4.0, 3.48, PANEL2)
text(s, 8.95, 2.4, 3.6, 3.2, ["The guess changes what the bot prefers.", "It never changes the safety test.",
                               "A wrong guess costs some efficiency, never the general."], size=17, after=16)

# --- idea 8 ----------------------------------------------------------------------
concept(8, "The arithmetic of an attack",
        ["An attacking stack shrinks on the way: one soldier stays on every square it crosses, and it fights anything in its path.",
         "The defending general keeps growing, and the enemy can bring other stacks home. The attack works only if what arrives beats all of that."],
        ("The game's own rules", "No paper needed here: these are the movement and production rules of the game, written as inequalities. The attack is then checked with the AND-OR test of idea 3.", "SOURCE"),
        [("What arrives: the stack, minus one per step, minus what it fights", "arrive"),
         ("The attack works only if it beats the general, its growth, and reinforcements", "attack")],
        ("(illustrative) a 60-stack, 10 steps, 5 in the way; general 2, reinforcements 20", "att_ex"),
        """SAY: Idea eight is just arithmetic, the key idea from the beginning written down.

The first formula: what arrives is the stack minus one soldier per step, minus everything it has to fight on the way.

The second: the attack only works if that is more than the general now, plus what it grows during the walk, plus what the enemy could bring home in time.

Example: a stack of 60 walks 10 squares through 5 enemy soldiers and arrives with 45. The enemy general has 2, grows by 5, and the enemy could bring 20 home: 27. 45 beats 27, so the attack can work.

The bot then also plays the attack out against four replies, the AND-OR test from idea three, before it commits.""")

s = new("""SAY: Idea eight: when does the bot attack?

From the key idea at the start, attacking a full general rarely works. So the bot only attacks when it can prove the attack works.

Three conditions. The stack must arrive bigger than the enemy general, including what the general grows and what the enemy could bring home in time. Our own general must stay safe while the stack is away. And the attack must win in imagined games against four different replies: the enemy rushes us, takes a castle, does nothing, or pulls everything home.

In practice, this happens when the enemy has emptied its general. In this screenshot, turn 405, the enemy general has only two soldiers.

This is also the bot's main weakness: against a careful player who never empties the general, it attacks late or not at all.""")
header(s, "Idea 8", "Attack only when it's sure to work")
chips(s, 1.5, "planning against every possible reply")
img(s, SH / "t405.png", 7.3, 1.45, h=5.4)
text(s, 0.7, 2.2, 6.3, 4.6, ["The bot attacks the enemy general only if:",
                             [("1  ", {"color": BRASS, "bold": True}), ("its stack arrives bigger than everything the enemy can have there;", {})],
                             [("2  ", {"color": BRASS, "bold": True}), ("our own general stays safe while the stack is away;", {})],
                             [("3  ", {"color": BRASS, "bold": True}), ("it wins against four imagined replies.", {})],
                             "In practice: when the enemy has emptied its general. Right: turn 405, the enemy general (gold box) has 2 soldiers.",
                             [("Weakness: against a careful player it attacks late.", {"color": INK2})]],
     size=16, after=12)

# --- sources -----------------------------------------------------------------------
s = new("""SAY: To sum up where the ideas came from.

From the textbook: updating beliefs with probability, planning against every possible reply, looking ahead in games, choosing under uncertainty, and a small neural network.

And six research papers. The two that shape the most behaviour are the Human.exe paper, which gave us the way to count how many soldiers can be gathered in time, used for both the safety score and the fast defence; and Puppet Search, which gave us the idea of searching over a few plans instead of every move.""")
header(s, "2 · How v51 decides", "Where each idea came from")
rows = [("Straka, Lisý & Schmid 2026", "how to count soldiers that can be gathered in time", "ideas 2, 4"),
        ("Barriga, Stanescu & Buro 2015", "search over a few plans, not every move", "idea 5"),
        ("Synnaeve & Bessière 2011", "keep a probability over the opponent's strategy", "idea 1"),
        ("Weber & Mateas 2009", "learn to recognise a strategy from game features", "idea 1"),
        ("Ganzfried & Sandholm 2015", "exploit a guess only as far as is safe", "idea 7"),
        ("Churchill & Buro 2011", "plan the economy by projecting income forward", "idea 6")]
for x_, h_ in zip((0.7, 4.3, 11.2), ("Paper", "What we borrowed", "Used in")):
    text(s, x_, 1.6, 4, 0.3, h_, size=12.5, color=BRASS, bold=True)
for i, (p_, took, where) in enumerate(rows):
    y = 2.0 + i * 0.8
    panel(s, 0.6, y, 12.2, 0.7, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.75, y + 0.2, 3.5, 0.5, p_, size=14, font=HEAD)
    text(s, 4.3, y + 0.2, 6.8, 0.5, took, size=14.5)
    text(s, 11.2, y + 0.2, 1.5, 0.5, where, size=14, color=BRASS)
text(s, 0.7, 6.9, 11.9, 0.3, "From the book: probability and belief updates, planning against replies, game lookahead, "
     "decisions under uncertainty, a small neural network.", size=12.5, color=INK2)

exec(compile((HERE / "hood_slides.py").read_text(), "hood_slides", "exec"))

# =========================================================================== #
section("4", "One real game", "Our bot (red) against a bot that plays the way people beat our earlier "
        "versions: grab land, pull everything into one stack, charge.", "t213.png",
        """SAY: Now let's watch all of this happen in one real game. The opponent plays the way people beat our earlier bots: it grabs land until turn 200, then pulls its whole army into one stack and charges. Every screenshot is the real board, with what our bot was thinking drawn on top.""")

walk_slide(40, "Opening: grab land", ["Both sides grab one square per move. Nobody is threatening anybody yet, so the bot just expands."],
           [("21", "our soldiers", RED), ("21", "their soldiers", BLUE)], "t40.png",
           """SAY: Turn 40. Both sides just grab land, one square per move. There's no threat, so the bot's plan is simply: expand.""")
walk_slide(205, "The opponent starts collecting", ["Its general is filling up: it is pulling its soldiers in.",
                                                  "Our bot still thinks it's a land grabber, which so far is true."],
           [("61", "our general", RED), ("79", "their general", BLUE)], "t205.png",
           """SAY: Turn 205. The opponent starts pulling its soldiers together; you can see its general growing. Our bot still thinks it's facing a land grabber. That's reasonable: up to now, that's exactly what it was.""",
           belief="belief205.png")
walk_slide(213, "It empties its general", ["Its general drops from 93 to 1. A 94-soldier stack appears (gold boxes).",
                                           "Our safety score drops to 12: an attack is possible soon."],
           [("94", "their big stack", BLUE), ("12", "safety score", BRASS)], "t213.png",
           """SAY: Turn 213. Here it is: the opponent empties its general, 93 soldiers down to 1, into one big stack of 94. Our safety score drops straight away, from 26 to 12.""")
walk_slide(215, "The guess changes", ["Now the board looks like a player that builds up and attacks: 97%.",
                                      "The bot makes its first defensive move."],
           [("103", "attacker", BLUE), ("66", "our general", RED)], "t215.png",
           """SAY: Turn 215. The guess flips: 97 percent "builds up, then attacks". The bot starts defending.""",
           belief="belief215.png")
walk_slide(219, "Attack confirmed: plan the defence", ["The stack is 18 squares away and coming closer.",
                                                      "The bot plans 14 moves that pull soldiers into its general (gold arrows). Every move from now on is defence."],
           [("111", "attacker", BLUE), ("18", "squares away", BRASS), ("68", "our general", RED)], "t219_plan.png",
           """SAY: Turn 219. The stack has moved closer twice in a few turns, so the bot is sure: an attack is coming, 18 squares away. It works out how short the general will be and plans 14 moves, the gold arrows, that pull soldiers into the general. From now on, every single move is defence.""")
walk_slide(224, "Pulling soldiers in", ["The farthest stacks move first, so soldiers pile up on the way in.",
                                        "The attacker shrinks as it walks."],
           [("119", "attacker", BLUE), ("13", "squares away", BRASS), ("71", "our general", RED)], "t224.png",
           """SAY: Turn 224. The gathering is under way, starting from the farthest stacks so they pick up soldiers as they come in. Notice the attacker is shrinking as it walks: the key idea from the start.""")
walk_slide(235, "Ready", ["The big stack next to the general steps in. The general now has 124.",
                          "The attacker is right next to it with 90."],
           [("124", "our general", RED), ("90", "attacker", BLUE)], "t235.png",
           """SAY: Turn 235. The last stack steps into the general. It went from 68 to 124 soldiers in 16 turns. The attacker is right next to it with 90.""")
walk_slide(237, "The attack fails", ["The attacker hits with 88. The general keeps 38.",
                                     "Our previous bot lost this exact position: 77 against 82."],
           [("38", "left in our general", RED), ("0", "attackers left", BLUE)], "t237.png",
           """SAY: Turn 237. The attack lands with 88 soldiers, and the general survives with 38. Our previous version, v46, lost this exact position on this exact turn, with 77 soldiers against 82. That's the difference the fast defence makes.""")
walk_slide(407, "Our attack, when it's sure", ["The bot went back to taking castles and pulled ahead.",
                                               "At turn 405 the enemy general had 2 soldiers. The attack passed all four tests. Captured at 407."],
           [("1,099", "our soldiers", RED), ("772", "their soldiers", BLUE)], "t407.png",
           """SAY: After that, the bot goes back to its plans, takes castles and pulls ahead. At turn 405 the enemy general has only 2 soldiers. The bot checks that an attack works against all four imagined replies, attacks, and captures the general at turn 407.""")

s = new("""SAY: The whole game on one chart. The two armies grow together until the opponent collects and charges. The attack costs both sides soldiers, but our general survives. After its first castle, our bot pulls away until it captures the general.""")
header(s, "4 · One real game", "The whole game in one picture")
img(s, ART / "army_game.png", 0.7, 1.6, w=8.0)
text(s, 9.1, 1.8, 3.6, 4.6, ["Until the opponent charges, both armies grow together.", "The attack costs both sides; our general survives.",
                             "After the first castle, our bot pulls away until the capture."], size=16, after=14)

# =========================================================================== #
section("5", "Results, then you play", "How it does against other bots, and what to try against it.", "t237.png",
        """SAY: Finally, the numbers, and then the live game.""")

s = new("""SAY: These are simulator games, twenty per bar, with both players taking turns at going first.

The red bars are v51, grey is our previous best bot. The first three groups are bots that collect and charge, the strategy people used to beat our earlier versions. That's where v51 improves the most: from 60 percent to 95 percent against the one that charges at turn 200.

Against a bot that spreads out and takes castles, it's slightly worse, because while it's busy defending, that opponent takes castles.

Head to head against our previous bot, it's about even: 32 wins to 28.

One detail if anyone asks: the defence numbers were measured with the fast defence but without the opponent classifier. The defence code is the same.""")
header(s, "5 · Results", "Where v51 is stronger")
img(s, ART / "results.png", 0.6, 1.6, w=8.4)
text(s, 9.3, 1.75, 3.4, 4.9, [[("Bots that collect and charge: ", {"bold": True, "color": BRASS}), ("the biggest improvement.", {})],
                              [("Spread-and-castle player: ", {"bold": True, "color": BRASS}), ("slightly worse.", {})],
                              [("Against our previous bot: ", {"bold": True, "color": BRASS}), ("about even, 32–28.", {})],
                              [("20 games per bar. Grey bars: our previous bot.", {"color": INK3, "size": 12.5})]],
     size=15.5, after=14)

s = new("""SAY: Now it's your turn, professor.

A few things to watch for. In the opening it just grabs land, one square per move.

If you build one big stack and walk it at its general, you'll see it stop expanding and start pulling its soldiers in from the edges. That's idea four.

If you keep your own general full, it won't attack it. It waits for you to empty it.

And if you want to beat it: spread out and take castles. That's where it's weakest.""")
header(s, "5 · Live game", "Your turn")
img(s, SH / "t120.png", 7.3, 1.45, h=5.4)
panel(s, 0.7, 1.65, 6.3, 0.75, PANEL2)
text(s, 0.95, 1.88, 6.0, 0.5, "python play_arena.py   then pick  Final · champion", size=13, font="Menlo", color=BRASS)
text(s, 0.7, 2.7, 6.3, 4.2, ["What to watch for:",
                             [("Opening: ", {"bold": True, "color": BRASS}), ("it grabs land, one square per move.", {})],
                             [("Send a big stack at its general: ", {"bold": True, "color": BRASS}), ("it stops and pulls soldiers in.", {})],
                             [("Keep your general full: ", {"bold": True, "color": BRASS}), ("it won't attack.", {})],
                             [("To beat it: ", {"bold": True, "color": BRASS}), ("spread out and take castles.", {})]],
     size=17, after=14)

s = new("References, for anyone who wants to read further.")
header(s, "References", "Book and papers")
text(s, 0.7, 1.7, 11.9, 5.0, ["Russell, S. & Norvig, P. Artificial Intelligence: A Modern Approach, 4th ed. Pearson, 2020.",
                              "Straka, M., Lisý, V. & Schmid, M. Superhuman AI for Generals.io using self-play reinforcement learning. arXiv 2606.23348, 2026.",
                              "Barriga, N., Stanescu, M. & Buro, M. Puppet Search: enhancing scripted behavior by look-ahead search with applications to RTS games. AIIDE 2015.",
                              "Synnaeve, G. & Bessière, P. A Bayesian model for opening prediction in RTS games with application to StarCraft. AIIDE 2011.",
                              "Weber, B. & Mateas, M. A data mining approach to strategy prediction. CIG 2009.",
                              "Ganzfried, S. & Sandholm, T. Safe opponent exploitation. ACM Transactions on Economics and Computation, 2015.",
                              "Churchill, D. & Buro, M. Build order optimization in StarCraft. AIIDE 2011.",
                              "Started from (section 1): Likhachev, Gordon & Thrun, ARA*, NIPS 2003; Koenig & Likhachev, RTAA*, AAMAS 2006; Hernández & Baier, depression avoidance, AAAI 2012; Buro, ProbCut, ICCA J. 1995; Russell & Wefald, metareasoning, AIJ 1991."],
     size=14, color=INK2, after=10)

prs.core_properties.title = "How our bot plays Generals: v51"
prs.save(str(OUT))
print("saved", OUT, N[0])
