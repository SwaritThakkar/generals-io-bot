# Section 3: every part of v51. Executed inside build3.py (helpers already defined).

section("3", "Every part of v51", "Everything the eight ideas skipped: the order of checks, the simulator, the learned "
        "win estimate, the classifier in full, and the small safety rules.", "t224.png",
        """SAY: The eight ideas are the core. This section goes through every remaining part of the bot, so that nothing in it is a black box: the exact order of checks each turn, the simulator, the learned model that scores imagined games, the classifier in full, and the small rules that came from specific lost games.""")

# --- the full turn ----------------------------------------------------------------
s = new("""SAY: This is the exact order of checks the bot runs every turn. The first check that says yes decides the move.

One: read the board and update the guess about the opponent.

Two: if one of our stacks next to the enemy general can take it, take it. That ends the game, so nothing else matters.

Three: is a big enemy stack coming, and will the general be short when it arrives? Then the fast defence from idea 4 picks the move.

Four: update the two switches from idea 7, the castle gate and the threat gate. The threat gate sets how much the general keeps in reserve.

Five: if our home is losing the race, look for a counter-strike: hitting their general first. Only if it passes the four-reply test.

Six: if things are urgent, the safety score is under the pad or we have a proven attack, the rule list from v36 decides: attack, defend, or pull soldiers home.

Seven: if a castle is worth taking and safe, take it.

Eight: otherwise follow the committed plan from idea 5.

And at the bottom: plan and castle moves are tested first. If a move fails the test, the next rule is tried, and if nothing passes, a simple defender script moves.""")
header(s, "3 · Every part of v51", "The full turn, in order")
rows = [("1", "Read the board; update the guess about the opponent", "always", INK3),
        ("2", "Can a stack take their general right now?", "take it", BRASS),
        ("3", "A big stack is coming and the general will be short?", "fast defence (idea 4)", BRASS),
        ("4", "Update the castle gate and the threat gate (idea 7)", "set the reserve R", INK3),
        ("5", "Is our home losing the race?", "counter-strike, if it passes the test", BRASS),
        ("6", "Urgent: safety score under the pad, or a proven attack?", "v36 rules: attack / defend / pull home", BRASS),
        ("7", "A castle pays back and stays safe?", "castle move (idea 6)", BRASS),
        ("8", "Otherwise", "the committed plan (idea 5)", BRASS)]
for i, (n_, cond, act, col) in enumerate(rows):
    y = 1.55 + i * 0.6
    panel(s, 0.7, y, 7.4, 0.52, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.85, y + 0.1, 0.4, 0.4, n_, size=16, font=HEAD, color=BRASS)
    text(s, 1.35, y + 0.13, 6.7, 0.4, cond, size=13.5)
    arrow(s, 8.15, y + 0.26, 8.55, y + 0.26, INK3)
    rect(s, 8.6, y, 4.1, 0.52, PANEL2, col, MSO_SHAPE.ROUNDED_RECTANGLE, 0.2)
    text(s, 8.75, y + 0.13, 3.9, 0.4, act, size=13, color=col if col != INK3 else INK2)
panel(s, 0.7, 6.5, 12.0, 0.6, PANEL2)
text(s, 0.95, 6.63, 11.6, 0.4, [[("Every plan and castle move is tested first (idea 3). ", {"bold": True, "color": BRASS}),
                                 ("If it fails, the next rule is tried; if nothing passes, a simple defender script moves.", {"color": INK2})]],
     size=13.5)

# --- simulator -------------------------------------------------------------------------
s = new("""SAY: Every time I've said "the bot imagines" or "plays it out in its head", it uses this: a simulator.

It's an exact copy of the game's rules, written in plain Python. The real game engine is written in JAX, which is fast for big batches but slow when you call it one turn at a time. Our copy is checked against the real engine on thousands of random moves, and the boards come out identical.

The rules it copies are short. Each turn each player moves one stack one square, all of it but one soldier, or half. When stacks meet, the smaller is subtracted from the larger. Generals and castles grow by one every two turns, every square of land by one every 50 turns. Take the enemy general and you win.

One turn changes at most four squares, so the simulator only updates those, and that's what makes it cheap enough to simulate over a thousand turns whenever the bot picks a new plan.

In the book's language this is the transition model, the RESULT function: given a state and both players' actions, what's the next state.""")
header(s, "3 · Every part of v51", "The simulator: where all the imagining happens")
img(s, SH / "t150_us.png", 7.3, 1.45, h=5.4)
text(s, 0.7, 1.6, 6.3, 2.6, [[("An exact copy of the game's rules ", {"bold": True, "color": BRASS}), ("in plain Python, checked against the real game engine on thousands of random moves: identical boards.", {})],
                             [("Rules: ", {"bold": True, "color": BRASS}), ("move one stack one square (all but one soldier, or half). Meeting stacks subtract. Take the general, win.", {})],
                             [("Cheap: ", {"bold": True, "color": BRASS}), ("one turn changes at most four squares, so only those are updated. Over a thousand simulated turns each time it picks a plan.", {})]],
     size=14.5, after=10)
panel(s, 0.7, 4.4, 6.3, 0.75, PANEL2)
eq(s, "growth_rule", 0.9, 4.58, h=0.32, maxw=5.9)
term(s, 0.7, 5.45, 6.3, 1.25, "Transition model (AIMA ch. 3, 5)",
     "RESULT(s, a): the state that follows state s after action a. Here a is a pair of moves, one per player, made at the same time.",
     "In the book's terms this is the transition model, the RESULT function: give it a board and both players' moves, it gives back the next board.")
text(s, 7.3, 6.88, 5.4, 0.3, "Right: our stacks' routes home (light red) and enemy routes (light blue), turn 150.",
     size=11, color=INK3, italic=True)

# --- phases ------------------------------------------------------------------------
s = new("""SAY: Idea 5 said the bot imagines forty turns ahead. That number actually depends on the phase of the game, and the bot works out the phase from the board, not from the clock.

Opening: nobody threatens anybody and more than 35 percent of the walkable land is still free. That's the second formula. Mid-game: there is contact or a live threat. End game: fewer than 60 turns left, or one side clearly ahead, twice the land plus 20 and twice the army plus 40. That's the first formula.

The table: in the opening it looks 24 turns ahead and keeps a plan for 6 turns; mid-game 40 and 10; end game 60 and 8, or right to the last turn if that's closer, where the game's tiebreak decides the winner exactly.

Keeping a plan for several turns matters: otherwise the bot changes its mind every turn and gets nowhere, which is exactly what we saw v11 do. But some events force a new plan straight away: the phase changes, a castle changes hands, the enemy general drops below half, or the plan is finished. That's the last formula.

In the book this is depth-limited search: look ahead a fixed number of steps, then score the position with an evaluation function, which is the next slides.""")
header(s, "3 · Every part of v51", "Phases: how far ahead to look, how long to keep a plan")
heads = [("", 0.95), ("opening", 3.3), ("mid-game", 5.3), ("end game", 7.3)]
panel(s, 0.7, 1.55, 8.4, 2.65)
for h_, x_ in heads:
    text(s, x_, 1.7, 1.9, 0.3, h_, size=13, color=BRASS, bold=True)
for i, (lab, vals) in enumerate((("look ahead", ("24 turns", "40 turns", "60 turns")),
                                 ("keep the plan", ("6 turns", "10 turns", "8 turns")),
                                 ("castles offered", ("2 nearest", "3 nearest", "3 nearest")),
                                 ("attack plans", ("no", "yes", "yes")))):
    y = 2.15 + i * 0.5
    text(s, 0.95, y, 2.3, 0.4, lab, size=14, color=INK2)
    for v_, (_h, x_) in zip(vals, heads[1:]):
        text(s, x_, y, 1.9, 0.4, v_, size=14.5, font="Menlo")
y = 4.4
for words, e in (("End game: few turns left, or one side clearly ahead", "ph_end"),
                 ("Opening: no threat yet and plenty of free land (f free squares, n, n′ owned)", "ph_open"),
                 ("A new plan before K turns are up if:", "replan")):
    text(s, 0.7, y, 8.4, 0.35, words, size=13, color=INK2)
    pic = eq(s, e, 0.75, y + 0.35, h=0.32, maxw=8.3)
    y += 0.35 + pic.height / 914400 + 0.17
term(s, 9.4, 1.55, 3.3, 3.0, "Depth-limited search (AIMA ch. 5)",
     "Look ahead a fixed number of moves, then score the position with an evaluation function instead of playing to the end.",
     "This is depth-limited search from chapter 5: look a fixed number of moves ahead and score the position there with an evaluation function, since you can't play every line to the end.")
text(s, 9.4, 4.75, 3.3, 2.0, "Keeping a plan for several turns stops the bot from changing its mind every turn, which is how v11 lost.",
     size=13.5, color=BRASS)

# --- plans ----------------------------------------------------------------------------
s = new("""SAY: Which plans does it choose between? Here's the full menu.

Expand: grab neutral land with the stacks that aren't on the general or a castle. Hold: keep soldiers banked on the general and castles, still expanding with the rest. Take castle: one plan for each of the two or three nearest castles, ranked by distance plus half the garrison, the formula. Gather home: pull everything into the general, offered only when the safety score is low. Strike: collect on our castle nearest the enemy general, then charge when the walk wins. Rush: attack with any stack whose walk already wins.

And the four imagined opponents: one that rushes our general, one that takes castles then rushes, one that only grabs land, and one that does nothing.

At most 8 plans times 4 opponents is 32 imagined games of 40 turns each: about 1,300 simulated turns, and it only does this when it picks a new plan.

One trick saves a lot: if two plans lead to exactly the same boards against all four opponents, which happens a lot in the opening where most moves just take one free square, they're merged and scored once. On average 6.1 plans become 4.6.

In the book these are macro actions, high-level actions from the chapter on hierarchical planning: each plan is a small script that expands into single moves.""")
header(s, "3 · Every part of v51", "The plan menu and the four imagined opponents")
plans = [("expand", "grab free land with stacks that aren't on the general or a castle"),
         ("hold", "keep soldiers banked on the general and castles; expand with the rest"),
         ("take castle ×2–3", "one plan per nearby castle, nearest first (formula)"),
         ("gather home", "pull everything into the general; only when the safety score is low"),
         ("strike", "collect on our castle nearest their general, charge when the walk wins"),
         ("rush", "attack now with any stack whose walk already wins")]
text(s, 0.7, 1.55, 7.3, 0.3, "OUR PLANS", size=12, color=BRASS, bold=True)
for i, (n_, d_) in enumerate(plans):
    y = 1.9 + i * 0.6
    panel(s, 0.7, y, 7.3, 0.52, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.9, y + 0.13, 2.1, 0.4, n_, size=14, font=HEAD, color=BRASS)
    text(s, 3.05, y + 0.14, 4.9, 0.4, d_, size=12.5)
text(s, 0.7, 5.6, 7.3, 0.35, "Castles are offered in this order:", size=13, color=INK2)
eq(s, "castle_rank", 0.75, 5.95, h=0.34, maxw=6.5)
text(s, 8.4, 1.55, 4.3, 0.3, "IMAGINED OPPONENTS", size=12, color=BLUE, bold=True)
for i, (n_, d_) in enumerate((("rusher", "walks its biggest stack at our general"), ("castle rusher", "takes castles, then rushes"),
                              ("land grabber", "only expands"), ("waiter", "does nothing"))):
    y = 1.9 + i * 0.6
    panel(s, 8.4, y, 4.3, 0.52, PANEL if i % 2 == 0 else PANEL2)
    text(s, 8.6, y + 0.13, 1.6, 0.4, n_, size=13.5, font=HEAD, color=BLUE)
    text(s, 10.2, y + 0.15, 2.45, 0.4, d_, size=11.5, color=INK2)
text(s, 8.4, 4.4, 4.3, 0.5, "8 plans × 4 opponents = 32 imagined games of 40 turns.", size=13)
text(s, 8.4, 4.95, 4.3, 0.9, "Plans that give identical boards against all four are merged: on average 6.1 plans become 4.6.",
     size=13, color=INK2)
term(s, 8.4, 5.9, 4.3, 1.1, "Macro actions (AIMA ch. 11)",
     "High-level actions that expand into many single moves, as in hierarchical planning.",
     "Each plan is what the book calls a high-level or macro action, from hierarchical planning: one choice that expands into many single moves.")

# --- value model: concept ------------------------------------------------------------
concept("5", "Turning a board into a chance of winning",
        ["At the end of each imagined game the bot needs one number: how likely am I to win from here?",
         "Logistic regression: add up a few measurements, each times a weight, then squash the total into 0 to 1 with the sigmoid. Each weight says how much one unit of a measurement changes the odds."],
        ("Russell & Norvig, AIMA ch. 19 and ch. 5", "Chapter 19: logistic regression, a linear model with a sigmoid output, fitted by minimising log-loss. Chapter 5: an evaluation function scores positions where the search stops.", "FROM THE BOOK"),
        [("The win estimate: a weighted sum, squashed by the sigmoid σ", "logit"),
         ("Same thing read as odds: each weight adds to the log-odds", "logodds"),
         ("Trained by minimising log-loss (with a small penalty on big weights)", "loss")],
        ("(illustrative) opening: land lead +20, army lead +50, income lead worth +0.5", "val_ex"),
        """SAY: This is the one learned part I haven't explained yet, and it's what gives every imagined game its score.

At the end of each 40-turn imagined game, the bot needs one number: from this board, how likely am I to win? It's not a neural network. It's logistic regression, from chapter 19.

The first formula: take nine measurements of the board, phi, multiply each by a weight, add them up into a score z, and squash z into a probability with the sigmoid, sigma. Big positive score, close to 100 percent; zero, 50 percent.

The second formula is the same model read differently: the log of the odds is just the weighted sum. So each weight tells you how much one unit of that measurement shifts the odds.

The third is how the weights are learned: log-loss. For each training position we know who actually won, y. The loss punishes the model for being confident and wrong. The last term, lambda, discourages huge weights.

Example at the bottom: an opening position where we're 20 squares ahead, 50 soldiers ahead, and ahead on income. The score is 1.29, the sigmoid turns that into 78 percent.

In chapter 5 terms, this is the evaluation function: the score you use where the lookahead stops.""",
        label="3 · The learned win estimate · concept")

# --- value model: features ---------------------------------------------------------------
s = new("""SAY: These are the nine measurements, all differences between us and the opponent, so the model sees the board from our side.

Land lead divided by 50. Army lead divided by 100. Producers, meaning generals plus castles. Income times turns left: income is half a soldier per producer per turn plus one per 50 squares of land, so this is roughly how much extra army our income will have produced by the end. Then three stack comparisons: our general against their biggest stack, which is defence; our biggest stack against their general, attack; and biggest stack against biggest stack. Then turns left, and a constant.

There's a separate set of weights for each phase. The chart shows them. In the opening, income times turns left matters a lot: early income compounds. In the mid-game, land dominates with a weight of 2. In the end game, army counts again, because the game is decided soon.

Two weights are exactly zero: the constant and turns left. That's not a bug. Every training position appears from both players' sides, so the model can't favour either side, and those two measurements are the same for both. For the same reason defence and attack get identical weights: seen from the other side, my defence is your attack.""")
header(s, "3 · The learned win estimate", "The nine measurements and what the model learned")
feats = [("land", "land lead", "÷ 50"), ("army", "army lead", "÷ 100"), ("prod", "producers (generals + castles) lead", "×1"),
         ("income", "income lead × turns left", "÷ 100"), ("defence", "our general − their biggest stack", "÷ 50"),
         ("attack", "our biggest stack − their general", "÷ 50"), ("big", "biggest stack lead", "÷ 50"),
         ("left", "turns left", "÷ 500"), ("bias", "constant", "1")]
for i, (k_, d_, sc) in enumerate(feats):
    y = 1.55 + i * 0.5
    panel(s, 0.7, y, 5.6, 0.44, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.85, y + 0.1, 1.2, 0.3, k_, size=12.5, font="Menlo", color=BRASS)
    text(s, 2.05, y + 0.1, 3.4, 0.3, d_, size=12.5)
    text(s, 5.45, y + 0.1, 0.8, 0.3, sc, size=12, color=INK2)
text(s, 0.7, 6.15, 5.6, 0.9, "Income = ½ per general or castle + 1 per 50 squares of land, per turn.", size=12.5, color=INK2)
img(s, ART / "value_weights.png", 6.6, 1.55, w=6.1)
text(s, 6.6, 4.75, 6.1, 2.2, [[("Opening: ", {"bold": True, "color": BRASS}), ("income × turns left counts most; early income compounds.", {})],
                              [("Mid-game: ", {"bold": True, "color": BRASS}), ("land dominates (weight 2.05).", {})],
                              [("Constant and turns left: exactly 0. ", {"bold": True, "color": BRASS}), ("Every position is seen from both sides, so neither side can be favoured.", {"color": INK2})]],
     size=13.5, after=8)

# --- value model: training and results -----------------------------------------------------
s = new("""SAY: How was it trained, and how good is it?

The data is self-play: seven simple scripted bots, economy bots with different settings, rushers, a land grabber, a striker, play each other on 200 maps, with games lasting 300, 500 or 800 turns so that "turns left" varies. Every 5 turns we record both players' measurements and, later, who won. Draws are dropped. 160 maps are for training, about 27,000 positions; 40 maps are held out for testing, about 7,000.

The fitting uses Newton's method, the first formula. Instead of taking small downhill steps, it uses the curvature of the loss, the matrix H, the second formula, to jump close to the minimum. For logistic regression the loss is convex, so this converges in a handful of steps to the single best set of weights. This was done in plain numpy; it's nine weights, there was no need for a GPU.

The chart compares it with a simpler model that only uses the land difference. Lower is better. In the opening, the full model is clearly better: 0.473 against 0.522. In the end game, better too. In the mid-game, honestly, no better than land alone: 0.372 against 0.359 on the held-out maps. That fits the weights we just saw, where the mid-game model is mostly land anyway.

So the win estimate is a simple, honest model: right about 75 percent of the time in the opening and 98 percent in the end game.""")
header(s, "3 · The learned win estimate", "Training it, and how well it predicts")
text(s, 0.7, 1.55, 6.0, 2.5, [[("Data: ", {"bold": True, "color": BRASS}), ("7 scripted bots play each other on 200 maps; games of 300, 500 or 800 turns.", {})],
                              [("Every 5 turns: ", {"bold": True, "color": BRASS}), ("both players' measurements, labelled with who won. Draws dropped.", {})],
                              [("Split by map: ", {"bold": True, "color": BRASS}), ("160 maps train (27,208 positions), 40 maps held out (6,986).", {})]],
     size=14, after=9)
text(s, 0.7, 4.0, 6.0, 0.35, "Fitted by Newton's method, one model per phase:", size=13, color=INK2)
eq(s, "newton", 0.75, 4.38, h=0.36, maxw=3)
eq(s, "grad", 0.75, 4.95, h=0.3, maxw=5.95)
term(s, 0.7, 5.6, 6.0, 1.3, "Newton's method (AIMA ch. 4.2)",
     "Uses the curvature H of the loss to jump towards the minimum. Log-loss is convex, so it reaches the one best answer in a few steps. Plain numpy: 9 weights.",
     "Newton's method is in chapter 4, local search in continuous spaces: use the gradient and the curvature to jump towards the minimum. Because log-loss for logistic regression is convex there is exactly one minimum, and Newton finds it in a few steps.")
img(s, ART / "value_results.png", 7.0, 1.55, w=5.7)
text(s, 7.0, 4.6, 5.7, 2.3, [[("Opening: ", {"bold": True, "color": BRASS}), ("clearly better than land alone; right 75% of the time (land alone: 71%).", {})],
                             [("Mid-game: ", {"bold": True, "color": BRASS}), ("no better than land alone on new maps.", {})],
                             [("End game: ", {"bold": True, "color": BRASS}), ("right 98% of the time.", {})]],
     size=13.5, after=8)

# --- opponent weights ------------------------------------------------------------------------
s = new("""SAY: Idea 5 averaged the four imagined opponents using our belief about which one we face. Where do those weights come from? Two sources, multiplied together.

The first source is the opponent's last move. Every turn, each of the four imagined opponents is asked: what would you have done on last turn's board? If its answer matches what the real opponent did, its weight goes up. The first formula: the factor is 0.8 for a match and 0.2 for a miss, so a match counts four times as much, and the 0.95 makes old evidence fade.

The second source is the classifier from idea 1. It knows ten styles, which are mapped onto the four imagined opponents: the table. A land grabber, a spread-out collector, an economy bot and our older planner bots all play like the land grabber script. Early and mid-game gatherers play like the rusher. A turtle plays like the waiter.

The second formula combines them: the square root of the move-based weight times the classifier's probability. The square root gives the move evidence less say, because one move is weak evidence. The 0.02 keeps every opponent possible, so the worst-case half of the score never forgets any of them.""")
header(s, "3 · Every part of v51", "Where the opponent weights in idea 5 come from")
text(s, 0.7, 1.55, 6.2, 0.4, "1 · Did each imagined opponent predict the last move?", size=15, font=HEAD)
eq(s, "movelik", 0.75, 2.05, h=0.32, maxw=6.1)
text(s, 0.7, 2.75, 6.2, 0.4, "2 · Combined with the classifier's ten styles:", size=15, font=HEAD)
panel(s, 0.7, 3.25, 6.2, 1.0, PANEL2)
eq(s, "combine", 0.95, 3.38, h=0.36, maxw=5.7)
text(s, 0.7, 4.45, 6.2, 2.4, ["A matching prediction counts four times as much as a miss; old evidence fades (0.95).",
                              "The square root gives one move less say than the classifier.",
                              "The 0.02 keeps every opponent possible, so the worst case never forgets one."],
     size=14, color=INK2, after=8)
text(s, 7.3, 1.55, 5.4, 0.3, "CLASSIFIER STYLE  →  IMAGINED OPPONENT", size=12, color=BRASS, bold=True)
for i, (a_, b_) in enumerate((("rusher; gathers early or mid-game", "rusher"), ("castle rusher", "castle rusher"),
                              ("land grabber; gathers late; spreads out; economy bot; planner bot", "land grabber"),
                              ("turtle", "waiter"))):
    y = 1.95 + i * 0.95
    panel(s, 7.3, y, 5.4, 0.85, PANEL if i % 2 == 0 else PANEL2)
    text(s, 7.5, y + 0.12, 3.3, 0.7, a_, size=13)
    arrow(s, 10.85, y + 0.42, 11.2, y + 0.42, INK3)
    text(s, 11.3, y + 0.27, 1.4, 0.4, b_, size=13, color=BLUE, font=HEAD)

# --- classifier: inputs and network --------------------------------------------------------------
s = new("""SAY: Now the classifier from idea 1 in full. It's the only neural network in the bot, and it's small.

The input is 14 numbers measured from the board every turn, all about the opponent: the turn number; its land and total army; what share of its army sits in its general, and in its biggest stack; how many castles it holds; how far its biggest stack is from our general, and how much closer it got in the last 5 turns; how much land and army it gained over the last 20 turns; how many stacks of 5 or more it has; how many times it has emptied its general; and whether its biggest stack is within 10 squares of us. Each is divided by a typical size so they're all roughly between 0 and 1.

The network: 14 inputs, two hidden layers of 64 with ReLU, which just sets negatives to zero, and 10 outputs turned into probabilities by the softmax. That's the first formula. Before going in, each input is standardised: minus its average, divided by its spread, from the training data.

It has 5,770 weights in total, the last formula. It runs in plain Python inside the bot, about a millisecond per turn.""")
header(s, "3 · The style classifier", "Fourteen measurements in, ten styles out")
feats = ["turn number", "land", "total army", "share of army in its general", "share in its biggest stack",
         "castles held", "biggest stack's distance to us", "how much closer in 5 turns", "land gained in 20 turns",
         "biggest stack's growth in 5 turns", "stacks of 5 or more", "times its general was emptied",
         "army gained in 20 turns", "biggest stack within 10 squares?"]
for i, f_ in enumerate(feats):
    col, row = divmod(i, 7)
    x = 0.7 + col * 3.05
    y = 1.6 + row * 0.56
    panel(s, x, y, 2.95, 0.48, PANEL if row % 2 == 0 else PANEL2)
    text(s, x + 0.12, y + 0.12, 0.4, 0.3, str(i + 1), size=11.5, color=BRASS, bold=True)
    text(s, x + 0.5, y + 0.12, 2.4, 0.3, f_, size=11.5)
text(s, 0.7, 5.6, 6.0, 0.6, "All measured on the opponent, from the board alone, and scaled to roughly 0 to 1.",
     size=13, color=INK2)
img(s, ART / "mlp.png", 7.1, 1.5, w=5.6)
eq(s, "mlp", 7.1, 4.45, h=0.3, maxw=5.6)
eq(s, "params", 7.1, 5.05, h=0.3, maxw=5.0)
term(s, 7.1, 5.6, 5.6, 1.3, "Feed-forward neural network (AIMA ch. 21)",
     "Layers of weighted sums with a simple non-linearity (ReLU) in between; the softmax turns the last layer into probabilities.",
     "This is a feed-forward network from chapter 21: each layer is a weighted sum followed by ReLU, which sets negatives to zero; the last layer goes through the softmax so the ten outputs are probabilities that add up to one.")

# --- classifier: training and accuracy -------------------------------------------------------------
s = new("""SAY: How it was trained, and how good it is, including where it gets confused.

Data: our previous bot, v46, played each of the ten styles on 33 maps, from both seats, 500 turns each, and we recorded the opponent's 14 measurements every second turn with its true style. 25 maps for training, about 99,000 rows; 8 different maps for testing, about 33,000 rows. So the test is always on maps it has never seen.

Training: the loss is cross-entropy, the formula: minus the log of the probability it gave the right style. We used the Adam optimiser, learning rate 0.002, small weight decay, batches of 512, 60 passes over the data, on the Mac's GPU.

The left chart: from one board on its own, it's right 25 percent of the time in the first 100 turns, because early on everybody just grabs land, rising to about 70 percent. The gold line is the belief built up over the game, the filter from idea 1: 39 percent by turn 100, 66 by turn 200, 70 by turn 300. Random guessing would be 10.

The right chart shows where it's wrong. Rows are the true style, columns the guess. The big mistake: a rusher is often taken for a turtle, 55 percent. That makes sense: a rusher sits at home building up until it launches, and until then it looks exactly like a turtle. The economy bot and the planner bot are also mixed up, but those are our own older bots and they play alike, and both map to the same imagined opponent, so that confusion costs nothing. And a rusher still gets caught: once it actually attacks, the threat gate opens anyway.""")
header(s, "3 · The style classifier", "Training it, and where it gets confused")
text(s, 0.7, 1.55, 5.9, 2.4, [[("Data: ", {"bold": True, "color": BRASS}), ("v46 plays each of the 10 styles on 33 maps, both seats, 500 turns. Measurements every 2nd turn.", {})],
                              [("Split by map: ", {"bold": True, "color": BRASS}), ("25 maps train (99,437 rows), 8 unseen maps test (33,318).", {})],
                              [("Training: ", {"bold": True, "color": BRASS}), ("Adam, learning rate 0.002, weight decay 0.0001, batches of 512, 60 passes, on the Mac GPU.", {})]],
     size=13.5, after=8)
eq(s, "xent", 0.75, 3.95, h=0.34, maxw=4.5)
img(s, ART / "cls_acc.png", 0.7, 4.45, w=5.9, h=2.6)
img(s, ART / "cls_confusion.png", 7.0, 1.45, h=4.65)
text(s, 7.0, 6.2, 5.7, 0.9, [[("Rusher read as turtle (55%): ", {"bold": True, "color": BRASS}),
                              ("both sit at home until the rusher launches. Economy bot vs planner bot: both map to the same imagined opponent.", {"color": INK2})]],
     size=12)

# --- reserve and mass threat ------------------------------------------------------------------------
s = new("""SAY: Now the small safety rules, each of which came from a specific lost game.

First, the reserve. In one human win, the human walked their general 25 squares, collecting soldiers on the way, from 84 to 137. Our safety score read plus 10 to plus 40 the whole time, until two turns before it was too late, because each turn the human's army was still spread out.

So when the general sends soldiers out, for a castle or anything else, it must keep a reserve: what it keeps, plus its growth, must be at least R times what every enemy walk could bring. Not only now, but also if the enemy waits 30 or 60 turns first and keeps growing. That's the first formula. R is 0.6 when the threat gate is open and 0.4 when it's shut: that's the gate from idea 7.

Second, the mass threat: the second formula. If the enemy spent H moves collecting, each move picks up at most one square, so the most it can collect is its H biggest squares, plus its general's growth, minus what it costs to walk the result to our general. The reserve has to cover that too.""")
header(s, "3 · Safety details", "The reserve: what the general must keep")
text(s, 0.7, 1.55, 6.3, 1.2, "From a lost game: a human walked their general 25 squares, collecting 84 → 137. Our safety "
     "score read +10 to +40 until two turns too late.", size=14.5, color=INK2)
text(s, 0.7, 2.6, 6.3, 0.4, "1 · Whatever leaves the general, it keeps a reserve, now and if the enemy waits:", size=14)
panel(s, 0.7, 3.05, 6.3, 0.75, PANEL2)
eq(s, "reserve", 0.9, 3.22, h=0.32, maxw=5.95)
text(s, 0.7, 4.0, 6.3, 0.4, "2 · The most the enemy could collect in H moves and walk at us:", size=14)
panel(s, 0.7, 4.45, 6.3, 0.75, PANEL2)
eq(s, "mass", 0.9, 4.62, h=0.32, maxw=5.95)
text(s, 0.7, 5.45, 6.3, 1.4, [[("R = 0.6 ", {"bold": True, "color": BRASS}), ("when the threat gate is open, ", {}),
                              ("0.4 ", {"bold": True, "color": BRASS}), ("when it is shut (idea 7). Each collecting move picks up at most one square, so H moves collect at most the H biggest.", {})]],
     size=13.5, color=INK2)
img(s, SH / "t213.png", 7.4, 1.45, h=5.3)
text(s, 7.4, 6.8, 5.3, 0.3, "Turn 213 of our game: the enemy empties its general into one stack (gold boxes).", size=11,
     color=INK3, italic=True)

# --- pad, lead, tiles once ------------------------------------------------------------------------------
s = new("""SAY: Three more details of the safety score from idea 2, each one line of maths.

The pad. "Small means danger": how small? The pad is the answer. It's 3 soldiers, plus 8 percent of the enemy's biggest moving stack, so that late in the game, with stacks of hundreds, a margin of 3 isn't treated as safe. Plus a trend term: if the safety score has been falling over the last 8 or more turns, the pad grows by 16 times the rate of the fall. A falling score means the enemy is building up faster than we are, so we react earlier.

The lead. Our stacks walking home must arrive with time to spare, so in the safety score our side only gets T minus 2 minus T over 10 moves while the enemy gets T. The longer the walk, the more slack.

Counting each square once. Two walks home can share the same corridor. Before this fix both walks counted the soldiers standing on that corridor, so we counted some soldiers twice and felt safer than we were. Now a later walk loses whatever an earlier chosen walk already picked up: the third formula. The same applies to the enemy's side.""")
header(s, "3 · Safety details", "The pad, the lead, and counting each square once")
for i, (t_, e_, d_) in enumerate((("The pad: how small is too small", "pad",
                                   "3 soldiers, plus 8% of their biggest moving stack, plus extra when the safety score has been falling (Δm/Δt over 8+ turns)."),
                                  ("The lead: our stacks must land early", "lead",
                                   "In the safety score, our walks home get fewer moves than the enemy's: slack grows with the length of the walk."),
                                  ("Each square counted once", "once",
                                   "Walk k loses what earlier chosen walks (cells C) already picked up on its path P_k. Before this fix, a shared corridor was counted twice."))):
    y = 1.55 + i * 1.8
    panel(s, 0.7, y, 11.9, 1.62, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.95, y + 0.15, 5.0, 0.4, t_, size=17, font=HEAD, color=BRASS)
    eq(s, e_, 0.95, y + 0.72, h=0.38, maxw=5.6)
    text(s, 6.9, y + 0.2, 5.5, 1.3, d_, size=14, color=INK2)

# --- swap-safe and counter-strike -----------------------------------------------------------------------------
s = new("""SAY: Two rules found by replaying games that humans won.

Left: never swap places with an attacker. Our general has 30 and an enemy stack of 25 is right next to it. The obvious move is to hit the attacker with the general: 30 beats 25. But moves are simultaneous. If the attacker steps onto our general in the same turn that our general's soldiers step out towards the attacker, the two stacks pass each other. The attacker lands on a general with only one soldier left and takes it. We saw exactly this in replays. So the general never steps toward an adjacent enemy stack; another stack hits it instead, or the general just waits and grows.

Right: the counter-strike. Sometimes home is going to fall: the safety score says the general loses the race at some turn T-fall. Then defending is hopeless, but attacking might not be. If one of our stacks can reach their general before T-fall, the first condition, and arrive bigger than their general plus its growth plus whatever they can pull home in time plus 3, the second condition, the bot launches it, but only if it also passes the four-reply test from idea 3. It's only allowed while the threat gate is open.""")
header(s, "3 · Safety details", "Two rules from replays of human wins")
text(s, 0.7, 1.55, 5.9, 0.4, "Never swap places with an attacker", size=19, font=HEAD, color=BRASS)
for k, (x, lab, col, num) in enumerate(((1.4, "our general", RED, "30"), (3.6, "attacker", BLUE, "25"))):
    rect(s, x, 2.6, 1.5, 1.5, col, None, MSO_SHAPE.ROUNDED_RECTANGLE, 0.08)
    text(s, x, 2.95, 1.5, 0.7, num, size=34, font=HEAD, align=PP_ALIGN.CENTER)
    text(s, x, 4.2, 1.5, 0.4, lab, size=12.5, color=INK2, align=PP_ALIGN.CENTER)
arrow(s, 2.9, 2.35, 4.2, 2.35, RED)
arrow(s, 4.2, 2.2, 2.2, 2.2, BLUE)
text(s, 0.7, 4.75, 5.9, 1.9, ["Same turn: the general steps out, the attacker steps in. They pass each other, and the attacker lands on a general with 1.",
                              "So the general never steps toward an adjacent enemy stack. Another stack hits it, or the general waits and grows."],
     size=13.5, color=INK2, after=8)
text(s, 7.0, 1.55, 5.7, 0.4, "Counter-strike: when home will fall anyway", size=19, font=HEAD, color=BRASS)
text(s, 7.0, 2.15, 5.7, 1.3, "If the safety score says our general falls at turn T_fall, launch a stack at their general if it arrives first and wins:",
     size=14, color=INK2)
panel(s, 7.0, 3.1, 5.7, 1.4, PANEL2)
eq(s, "counter1", 7.2, 3.22, h=0.36, maxw=5.3)
eq(s, "counter2", 7.2, 3.85, h=0.36, maxw=5.3)
text(s, 7.0, 4.6, 5.7, 1.5, ["G′, D′_home: their general, and what they can pull home in time. +3 for safety.",
                             "Only with the threat gate open, and only if it passes the four-reply test (idea 3)."],
     size=13.5, color=INK2, after=8)

# --- castles in full -------------------------------------------------------------------------------------------
s = new("""SAY: Idea 6 in full: the exact castle rules.

Which castles: any castle we don't own within 30 squares of our general, with a walk of at most 25. Sources: the general, sending half or all but one; a castle we own with at least 4; or one of our big stacks.

Pays back: the first formula. A castle makes half a soldier per turn, so with L turns left, after a walk of d, it makes half of L minus d. That must be at least the garrison g plus the walk d. Multiply by two and you get the formula; we also cap it at 400 turns, so very long games don't make every castle look worth it.

Which first: the second formula. Cheap and close first; castles towards the enemy general get a bonus, 0.6 per square closer, because they become strike bases; enemy castles get 15, because taking one hurts them twice; and a full send costs 2 more than a half send. The bot checks the best four options each turn.

Safe: the third formula. Fast-forward the board to the end of the walk, move every enemy walk d steps closer, and the safety score must still clear the pad plus 20 percent of the enemy's biggest stack. And if the soldiers come from the general, the reserve rule applies too.

Needed: the castle gate. Project both armies to the end of the game at current income. If our lead is more than a quarter of their army, and our income and land are at least as big, the race is won and only cheap castles are taken: garrison plus walk at most 30, never from the general. Unless the opponent holds or masses on castles, which opens the gate again.""")
header(s, "3 · Every part of v51", "Castles in full")
y = 1.55
for words, e in (("Pays back: a castle makes ½ a soldier per turn (g garrison, d walk, L turns left)", "cpay"),
                 ("Which first: cheap and close; toward their general; enemy castles", "ckey"),
                 ("Safe: fast-forward to the end of the walk", "wsafe"),
                 ("Needed? The castle gate: project both armies to the end of the game", "cgate")):
    text(s, 0.7, y, 7.6, 0.35, words, size=13.5, color=INK2)
    pic = eq(s, e, 0.75, y + 0.38, h=0.34, maxw=7.5)
    y += 0.38 + pic.height / 914400 + 0.25
panel(s, 8.7, 1.55, 4.0, 3.6, PANEL2)
text(s, 8.95, 1.7, 3.6, 5.1, [[("Candidates: ", {"bold": True, "color": BRASS}), ("castles within 30 squares, walk at most 25. Best 4 checked each turn.", {})],
                              [("Sources: ", {"bold": True, "color": BRASS}), ("the general (half, or all but one), an owned castle with 4+, or a big stack.", {})],
                              [("From the general: ", {"bold": True, "color": BRASS}), ("the reserve rule applies too.", {})],
                              [("Race won: ", {"bold": True, "color": BRASS}), ("only cheap castles (garrison + walk ≤ 30), never from the general, unless they hold castles.", {})]],
     size=13.5, after=10)

# --- expansion and the rule list ---------------------------------------------------------------------------------------
s = new("""SAY: Two last pieces of everyday play.

Expansion, the formula. For every square we could step into, a score: 10 for a free square, 20 for an enemy square we can beat; a little for the size of the stack moving; half a point for each free square next to the target, so it heads into open land; a small penalty for being far from the enemy general; minus 5 if the soldiers would leave a general or castle; plus 4 if the square is on an enemy attack route, which blocks it. Highest score wins.

The rule list, used when things are urgent or when a plan's move fails the test. It goes top to bottom and takes the first move that passes the safety test: a proven attack; if an attack is under way and we're short, hit an adjacent attacker we can beat, or pull the best stack home; pull home if we're short but nothing is moving yet; continue or start a castle walk; concentrate for the kill if we're far ahead; expand; walk a stack to the edge of our land; build up on the forward castle if we're very safe; let the general expand, with half if it has more than 12; and finally gather home. It runs at most 6 of the expensive tests per turn, to stay fast. If nothing passes: the defender script.""")
header(s, "3 · Every part of v51", "Everyday play: expansion and the rule list")
text(s, 0.7, 1.55, 11.9, 0.35, "Expansion: score every square we could step into, take the best", size=14, color=INK2)
panel(s, 0.7, 1.95, 11.9, 0.75, PANEL2)
eq(s, "expand", 0.95, 2.12, h=0.34, maxw=11.4)
text(s, 0.7, 2.85, 11.9, 0.4, "gain: 1 for a free square, 2 for an enemy square we beat. The threat-route bonus blocks attack paths.",
     size=12.5, color=INK3)
text(s, 0.7, 3.4, 11.9, 0.3, "THE RULE LIST (v36): FIRST MOVE THAT PASSES THE SAFETY TEST WINS", size=12, color=BRASS, bold=True)
rules = ["proven attack", "under attack: hit or pull home", "short: pull home", "castle walk", "far ahead: concentrate",
         "expand", "walk to the edge", "build up forward", "general expands", "gather home"]
for i, r_ in enumerate(rules):
    col, row = divmod(i, 5)
    x = 0.7 + row * 2.4
    y = 3.8 + col * 1.05
    panel(s, x, y, 2.25, 0.88, PANEL if (i % 2 == 0) else PANEL2)
    text(s, x + 0.12, y + 0.08, 0.5, 0.4, str(i + 1), size=15, font=HEAD, color=BRASS)
    text(s, x + 0.12, y + 0.42, 2.05, 0.45, r_, size=12.5)
text(s, 0.7, 6.05, 11.9, 0.8, "At most 6 expensive tests per turn. If nothing passes: the defender script (hit an adjacent attacker we beat, else pull the best stack home, else wait).",
     size=13, color=INK2)

# --- switched off -------------------------------------------------------------------------------------------------------
s = new("""SAY: Last, one part that's built into the bot but switched off, so you know it exists.

Across games, the bot could learn which of four settings works best against a particular opponent: base, greedy, safe, and castle-hungry. That's a multi-armed bandit, from the book's chapter on sequential decisions, and we used the Exp3 algorithm. The first formula: each game, pick a setting with probability mostly proportional to its weight, plus a 20 percent share spread evenly so that every setting keeps getting tried. The second: after the game, multiply the chosen setting's weight up according to the reward, divided by the probability it had, so rarely picked settings aren't unfairly ignored. Reward: 1 for a win, 0 for a loss, otherwise between 0.25 and 0.75 by share of the land.

Why it's off: it only switches on if you give it a file to remember results in. One game's result is very noisy, it would need many games against the same opponent to learn anything, and it would make our test matches depend on the order games were played in. So in every game you'll see today, it plays the base setting.""")
header(s, "3 · Every part of v51", "Built, but switched off: learning across games")
text(s, 0.7, 1.55, 6.2, 1.0, "Across games, pick one of four settings and learn which works against this opponent (Exp3).",
     size=15)
for i, (a_, d_) in enumerate((("base", "reserve +0, cheap castles ≤ 30"), ("greedy", "reserve −0.1, cheap castles ≤ 60"),
                              ("safe", "reserve +0.1, cheap castles ≤ 20"), ("castles", "reserve +0, cheap castles ≤ 90"))):
    y = 2.45 + i * 0.58
    panel(s, 0.7, y, 6.2, 0.5, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.9, y + 0.12, 1.4, 0.3, a_, size=13.5, font=HEAD, color=BRASS)
    text(s, 2.4, y + 0.13, 4.4, 0.3, d_, size=13, color=INK2)
text(s, 0.7, 4.95, 6.2, 1.9, [[("Off in the final bot: ", {"bold": True, "color": BRASS}),
                              ("one game is noisy evidence, it would need many games against the same opponent, and test results would depend on game order. Every game today uses ", {}),
                              ("base", {"bold": True}), (".", {})]], size=14, color=INK2)
y = 1.6
for words, e in (("Pick a setting: mostly by weight, 20% spread evenly", "exp3p"),
                 ("After the game: reward r, scaled by how likely the pick was", "exp3w")):
    text(s, 7.3, y, 5.4, 0.4, words, size=13.5, color=INK2)
    pic = eq(s, e, 7.35, y + 0.4, h=0.38, maxw=5.2)
    y += 0.4 + pic.height / 914400 + 0.35
text(s, 7.3, y, 5.4, 0.6, "r = 1 win, 0 loss, else 0.25 + 0.5 × our share of the land.", size=13, color=INK2)
term(s, 7.3, 5.3, 5.4, 1.5, "Multi-armed bandit (AIMA ch. 17)",
     "Repeatedly choose between options with unknown payoffs, balancing trying each (exploration) against using the best so far (exploitation).",
     "This is a multi-armed bandit from chapter 17: repeated choices between options with unknown payoffs, where you balance trying options against using the best one found so far. Exp3 is a version that works even against an opponent that adapts.")
