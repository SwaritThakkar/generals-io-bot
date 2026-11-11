# Section 1: how we got here. Executed inside build3.py (helpers already defined).

section("1", "How we got here", "Fifty versions, three changes of direction. What the five papers from our first two "
        "presentations gave us, and the games that made us change course.", "v11_trail.png",
        """SAY: Before I explain v51, a few minutes on how we got to it, because v51 is not where we started.

In our first two presentations we picked five search papers and planned to build the bot from them. We built most of them. They worked as designed, and the bot did not get stronger.

I'll show you the evidence for that, then the three moments where real games forced us to change direction. The board on the right is one of those games: our old bot is red, and it is about to lose.""")

# --- the five papers -----------------------------------------------------------
s = new("""SAY: Here are the five papers from our first two presentations. For each one: what it is for, what it quietly assumes, and what we found when we measured our game.

ARA* finds a path quickly and improves it while time remains. It assumes finding a path is the hard part. In our game a complete distance map over all 529 squares takes four hundredths of a millisecond.

RTAA* and depression avoidance learn better distance estimates while the agent moves, and escape the dead ends that wrong estimates create. They assume you don't know the distances. Our map is fully visible, so distances are exact from the first turn and there is nothing to learn.

ProbCut lets you search deeper by skipping branches a shallow search already rules out. It assumes deeper search wins more games. I'll show you on the next slide that it didn't.

Metareasoning decides how long to think. It assumes thinking time is scarce. Our final bot uses 10 milliseconds out of 500.

So all five make search faster or deeper. Our games were being lost for a different reason, and none of the five papers looks at it: the general being left without enough soldiers.""")
header(s, "1 · How we got here", "The five papers we started with")
cols = [(0.7, 2.6, "Paper"), (3.4, 3.2, "What it is for"), (6.75, 2.85, "What it assumes"), (9.75, 2.95, "What we measured")]
for x_, w_, h_ in cols:
    text(s, x_, 1.6, w_, 0.3, h_, size=12.5, color=BRASS, bold=True)
rows = [("ARA*", "Likhachev et al. 2003", "find a path fast, improve it while time is left", "finding the path is the hard part",
         "a full distance map takes 0.04 ms"),
        ("RTAA*", "Koenig & Likhachev 2006", "learn better distance estimates while moving", "distances are unknown",
         "the map is visible: distances are exact"),
        ("Depression avoidance", "Hernández & Baier 2012", "escape dead ends of a wrong estimate", "estimates can be wrong",
         "exact distances have no dead ends"),
        ("ProbCut", "Buro 1995", "search deeper by skipping hopeless branches", "deeper search wins more",
         "deeper search won no more games"),
        ("Metareasoning", "Russell & Wefald 1991", "decide how long to think", "thinking time is scarce",
         "we use 10 ms of the 500 ms allowed")]
for i, (p_, cite, for_, ass, found) in enumerate(rows):
    y = 2.0 + i * 0.86
    panel(s, 0.6, y, 12.2, 0.76, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.75, y + 0.1, 2.6, 0.6, [[(p_, {"font": HEAD, "size": 15})], [(cite, {"size": 10.5, "color": INK3})]], after=0)
    text(s, 3.4, y + 0.2, 3.2, 0.5, for_, size=13)
    text(s, 6.75, y + 0.2, 2.85, 0.5, ass, size=13, color=INK2)
    text(s, 9.75, y + 0.2, 2.95, 0.5, found, size=13, color=BRASS)
text(s, 0.7, 6.4, 11.9, 0.6, "All five make search faster or deeper. Our games were lost for another reason: a general left "
     "without enough soldiers.", size=15.5)

# --- proof 1: probcut ------------------------------------------------------------------
s = new("""SAY: First piece of evidence: ProbCut.

ProbCut's idea is that a shallow search predicts a deep search. The formula at the top: the deep value v is roughly a times the shallow value v-prime plus b, plus some noise. If the prediction is already clearly outside the window you care about, you skip the deep search and spend the time going deeper elsewhere.

We measured this on our game. The scatter on the left is 160 real positions: shallow value against deep value. They sit almost exactly on a line, correlation 0.996.

That sounds like great news for ProbCut. But read it the other way: if the shallow search already tells you the deep answer almost exactly, then searching deeper hardly ever changes your decision.

And that's what the bar chart shows. Every bot we built from these ideas, played against our best bot of that time, v11. ProbCut variants, four-ply search, learned ProbCut: none of them beats a coin flip. The red bar is v36, the first bot built on a different idea, the safety rule. It won every game against v11.""")
header(s, "1 · How we got here", "Evidence 1: searching deeper didn't help")
panel(s, 0.7, 1.55, 11.9, 0.75, PANEL2)
eq(s, "probcut", 0.95, 1.72, h=0.4, maxw=11.4)
img(s, ART / "probcut_scatter.png", 0.7, 2.5, h=3.55)
text(s, 0.7, 6.15, 4.7, 0.9, [[("Shallow predicts deep almost exactly. ", {"bold": True, "color": BRASS}),
                               ("160 positions; ", {}), ("v = 0.955 v′ + 1.97, σ = 2.18.", {"color": INK2})]], size=12.5)
img(s, ART / "paper_era.png", 5.8, 2.5, w=6.9)
text(s, 5.8, 6.15, 6.9, 0.9, [[("Win rate against v11, our best bot at the time. ", {"bold": True, "color": BRASS}),
                               ("Search-paper bots (grey) stay at a coin flip or below. v36 (red) is the first bot built on the safety rule.", {"color": INK2})]],
     size=12.5)

# --- proof 2: time ------------------------------------------------------------------
s = new("""SAY: Second piece of evidence: time.

Three of the five papers, RTAA*, depression avoidance and metareasoning, are about being short of time or short of information. The chart is on a log scale. The game gives each move 500 milliseconds. Our final bot needs about 10. One exact distance map takes 0.04 milliseconds.

RTAA*'s update rule is the first formula: it raises the distance estimate h of the states it searched, using what the search found. But when your estimate is already exact, h equals h-star, the update gives back exactly the same number. There is nothing to learn, and no wrong estimates means no depressions to avoid.

Metareasoning, the last formula, says: think more only if the expected improvement in your decision is worth the time. Here the time is almost free, and as we just saw, thinking deeper rarely changed the decision. So the value of extra thinking was close to zero from both sides.

These papers are good papers. They are written for robots on unknown maps, and that's not our game.""")
header(s, "1 · How we got here", "Evidence 2: we were never short of time")
img(s, ART / "time_scale.png", 0.7, 1.7, w=6.4)
text(s, 0.7, 4.35, 6.4, 1.3, ["Time allowed per move: 500 ms. The final bot decides in about 10 ms. An exact distance map "
                              "over the whole board: 0.04 ms."], size=15)
term(s, 0.7, 5.5, 6.4, 1.3, "Admissible vs exact heuristic (AIMA ch. 3)",
     "A heuristic h estimates the distance to the goal. When h equals the true distance h*, search goes straight to the goal and there is nothing left to learn.",
     "In chapter 3 terms: these algorithms improve a heuristic, an estimate of distance. Ours is the true distance, computed by breadth-first search, so there is nothing to improve.")
y = 1.65
for words, e in (("RTAA*: raise the estimate of every state it searched", "rtaa"),
                 ("With exact distances the update changes nothing", "rtaa_ex"),
                 ("Metareasoning: think more only if it is worth the time", "voc")):
    text(s, 7.5, y, 5.2, 0.4, words, size=13.5, color=INK2)
    pic = eq(s, e, 7.55, y + 0.4, h=0.36, maxw=5.1)
    y += 0.4 + pic.height / 914400 + 0.35
text(s, 7.5, y + 0.1, 5.2, 1.2, "Time cost close to zero, and deeper thinking rarely changed the move: the value of extra "
     "thinking was close to zero.", size=14, color=BRASS)

# --- proof 3: a real loss -----------------------------------------------------------
s = new("""SAY: Third piece of evidence, and the one that changed our direction: real losses.

We played v11, our best search-era bot, against two simple rush bots on 60 maps. It lost its general in 25 of them.

Here's one, map seed 2. Our bot is red. On the left is turn 98. At turn 81, v11 moved all 40 soldiers out of its general to go expanding. Then, the light red line: for 16 turns that stack circled the same four squares, re-deciding every turn. Meanwhile the rusher, the light blue line, walked 17 squares straight to our general, which had grown back to only 10. Captured at turn 99.

On the right, the same map with v36. It never moved the soldiers out of its general: 39 at home. The rusher's stack, 21, has nowhere useful to go. v36 won this game, and 24 of the 25 that v11 lost.

The search wasn't the problem. The problem was that nothing in the bot asked: if I move these soldiers, can my general still be taken?""")
header(s, "1 · How we got here", "Evidence 3: how the search-era bot actually lost")
img(s, SH / "v11_trail.png", 0.7, 1.5, h=4.3)
img(s, SH / "v36_98.png", 5.15, 1.5, h=4.3)
text(s, 0.7, 5.9, 4.3, 1.1, [[("v11, turn 98. ", {"bold": True, "color": RED}),
                              ("At turn 81 it moved 40 soldiers out of its general. Its stack circles (light red); the rusher walks straight in (light blue).", {"color": INK2})]],
     size=12.5)
text(s, 5.15, 5.9, 4.3, 1.1, [[("v36, same map, turn 98. ", {"bold": True, "color": RED}),
                               ("39 soldiers stay in the general. The rusher's 21 can't break it.", {"color": INK2})]],
     size=12.5)
stat(s, 9.85, 1.6, "25 / 60", "games v11 lost its general to two simple rush bots", RED, w=2.9)
stat(s, 9.85, 3.05, "24 / 25", "of those games won by v36 on the same maps", BRASS, w=2.9)
stat(s, 9.85, 4.5, "turn 99", "v11's general falls: 10 soldiers against 21", BLUE, w=2.9)

# --- pivot 1 --------------------------------------------------------------------------
s = new("""SAY: So the first change of direction, v36: a safety rule before anything else.

The rule is the first formula. For every enemy stack s: what my general has now, plus what it grows while s walks over, must be more than what s would arrive with.

Apply it to the move v11 made at turn 81. If the general keeps 1 soldier, it has 1 plus 9 growth, 10, when an enemy stack of 42, 18 squares away, could arrive with up to 24. 10 is less than 24: the move is refused. If the general keeps its 40, it has 49 against 24, safe.

Everything else in the bot is allowed only if it keeps this rule. The chart shows the effect: our search-era bots against the direct rusher, small test sets, between 25 and 83 percent. v36: every game, 20 out of 20.

This was the moment we stopped trying to make search deeper and started modelling what actually loses games.""")
header(s, "1 · How we got here", "Change 1: a safety rule before anything else (v36)")
text(s, 0.7, 1.6, 6.0, 0.4, "In words: for every enemy stack, my general must still win if that stack walks in.", size=15)
panel(s, 0.7, 2.15, 6.0, 0.8, PANEL2)
eq(s, "inv36", 0.95, 2.33, h=0.4, maxw=5.5)
text(s, 0.7, 3.2, 6.0, 0.4, "v11's move at turn 81, if the general keeps 1:", size=13.5, color=INK2)
eq(s, "v11_ex", 0.75, 3.6, h=0.36, maxw=5.9)
text(s, 0.7, 4.3, 6.0, 0.4, "If it keeps its 40:", size=13.5, color=INK2)
eq(s, "v36_ex", 0.75, 4.7, h=0.36, maxw=5.9)
term(s, 0.7, 5.5, 6.0, 1.3, "Constraint before optimisation",
     "First rule out every move that breaks a hard constraint, then choose the best of what is left. Here the constraint is: the general must survive any single enemy walk.",
     "The general principle: rule out the moves that break a hard constraint first, then optimise among the rest.")
img(s, ART / "rusher_dark.png", 7.1, 1.65, w=5.6)
text(s, 7.1, 5.0, 5.6, 1.4, [[("Win rate against the direct rusher. ", {"bold": True, "color": BRASS}),
                              ("Search-era bots (grey) on 4 to 6 game sets; v36 (red) on 20 games.", {"color": INK2})]], size=13)

# --- pivot 2 --------------------------------------------------------------------------
s = new("""SAY: The second change came from people. After v37 we played against it ourselves, recorded the games, and found a way to beat it.

These are two moments from a real human win. Here the bot is red and the human is blue; the gold boxes are the human's biggest stack and the bot's general.

At turn 520 the human's biggest single stack is 605, smaller than the bot's general at 672. So by the old rule, one stack at a time, the bot felt safe. But the human owned twice the castles and had soldiers spread over much more land. By turn 845 the human has collected everything into one stack of 1,694 against a general of 1,323, and wins.

The old rule, the first formula, looks at one stack at a time. The fix, the second formula, which is the safety score from idea 2: count everything the enemy can collect and bring to my general in T moves, and everything I can bring home, for every T. That's v42, and it's where the Human.exe paper came in.""")
header(s, "1 · How we got here", "Change 2: people beat it by collecting (v42)")
img(s, SH / "human_520.png", 0.7, 1.5, h=3.55)
img(s, SH / "human_845.png", 4.4, 1.5, h=3.55)
text(s, 0.7, 5.1, 3.55, 0.6, [[("Turn 520. ", {"bold": True, "color": BLUE}), ("Human's biggest stack 605, bot's general 672.", {"color": INK2})]], size=12)
text(s, 4.4, 5.1, 3.55, 0.6, [[("Turn 845. ", {"bold": True, "color": BLUE}), ("Collected: 1,694 against 1,323.", {"color": INK2})]], size=12)
panel(s, 0.7, 5.8, 11.9, 1.25, PANEL2)
text(s, 0.95, 5.9, 2.2, 0.4, "Old rule", size=12.5, color=INK2, bold=True)
eq(s, "old_margin", 2.6, 5.88, h=0.36, maxw=8.5)
text(s, 0.95, 6.5, 2.2, 0.4, "New rule (idea 2)", size=12.5, color=BRASS, bold=True)
eq(s, "new_margin", 2.6, 6.47, h=0.36, maxw=8.5)
text(s, 8.4, 1.6, 4.3, 0.4, "Here the bot is red, the human blue.", size=12.5, color=INK3, italic=True)
text(s, 8.4, 2.1, 4.3, 2.6, ["No single human stack was bigger than the bot's general, so the old rule said: safe.",
                             "But the human held more land and castles 4 to 0, and collected it all into one stack.",
                             "New rule: count everything that can be collected in T moves, on both sides."],
     size=14.5, after=10)
text(s, 8.4, 5.1, 4.3, 0.6, "Borrowed from Human.exe (Straka et al. 2026).", size=12.5, color=BRASS)

# --- pivot 3 --------------------------------------------------------------------------
s = new("""SAY: The third change is v51 itself. v46, the version before, still lost to one pattern: build up quietly, then charge with one big stack.

The two charts are the same game, same map, same opponent. Red is our general, blue is the attacker's biggest stack.

On the left, v46. The attacker builds up from turn 200 and charges. v46 noticed the danger, defended for one turn, saw the safety score recover, went back to expanding, and so on. Its general stayed in the 70s and fell at turn 237, 77 against 82. At that moment it had 400 soldiers spread over the board and two stacks only three squares away.

On the right, v51. Once the attack is clearly coming, the grey band, it commits: every move pulls soldiers into the general until the attack is over. The general climbs to 124 and the attack fails.

The formula at the bottom is the whole difference: v46 decided to defend turn by turn; v51 decides once and keeps defending while the attack is coming. That's idea 4, and I'll come back to this exact game later.""")
header(s, "1 · How we got here", "Change 3: defend once, and keep defending (v51)")
img(s, ART / "trace_pair_dark.png", 0.7, 1.5, w=11.9)
panel(s, 0.7, 6.15, 11.9, 0.75, PANEL2)
eq(s, "flipflop", 0.95, 6.33, h=0.36, maxw=11.4)

# --- the path ------------------------------------------------------------------------
s = new("""SAY: So here's the whole path on one slide. Each line is a version, what it changed, and the evidence that made us do it.

Versions 1 to 35 followed the papers and never beat a coin flip against our own best bot. v36 added the safety rule after v11 lost 25 of 60 games to rush bots. v37 replaced single moves with plans and a 40-turn lookahead with a learned win estimate. v39 added a rule about swapping places with an attacker, found by replaying games humans won. v42 and v44 count collected armies, after a human collected 1,694 soldiers. v46 guesses the opponent's style. And v51 commits to defence, after v46 lost the game I just showed.

Every change after v35 came from watching a game we lost. That's the main lesson of the project, and now let's look at how v51 actually works.""")
header(s, "1 · How we got here", "The whole path: every change came from a lost game")
steps = [("v1–v35", "faster, deeper search: the five papers", "no better than a coin flip against v11"),
         ("v36", "safety rule: never leave the general beatable", "v11 lost 25 of 60 to rush bots"),
         ("v37", "plans, 40-turn lookahead, learned win estimate", "v36 stalled: 16 of 20 against land grabbers"),
         ("v39", "never swap places with an attacker", "replays of games humans won"),
         ("v42–v44", "count collected armies; reserve; counter-strike", "a human collected 1,694 against 1,323"),
         ("v46", "guess the opponent's style", "17–3 against v37"),
         ("v51", "commit to defence; trained style classifier", "v46 lost the gather-and-charge game")]
for x_, h_ in ((0.95, "Version"), (2.7, "What changed"), (8.6, "Why")):
    text(s, x_, 1.55, 4, 0.3, h_, size=12.5, color=BRASS, bold=True)
for i, (v_, d_, r_) in enumerate(steps):
    y = 1.95 + i * 0.7
    panel(s, 0.7, y, 11.9, 0.6, PANEL if i % 2 == 0 else PANEL2)
    text(s, 0.95, y + 0.13, 1.6, 0.4, v_, size=15, font=HEAD, color=BRASS)
    text(s, 2.7, y + 0.16, 5.8, 0.4, d_, size=14)
    text(s, 8.6, y + 0.16, 3.9, 0.4, r_, size=13, color=INK2)
