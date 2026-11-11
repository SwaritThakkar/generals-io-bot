"""Reorder and annotate the hand-edited v51 deck using standard OOXML only.

This preserves the source deck and all existing visual assets. Only the slide
list, selected text, selected notes, and the paper-only bibliography slide are
changed. No python-pptx dependency is used.
"""
from __future__ import annotations
import copy, os, re, zipfile, xml.etree.ElementTree as ET

SRC = "presentation/v51-talk.pptx"
OUT = "presentation/v51-talk-revised.pptx"
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
for prefix, uri in NS.items():
    if prefix != "rel": ET.register_namespace(prefix, uri)
ET.register_namespace("", NS["rel"])

# Narrative sequence: mechanics; research and failed search; safety evolution;
# v51's decisions; one compressed replay; results and sources.
ORDER = [1,2,3,6,5,7,8,9,12,14,15,18,20,22,25,27,29,31,34,36,39,41,42,43,44,
         45,46,48,50,52,53,54,55,57,58,59]

NOTES = {
1: "This talk explains how v51 responds to a specific failure pattern in the modified Generals game: an opponent waits, gathers on productive ground, and then sends a large stack down a short route. The deck follows the evidence from early search experiments to the committed-defence change, then shows one recorded game and the broader match results. The screenshot is from a real game; red is v51. The claim is deliberately scoped to v51 and the tested opponents in this repository.",
2: "The board is a 23 by 23 grid. The number on a tile is its army, the crown marks a general, and grey squares are mountains. Each side chooses one move at the same time, so neither can react to the other's current move before committing. When armies meet, combat subtracts the smaller army from the larger. Generals and castles grow faster than ordinary owned land. Taking the opposing general ends the game; otherwise the game uses its end condition. These mechanics make time and production part of every attack calculation.",
3: "The picture isolates the key trade-off. A stack that leaves a tile must leave one soldier behind, so a 30-soldier stack walking eight squares arrives with 22 if it fights nothing. Meanwhile a stationary general grows from 12 to 16 over those eight ticks. A full general is therefore difficult to beat with one distant stack. It becomes vulnerable when its army has been moved away, or when several arrivals combine over time. The bot has to predict delivery at a future time, not compare only the largest stacks visible now.",
5: "This slide is a bibliography only. ARA* is by Likhachev, Gordon, and Thrun (2003); RTAA* is by Koenig and Likhachev (2006); depression avoidance is by Hernández and Baier (2012); ProbCut is by Buro (1995); metareasoning is by Russell and Wefald (1991). The full paper titles and venues appear here. The next slides explain what we measured when we tried the search direction.",
6: "This experiment compares a shallow evaluation, written v-prime, with a deeper evaluation, written v. The fitted relation is v approximately a times v-prime plus b plus residual e, with e modelled as a normal random error of standard deviation sigma. In the fit, v is the deep-search score, v-prime is the shallow-search score, a is slope 0.955, b is intercept 1.97, and sigma is 2.18 over 160 saved positions; r = 0.996 is the correlation coefficient. ProbCut's cutoff formula standardizes the predicted deep score against a threshold beta using sigma, then compares it with t, the chosen confidence threshold in standard-deviation units. Here beta is the alpha-beta bound being tested. The close fit meant deeper results seldom changed the ranking. Across tested search variants, match results did not improve enough to address the loss mechanism. This is evidence about these variants and this game setup, not a general result against search.",
7: "The first failure was leaving home unprotected. At tick 81 v11 moved 40 soldiers out of its general. Its large stack then circled nearby tiles while the opponent's smaller stack took the direct route. At tick 99 the general held 10 and the attacker arrived with 21. The same map replay shows v36 keeping 39 at home, enough to survive the 21-soldier attack. Across the recorded sets, v11 lost its general in 25 of 60 rush games, and v36 won 24 of those 25 same-map rematches. This motivated a hard safety constraint before any positional optimization.",
8: "The displayed inequalities compare the general's remaining army and future growth with an enemy stack's delivered strength. In the invariant, G means our general's current army; growth(d) is its production over d ticks; d_s is the travel time of enemy stack s; x_arrive(s) is that stack's army when it reaches the general. In the v11 example, keeping one means one soldier remains at home and floor(18/2) is the general's growth over 18 ticks. In the v36 example, 40 stay home and the attacker loses one per step along an 18-step route. v36 rejects moves that violate this single-stack survival check. The rule fixed the obvious empty-general rush but did not yet model an opponent collecting multiple stacks.",
9: "The human replay shows why checking only one stack is insufficient. At tick 520 the human's largest stack is 605 while the bot's general has 672, so the old comparison appears safe. Yet the human has more land and four castles to the bot's zero; after continued collection, the human brings 1,694 soldiers against 1,323 by tick 845. In the old margin, m_old is the worst single-stack safety score, s indexes an enemy stack, d_s is its travel time, and x_s is the stack's strength when evaluated for that route. In the new margin, G is the general army, growth(T) its production over T ticks, D_us(T) the defending army our side can deliver within T ticks, and D_them(T) the enemy army deliverable in the same budget. The minimum over T is the worst projected margin. This is a greedy estimate inspired by Human.exe's collection modelling, not an exact search over every possible route.",
12: "Each tick follows the six steps shown. The agent reads the fully visible board, updates its estimate of opponent style, detects whether a large stack is approaching, and chooses immediate defence or a longer-term plan. Before acting, it tests whether the move could expose the general. The actions are simultaneous, and the whole decision loop runs within the game's per-tick allowance. Two controls dominate the rest: unsafe moves are rejected, and a detected committed attack keeps the bot in defence until the threat has passed or the shortfall is resolved.",
14: "The classifier estimates one of ten scripted styles from board measurements. The board itself is visible; the uncertainty is the opponent's intent. Features include land, army at the general, largest-stack size, and whether that stack is moving closer. The charts show a delayed gather-rusher: at tick 205 its visible behaviour still resembles a land-grabber; after it empties its general, the later board resembles a build-up-and-attack style. The reported held-out style accuracy is about 70% in the middle game, compared with 10% for ten-way random guessing. That accuracy does not by itself establish a win-rate gain; the head-to-head data later show that removing this classifier did not change the 32–28 record against v46.",
15: "A single board can be ambiguous, so the bot smooths evidence over time. In the equation, ell at tick t for style y is the accumulated log evidence for that style; ell at tick t minus one is the previous log evidence; 0.95 is the memory-retention factor; P(y given x_t) is the classifier probability for style y from the current feature vector x_t; and 0.4 is the weight on this new observation. The log operation turns repeated evidence into an additive update. The belief at tick 205 still favours land-grabbing; after the general empties, the next observation shifts the belief toward a gather-and-attack style. This is a lightweight online belief update, not an inference over all possible strategies.",
18: "This slide makes the safety signal explicit. G is the current general army; growth(T) is production during T ticks; D_us(T minus lead) is the army our side can deliver with a small timing cushion; D_them(T) is the enemy army that can arrive within T ticks; T is a future move horizon; lead is the number of ticks reserved so defence starts early; and m is the minimum projected safety margin over the tested horizons. A low m means danger. The graph illustrates how the margin falls when the opponent begins collecting. The delivery calculation is a greedy approximation because choosing the best overlapping routes under a move limit is combinatorial. It can trigger a response, but it is not a proof that every enemy schedule has been covered.",
20: "The diagram shows the safety gate as a reply check. First the bot simulates its proposed move. If the general remains safe, the move can proceed. If the score is close to the danger threshold, the bot considers plausible enemy routes and simulates its own defence against them. If any tested line loses the general, the move is rejected. The left branch shows an attack the general survives; the final branch shows why one losing reply is enough to discard the candidate. This is bounded verification over selected attacks, not a complete minimax search over all simultaneous moves.",
22: "The fast-defence procedure has three stages. Notice an enemy stack that has made sustained progress toward the general. Measure the shortfall at the expected arrival time: enemy delivery minus the general's expected army, with a margin for uncertainty. If the shortfall is positive, gather. The gold arrows show a plan whose paths all lead into the general. v51 builds a rooted prize-collecting tree: each owned tile offers its army less the one soldier it must leave behind, and each move has a cost. It searches a price per move, greedily adds valuable routes, and executes from outer branches inward. This borrows the collection structure described for Human.exe, adapted to a defensive deadline. Once triggered, the mode is latched so a temporary recovery in the score does not restart expansion while the same stack is still approaching.",
25: "The table compares candidate macro plans against four simplified opponent scripts. A rollout means simulating a plan forward for a fixed horizon. The row scores are win estimates at the end of the rollout, not observed win rates. The Hurwicz formula mixes expected value and worst-case value: alpha is one half, EV is the belief-weighted average score, and MM is the minimum score across opponents. Equivalently, q indexes a candidate plan; m indexes an imagined opponent; p_m is the current probability weight for opponent m; V(q,m) is the learned value after simulating plan q against m; and the minimum term is the worst score. This is inspired by Puppet Search's use of scripts and choice points, but it is a project-specific macro-plan search rather than a full implementation of that paper.",
27: "Castles are treated as investments. They cost a garrison and travel time, but then produce army faster than ordinary land. The bot estimates whether the new production can repay the soldiers committed before the game ends, and checks whether the general remains safe after the trip. Churchill and Buro's StarCraft build-order work motivates thinking in terms of future income and payback; v51 does not use their optimizer. The image shows the first castle taken in the recorded game, after the gather-rusher attack has failed and home defence is no longer urgent.",
29: "The opponent model changes preferences but does not remove the safety gate. When aggressive behaviour is likely, the bot retains a larger home reserve and may allow a counter-attack; when the opponent appears quiet, it can spend more on land and castles. When the bot is behind, it can pursue productive castles; when clearly ahead, it selects cheaper opportunities. Ganzfried and Sandholm's paper studies safe deviations from equilibrium in repeated zero-sum games. v51 borrows the principle of limiting exposure to a mistaken read, but it does not implement that paper's formal algorithm or inherit its guarantee. The reliable protection here comes only from the bot's own threat estimates and tests.",
31: "The attack check accounts for the same attrition and production that make defence difficult. The attacking stack loses soldiers along its route and to defenders; the target general grows and may receive reinforcements. The bot also checks whether sending its own stack away would expose its home general. It then tests the attack against several scripted replies before launching. The image shows the late-game example: at tick 405 the opposing general has only two soldiers, so the projected attack can pass its checks; the capture occurs at tick 407. The method is deliberately conservative and can attack late against a careful opponent.",
34: "This is the full priority order. Update the opponent belief; take an immediately winning general capture; trigger committed defence when a large stack approaches and the estimated shortfall is positive; update reserve and threat gates; consider a counter-strike only if home is projected to fall and the strike passes its own test; apply urgent safety rules; consider a castle if its payback and safety checks pass; otherwise follow the committed macro plan. Every proposed castle or plan move is tested first. If no candidate survives, the bot uses a simple defensive fallback. This ordering prevents lower-priority economic plans from overriding an urgent home defence.",
36: "The plan menu keeps the rollout branching factor small. Plans include expansion, holding army, taking one of a few nearby castles, gathering home, striking from a forward castle, or rushing when the attack is already favourable. The imagined opponents are a direct rusher, castle rusher, land-grabber, and waiter. Eight candidate plans against four scripts gives at most 32 imagined games before identical outcomes are merged. In the ranking formula, c is a candidate castle, d(c) is its travel distance, and garrison(c) is the neutral army guarding it; lower distance and lower garrison reduce the estimated cost. The ranking is only an ordering heuristic; safety and payback tests still decide whether a plan may be played.",
39: "The learned win estimate scores rollout endpoints. The training data contain scripted games on many maps, with positions labelled by who eventually won. Separate models are trained for opening, middle, and end phases. The slide reports held-out prediction quality: the richer feature model helps in opening and endgame, but is no better than land alone in the middle phase. In Newton's update w is the vector of model weights, H is the Hessian (the matrix of second derivatives of the regularized loss), and the gradient is the vector of first derivatives; the minus sign moves downhill. In the displayed loss, n is the number of training samples, X is the feature matrix, p is the vector of predicted win probabilities, y is the vector of binary outcomes, lambda is the penalty on large weights, I is the identity matrix, and S is the diagonal matrix of logistic variances p_i(1-p_i). The model evaluates short rollouts; it is not a policy that directly chooses moves.",
41: "The network has fourteen input measurements and ten output styles, with two hidden layers of 64 units each. Inputs summarize the opponent's land and army, how much army remains at its general, the largest stack and its distance, recent movement and growth, castles, and evidence of previous general-emptying. In the architecture expression, x is the 14-value input vector; y is an opponent style; P(y given x) is its predicted probability; W1, W2, and W3 are learned weight matrices; b1, b2, and b3 are learned bias vectors. ReLU keeps positive hidden activations and sets negative ones to zero; softmax turns the ten final scores into probabilities. The widths 14, 64, 64, and 10 describe the layers. For normalized input x-hat, mu is the per-feature training mean and sigma is the per-feature training standard deviation. The parameter count is 5,770 learned weights and biases. This is a compact classifier trained on game traces, not a deep reinforcement-learning policy.",
42: "Training uses v46 against ten scripted styles on 33 maps, with both starting seats. Rows are sampled every other tick; the split is by map, with 25 maps for training and eight unseen maps for testing. The reported datasets contain 99,437 training rows and 33,318 held-out rows. The confusion example shows why early prediction is difficult: a rusher that waits at home resembles a turtle until its launch. Economy and planner scripts also look alike, but both map to the same rollout opponent. In the cross-entropy expression, theta denotes all network parameters; N is the number of examples; i indexes examples; y_i is the one-hot true style label; x_i is the feature vector for example i; and P_theta(y_i given x_i) is the predicted probability assigned to its true style. The sum penalizes low probability for the correct style. GPU training made fitting quick, but the classifier did not measurably improve the tested match record.",
43: "The reserve rule addresses a failure found in a human game: the general could look safe until too late because the estimate did not account for a patient collection phase. G_keep is the army left on the general after a proposed move; growth is production over the horizon; H is a tested future horizon; d_s is the travel time of stack s; x_s(H) is its projected strength; and R is the required retained fraction, 0.6 when the threat gate is open and 0.4 otherwise. The second expression estimates a mass threat from the H largest collectible stacks: a_c is the army on a candidate tile, one is the soldier left behind, and cost is attrition or movement cost. The turn-213 image shows the opponent gathering before launch. These are conservative guardrails, not an exact adversarial solution.",
44: "Two edge cases come from replay analysis. First, the general must not step toward an adjacent enemy stack in a simultaneous move: the two can pass, leaving the attacker to land on the general after it moves away. The bot therefore waits or uses another stack to deal with that attacker. Second, if the projected time until our general falls is T_fall, a counter-strike is considered only when our attack arrives sooner and beats the enemy general plus its growth and home reinforcements, with a three-soldier cushion. Here d is our attack travel time, G-prime is the enemy general army, growth(d) is its production, and D-prime_home(d minus one) is enemy reinforcement deliverable before impact. The strike also must pass the attack verification and threat gate.",
45: "The replay follows v51 against a delayed gather-rusher. The opponent expands first, then consolidates its army and charges along a short path toward the general. This is the pattern that exposed v46. The next sequence shows the opening, the moment the opponent empties its general, v51's gathering plan, the impact, and the later counterattack. It is one traced match, useful for explaining the mechanism but not a substitute for the multi-game results.",
46: "At tick 40 both sides are still expanding. The opponent has not assembled an incoming stack, so the defensive detector is quiet and v51 takes ordinary land. The 21 versus 21 army counts show a balanced opening in this game. The point is that committed defence is conditional: it does not mean the bot sits at home from the start. It switches when the board provides evidence of a large stack making progress toward the general.",
48: "At tick 213 the opponent's general empties and a 94-soldier stack appears. The safety estimate drops to 12, showing that future delivery is now a near-term concern even though the stack has not yet reached the general. This is the turning point from ordinary expansion to threat response. The detector uses both stack size and recent distance change, so a large stationary army is not automatically treated as an attack.",
50: "At tick 219 the stack is 18 squares away and still approaching. The threat is now confirmed. v51 computes a gathering route with fourteen moves and commits to defence. The gold arrows show the route plan into the general. The measured distance is an approximate arrival time because each player moves once per tick; actual travel can be affected by fighting and route interactions. The latch prevents the planner from abandoning defence after one favourable score update.",
52: "At tick 235 a nearby stack joins the general. The general now holds 124 soldiers; the attacker has 90 and is two squares away. This is the intended effect of collecting on a schedule: distant soldiers move first and can accumulate before the attack lands. The decision was made before impact, not as a reaction after seeing the opponent's move, which matters because both players act simultaneously.",
53: "At tick 237 the attacker reaches the general with 88 soldiers, and the general retains 38. The preceding bot, v46, lost this same position at 77 against 82. This comparison isolates why a persistent gathering plan matters: the difference is not a deeper tactical line at impact, but having enough army at the target before impact. The example is one seed and should be read alongside the larger match suite.",
54: "After surviving, v51 resumes economic play and later attacks when the opponent's general is nearly empty. The image summarizes tick 407, after the enemy general had fallen to two soldiers at tick 405. The attack passed the checks against the tested replies, and the game ended by capture. This shows the full cycle: defend the collection rush, return to expansion, then attack when the target is genuinely weak. Against a careful opponent who keeps its general full, v51 may wait too long for this opening.",
55: "The timeline condenses the same game. Early play is expansion; the opponent then gathers; v51 detects the charge and pulls soldiers home; the attack fails; after taking a castle and improving its economy, v51 captures the enemy general. The notes on the individual frames give the exact counts and ticks. The timeline is illustrative rather than statistical; the aggregate table is the proper evidence for performance across games.",
57: "The table compares v51 or its classifier-free fast-defence counterpart with v46. Against delayed gather-rushers, the strongest measured gains are 19 wins out of 20 at a tick-200 start versus 12/20, 20/20 versus 18/20 at tick 120, and 19/20 versus 16/20 at tick 60. It also won all 40 tested games against direct and castle rushers. Against spread-and-collect play, it was slightly worse: 13/20 versus 14/20. In 60 games against v46, v51 went 32–28; v52 without the classifier had the same record. The evidence supports a targeted defence improvement and exposes an opportunity cost against spread collectors; it does not establish universal superiority.",
58: "The final prompt invites a live game. The command starts the local arena and the selected entry runs the v51 champion. The audience should watch for three behaviours: it expands early, it gathers home when a large stack approaches, and it usually waits for the enemy general to empty before attacking. Spreading out and taking castles is a different test because it can exploit the defender's time at home. This is the known weakness suggested by the results, not a guaranteed winning strategy against every setup.",
59: "The references are the research sources discussed throughout the talk. ARA*, RTAA*, depression avoidance, ProbCut, and metareasoning motivated the initial search experiments. The remaining papers inform plan search, opponent prediction, safe exploitation, economic projection, and the collection structure associated with Human.exe. The talk distinguishes those papers' original problem settings from v51's narrower adaptations; in particular, it does not claim to reproduce the full Human.exe or Puppet Search systems.",
}

# Keep source links in the notes so claims can be checked during the talk.
NOTES[5] += " Sources: Likhachev et al., https://papers.nips.cc/paper_files/paper/2003/hash/ee8fe9093fbbb687bef15a38facc44d2-Abstract.html; Koenig & Likhachev, https://publications.ri.cmu.edu/real-time-adaptive-a; Hernández & Baier, https://doi.org/10.1613/jair.3590; Buro, https://journals.sagepub.com/doi/pdf/10.3233/ICG-1995-18202."
NOTES[6] += " ProbCut source: https://journals.sagepub.com/doi/pdf/10.3233/ICG-1995-18202."
NOTES[9] += " Human.exe/self-play paper: https://arxiv.org/abs/2606.23348."
NOTES[14] += " Sources: Synnaeve & Bessière, https://arxiv.org/abs/1111.3735; Weber & Mateas, https://alumni.soe.ucsc.edu/~bweber/pubs/cig_2009.pdf."
NOTES[15] += " Background paper on Bayesian RTS strategy prediction: https://arxiv.org/abs/1111.3735."
NOTES[18] += " Human.exe/self-play paper: https://arxiv.org/abs/2606.23348."
NOTES[22] += " Human.exe/self-play paper: https://arxiv.org/abs/2606.23348."
NOTES[25] += " Puppet Search paper: https://ojs.aaai.org/index.php/AIIDE/article/view/12779."
NOTES[27] += " Build-order paper: https://ojs.aaai.org/index.php/AIIDE/article/view/12435."
NOTES[29] += " Safe exploitation paper: https://www.cs.cmu.edu/~sandholm/safeExploitation.ec12.pdf."
NOTES[59] += " Reference links: https://arxiv.org/abs/2606.23348; https://ojs.aaai.org/index.php/AIIDE/article/view/12779; https://arxiv.org/abs/1111.3735; https://alumni.soe.ucsc.edu/~bweber/pubs/cig_2009.pdf; https://ojs.aaai.org/index.php/AIIDE/article/view/12435; https://www.cs.cmu.edu/~sandholm/safeExploitation.ec12.pdf."

def parse(data):
    return ET.fromstring(data)

def xml_bytes(root):
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)

def get_slide_id_map(root, rels):
    targets = {}
    for rel in rels.findall("{%(rel)s}Relationship" % NS):
        targets[rel.get("Id")] = rel.get("Target")
    out = {}
    for sid in root.find("p:sldIdLst", NS):
        rid = sid.get("{%s}id" % NS["r"])
        target = targets[rid]
        m = re.search(r"slide(\d+)\.xml$", target)
        if m: out[int(m.group(1))] = copy.deepcopy(sid)
    return out

def all_text(el):
    return "".join(t.text or "" for t in el.findall(".//a:t", NS))

def set_shape_text(shape, text):
    tx = shape.find("p:txBody", NS)
    if tx is None: return
    body = tx.find("a:bodyPr", NS)
    lst = tx.find("a:lstStyle", NS)
    proto = next(iter(tx.findall("a:p", NS)), None)
    for p in list(tx.findall("a:p", NS)): tx.remove(p)
    if proto is None:
        proto = ET.Element("{%s}p" % NS["a"])
        ET.SubElement(proto, "{%s}r" % NS["a"])
        ET.SubElement(proto[0], "{%s}t" % NS["a"])
    paras = text.split("\n") if text else [""]
    for line in paras:
        p = copy.deepcopy(proto)
        for child in list(p):
            if child.tag in ("{%s}r" % NS["a"], "{%s}fld" % NS["a"], "{%s}br" % NS["a"]): p.remove(child)
        r = ET.Element("{%s}r" % NS["a"])
        rPr = ET.Element("{%s}rPr" % NS["a"], {"lang":"en-US"})
        r.append(rPr)
        t = ET.SubElement(r, "{%s}t" % NS["a"])
        t.text = line
        ppr = p.find("a:pPr", NS)
        p.insert(1 if ppr is not None else 0, r)
        if not p.findall("a:endParaRPr", NS): ET.SubElement(p, "{%s}endParaRPr" % NS["a"], {"lang":"en-US"})
        tx.append(p)

def replace_notes(root, note):
    body_shape = None
    for shape in root.findall(".//p:sp", NS):
        ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
        if ph is not None and ph.get("type") == "body":
            body_shape = shape; break
    if body_shape is None: return
    set_shape_text(body_shape, note)

def blank_book_shapes(root):
    parent = {child: p for p in root.iter() for child in p}
    tagged = []
    for shape in root.findall(".//p:sp", NS):
        text = all_text(shape).lower()
        if "aima" in text:
            off = shape.find("p:spPr/a:xfrm/a:off", NS)
            ext = shape.find("p:spPr/a:xfrm/a:ext", NS)
            if off is not None and ext is not None:
                x,y,w,h=[int(off.get(k))/914400 for k in ("x","y")]+[int(ext.get(k))/914400 for k in ("cx","cy")]
                tagged.append((x,y,w,h))
            p = parent.get(shape)
            if p is not None: p.remove(shape)
        elif "from the book" in text or "book and papers" in text:
            p = parent.get(shape)
            if p is not None: p.remove(shape)
    # TERM panels are drawn as a text shape over a separate outlined rectangle.
    # Remove only the matching empty backing rectangle; preserve nearby content.
    for shape in list(root.findall(".//p:sp", NS)):
        if all_text(shape).strip(): continue
        off = shape.find("p:spPr/a:xfrm/a:off", NS)
        ext = shape.find("p:spPr/a:xfrm/a:ext", NS)
        if off is None or ext is None: continue
        x,y,w,h=[int(off.get(k))/914400 for k in ("x","y")]+[int(ext.get(k))/914400 for k in ("cx","cy")]
        for tx,ty,tw,th in tagged:
            if abs(x-(tx-.2))<.04 and abs(y-(ty-.12))<.04 and abs(w-(tw+.4))<.05 and abs(h-(th+.2))<.05:
                p=parent.get(shape)
                if p is not None: p.remove(shape)
                break
    # Remove any whole paragraph that is itself a book citation.
    for para in root.findall(".//a:p", NS):
        s = "".join(t.text or "" for t in para.findall(".//a:t", NS)).lower()
        if "russell" in s and "norvig" in s:
            for t in para.findall(".//a:t", NS): t.text = ""

def set_paper_only_table(root):
    title_shape = None
    for shape in root.findall(".//p:sp", NS):
        if "The five papers we started with" in all_text(shape): title_shape = shape
        # Remove the section tag so the slide itself contains only paper names.
        if "HOW WE GOT HERE" in all_text(shape): set_shape_text(shape, "")
    if title_shape is not None: set_shape_text(title_shape, "Five papers")
    citations = [
        "ARA*: Anytime A* with Provable Bounds — Likhachev et al., NeurIPS 2003",
        "Real-Time Adaptive A* — Koenig & Likhachev, AAMAS 2006",
        "Depression Avoidance — Hernández et al., JAIR 43 (2012)",
        "ProbCut: Selective Extension of Alpha-Beta — Buro, ICCA J. 1995",
        "Principles of Metareasoning — Russell & Wefald, AI Journal 1991",
    ]
    prefixes = ("ARA*", "RTAA*", "Depression avoidance", "ProbCut", "Metareasoning")
    citation_idx = 0
    for shape in root.findall(".//p:sp", NS):
        value = all_text(shape).strip()
        matched = next((p for p in prefixes if value.startswith(p)), None)
        if matched:
            set_shape_text(shape, citations[citation_idx])
            citation_idx += 1
        elif value in ("Paper", "What it is for", "What it assumes"):
            set_shape_text(shape, "")
        elif value.startswith(("find a path", "learn better distance", "escape dead ends", "search deeper", "decide how long", "finding the path", "distances are unknown", "estimates can be wrong", "deeper search wins", "thinking time is scarce", "All five make search")):
            set_shape_text(shape, "")

def renumber_footer(root, number):
    for shape in root.findall(".//p:sp", NS):
        s = all_text(shape).strip()
        off = shape.find("p:spPr/a:xfrm/a:off", NS)
        ext = shape.find("p:spPr/a:xfrm/a:ext", NS)
        if s.isdigit() and off is not None and ext is not None:
            x, y, cx, cy = [int(off.get(k)) for k in ("x","y")] + [int(ext.get(k)) for k in ("cx","cy")]
            if x > 10_800_000 and y > 6_100_000 and cx < 1_000_000 and cy < 500_000:
                for t in shape.findall(".//a:t", NS): t.text = str(number)

with zipfile.ZipFile(SRC, "r") as zin:
    files = {n: zin.read(n) for n in zin.namelist()}

pres = parse(files["ppt/presentation.xml"])
rels = parse(files["ppt/_rels/presentation.xml.rels"])
sid_map = get_slide_id_map(pres, rels)
lst = pres.find("p:sldIdLst", NS)
for child in list(lst): lst.remove(child)
for old_num in ORDER: lst.append(sid_map[old_num])
files["ppt/presentation.xml"] = xml_bytes(pres)

for new_num, old_num in enumerate(ORDER, 1):
    slide_path = f"ppt/slides/slide{old_num}.xml"
    root = parse(files[slide_path])
    if old_num == 5:
        set_paper_only_table(root)
    blank_book_shapes(root)
    if old_num == 1:
        for shape in root.findall(".//p:sp", NS):
            text = all_text(shape)
            if "built from the textbook" in text:
                set_shape_text(shape, "v51: collection-aware defence against wait-and-charge attacks.")
    if old_num == 59:
        for shape in root.findall(".//p:sp", NS):
            if "Book and papers" in all_text(shape): set_shape_text(shape, "Research papers")
    renumber_footer(root, new_num)
    files[slide_path] = xml_bytes(root)
    note_path = f"ppt/notesSlides/notesSlide{old_num}.xml"
    nroot = parse(files[note_path])
    replace_notes(nroot, NOTES[old_num])
    files[note_path] = xml_bytes(nroot)

with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zout:
    for name, data in files.items(): zout.writestr(name, data)
print(f"Wrote {OUT} with {len(ORDER)} slides")
