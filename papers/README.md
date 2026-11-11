# papers/

References used in the project. The PDFs of conference and journal papers are not
redistributed in this repository (publisher copyright); each entry below gives the full
reference and a link. The arXiv papers are included.

## Papers the final bot (v51) is built on

| Reference | Used for | Link |
|---|---|---|
| G. Synnaeve and P. Bessière. *A Bayesian model for plan recognition in RTS games applied to StarCraft.* Proc. AIIDE, pp. 79–84, 2011. | belief over the opponent's strategy | [dblp](https://dblp.org/search?q=Bayesian+model+for+plan+recognition+in+RTS+games+applied+to+StarCraft) |
| B. G. Weber and M. Mateas. *A data mining approach to strategy prediction.* IEEE CIG, 2009. | classifier over opponent strategies | [doi:10.1109/CIG.2009.5286483](https://doi.org/10.1109/CIG.2009.5286483) |
| N. A. Barriga, M. Stanescu and M. Buro. *Puppet Search: Enhancing scripted behavior by look-ahead search with applications to real-time strategy games.* Proc. AIIDE, 2015. | search over a few scripted plans | [pdf](https://skatgame.net/mburo/ps/aiide15-puppet.pdf) |
| M. Straka, V. Lisý and M. Schmid. *Superhuman AI for Generals.io using self-play reinforcement learning.* arXiv:2606.23348, 2026. | game rules; Human.exe's prize-collecting Steiner-tree gathering | [arXiv](https://arxiv.org/abs/2606.23348) · `straka2026_superhuman_generals_selfplay_arXiv.pdf` |
| D. Churchill and M. Buro. *Build order optimization in StarCraft.* Proc. AIIDE, pp. 14–19, 2011. | projecting income forward (castle economy) | [dblp](https://dblp.org/search?q=Build+order+optimization+in+StarCraft) |
| S. Russell and P. Norvig. *Artificial Intelligence: A Modern Approach*, 4th ed. Pearson, 2020. | course textbook | [aima.cs.berkeley.edu](https://aima.cs.berkeley.edu/) |

## The five search papers of the first two presentations

| Reference | Link |
|---|---|
| M. Likhachev, G. Gordon and S. Thrun. *ARA\*: Anytime A\* with provable bounds on sub-optimality.* NIPS, 2003. | [pdf](https://www.cs.cmu.edu/~maxim/files/ara_nips03.pdf) |
| S. Koenig and M. Likhachev. *Real-time adaptive A\*.* AAMAS, 2006. | [pdf](https://idm-lab.org/bib/abstracts/papers/aamas06.pdf) |
| C. Hernández and J. A. Baier. *Real-time adaptive A\* with depression avoidance.* Proc. ICAPS, 2011. | [dblp](https://dblp.org/search?q=Real-Time+Adaptive+A*+with+Depression+Avoidance) |
| M. Buro. *ProbCut: An effective selective extension of the alpha-beta algorithm.* ICCA Journal, 1995. | [pdf](https://skatgame.net/mburo/ps/probcut.pdf) |
| S. Russell and E. Wefald. *Principles of metareasoning.* Artificial Intelligence, 1991. | [doi:10.1016/0004-3702(91)90015-C](https://doi.org/10.1016/0004-3702(91)90015-C) |

## Further reading collected during the project

| Reference | Link |
|---|---|
| M. Straka and M. Schmid. *Artificial Generals Intelligence: Mastering Generals.io with reinforcement learning.* arXiv:2507.06825, 2025. | [arXiv](https://arxiv.org/abs/2507.06825) · `straka2025_artificial_generals_intelligence_zerov3.pdf` |
| A. Bhatia, A. Davis et al. *Generally Genius: A Generals.io agent development and data collection framework.* Proc. AIIDE, 2023. | [dblp](https://dblp.org/search?q=Generally+Genius+Generals.io) |
| D. Churchill, A. Saffidine and M. Buro. *Fast heuristic search for RTS game combat scenarios.* Proc. AIIDE, 2012. | [pdf](https://skatgame.net/mburo/ps/aiide12-combat.pdf) |
| B. Cserna and W. Ruml. *Planning time to think: Metareasoning for on-line planning with durative actions.* Proc. ICAPS, 2017. | [dblp](https://dblp.org/search?q=Planning+Time+to+Think+Metareasoning+Durative+Actions) |
| D. O'Ceallaigh and W. Ruml. *Metareasoning in real-time heuristic search.* Proc. SoCS, 2015. | [dblp](https://dblp.org/search?q=Metareasoning+in+Real-Time+Heuristic+Search) |
| S. Minton, M. Johnston, A. Philips and P. Laird. *Solving large-scale constraint satisfaction and scheduling problems using a heuristic repair method.* Proc. AAAI, 1990. | [pdf](https://cdn.aaai.org/AAAI/1990/AAAI90-003.pdf) |
| R. O. Moraes and L. H. S. Lelis. *Asymmetric action abstractions for multi-unit control in adversarial real-time games.* Proc. AAAI, 2018. | [arXiv](https://arxiv.org/abs/1711.08101) · `moraes2018_asymmetric_action_abstractions_RTS_AAAI.pdf` |
| S. Russell and E. Wefald. *Fine-grained decision-theoretic search control.* Proc. UAI, 1990. | [arXiv](https://arxiv.org/abs/1304.1133) · `russell1990_finegrained_decision_theoretic_search_control_UAI.pdf` |

`human-exe-decomposed.md` is our breakdown of the Human.exe gathering algorithm described in the
2026 paper. `restore-removed.sh` downloads further background papers (A\*, MAPF, RTS search)
from their authors' pages into this folder.
