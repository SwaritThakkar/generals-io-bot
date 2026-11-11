# media/

Twelve figures, AIMA-style line art. Each one exists as **SVG** (vector — insert
this into PowerPoint so it stays sharp at any size) and **PNG** (2× raster,
if SVG gives your version of PowerPoint trouble).

Regenerate everything with:

```bash
python tools/fig_core.py     # figures 1–3
python tools/fig_ideas.py    # figures 4–12
cd web && node scripts/svg2png.mjs ../media 2    # SVG → PNG
```

## The core three

| file | shows |
|---|---|
| `fig1-fog-vs-nofog` | The same real position twice: full information, and what Red sees under fog. **76% of tiles hidden.** Red to play. |
| `fig2-branching-factor` | Chess (b ≈ 35) against ours (**b = 204, measured in that position**), each with its move fan drawn out, and a table of how alpha–beta scales with depth. |
| `fig3-tree-regions` | One tick as a game tree, annotated with where each technique acts: whether to search, what to search, how long, how deep, what to remember. |

## One per idea

| file | idea |
|---|---|
| `fig4-value-of-computation` | **Measured.** The tie fraction is bimodal — 38% of ticks are settled by the 0.6 ms scan, 57% are ones where 1-ply is blind. |
| `fig5-speculative-precomputation` | Searching next tick's problem during this one, because production is deterministic. |
| `fig6-queue-as-budget` | A 12-move queued path buys 6 seconds of uninterrupted thinking. |
| `fig7-emergent-phases` | Game phases read off the signal instead of hardcoded tick ranges. |
| `fig8-derived-weights` | One castle = 25 tiles; spending 45 army on land beats a castle by ~1.8×. |
| `fig9-fifty-tick-cycle` | A tile's value depends on how close the next land tick is. |
| `fig10-conflict-zones` | The 400 × 400 matrix decomposes into independent zones. |
| `fig11-safety-proof` | AND–OR search proving the general cannot be captured within k ticks. |
| `fig12-adaptive-beam` | Beam width read from the shape of the score distribution. |

## Where the numbers come from

Figures 1, 2, 4, 7, 10 and 12 use **real positions and real measurements** from the
simulator in `genmod/`, not mock-ups. The board in figures 1, 2 and 10 is the same
game at tick 150; the branching factor in figure 2 is counted from that position;
the distributions in figures 4, 7 and 12 come from a full 300-tick self-play game.

Figures 3, 5, 6, 8, 9 and 11 are explanatory diagrams — the arithmetic in 8 and 9
is derived from the game's production rules.

`_gapdata.json` is the cached measurement behind figures 4 and 7. Delete it and
rerun to re-measure.
