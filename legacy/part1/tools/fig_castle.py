"""d21 - one castle equals twenty-five plain tiles. Production-rate identity."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from figstyle import *

OUT = pathlib.Path(__file__).resolve().parents[1] / "media"

def castle_glyph(f, cx, cy, s, stroke=INK, fill="none"):
    b = s * 0.34
    f.path(f"M {cx-b} {cy+b} L {cx-b} {cy-b*0.35} L {cx-b*0.55} {cy-b*0.35} "
           f"L {cx-b*0.55} {cy-b} L {cx-b*0.18} {cy-b} L {cx-b*0.18} {cy-b*0.35} "
           f"L {cx+b*0.18} {cy-b*0.35} L {cx+b*0.18} {cy-b} L {cx+b*0.55} {cy-b} "
           f"L {cx+b*0.55} {cy-b*0.35} L {cx+b} {cy-b*0.35} L {cx+b} {cy+b} Z",
           fill=fill, stroke=stroke, sw=1.3)

f = Fig(680, 340)

f.text(36, 40, "One castle = twenty-five plain tiles", size=17,
       family=SERIF, weight="700")
f.text(36, 62, "An identity that follows from the production rates.", size=12, fill=GREY)
f.line(36, 78, 644, 78, stroke="#c8c8c8", sw=0.8)

CX, CY = 104, 178
f.rect(CX - 34, CY - 34, 68, 68, fill=RED_LT, stroke=INK, sw=1.4)
castle_glyph(f, CX, CY, 68, stroke=INK)
f.text(CX, 285, "one castle", size=12.5, anchor="middle", weight="600")
f.text(CX, 303, "0.50 army / tick", size=12, anchor="middle",
       fill=RED, family=MONO)

f.text(196, CY + 9, "=", size=30, anchor="middle", fill=INK, family=SERIF)

X0, Y0, S, G = 250, 95, 30, 4
for r in range(5):
    for c in range(5):
        f.rect(X0 + c * (S + G), Y0 + r * (S + G), S, S,
               fill=RED_LT, stroke=INK, sw=1.0)
BW = 5 * S + 4 * G
f.text(X0 + BW / 2, 285, "twenty-five plain tiles", size=12.5,
       anchor="middle", weight="600")
f.text(X0 + BW / 2, 303, "25 x 0.02 = 0.50 army / tick", size=12,
       anchor="middle", fill=RED, family=MONO)

f.line(452, 95, 452, 308, stroke="#d4d4d4", sw=0.8)
f.text(480, 150, "ratio", size=12, fill=GREY)
f.text(480, 178, "0.50 / 0.02 = 25", size=15, fill=ACCENT, family=MONO, weight="700")
f.text(480, 214, "Neutral castles hold 40-50 units.", size=11.5, fill=GREY)
f.text(480, 231, "Beat the garrison once,", size=11.5, fill=GREY)
f.text(480, 248, "collect for the rest of the game.", size=11.5, fill=GREY)

f.save(OUT / "d21-castle-value.svg")
print("wrote d21-castle-value.svg")
