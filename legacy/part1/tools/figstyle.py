"""Shared drawing helpers for the figures. AIMA-style line art: thin black
strokes, light grey fills, sparing colour, sans-serif labels, serif caption."""

from __future__ import annotations

INK = "#1a1a1a"
GREY = "#8a8a8a"
LINE = "#4a4a4a"
FILL_LIGHT = "#f0f0ee"
FILL_MID = "#dcdcd8"
FILL_DARK = "#b8b8b4"
RED = "#c0392b"
RED_LT = "#e8b4ae"
BLUE = "#2c5f8a"
BLUE_LT = "#aac4d8"
HIDDEN = "#6e6e6e"
ACCENT = "#8a6d10"

SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
SERIF = "Georgia, Times New Roman, serif"
MONO = "Menlo, Consolas, monospace"


class Fig:
    """Minimal SVG builder."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.parts: list[str] = []

    def add(self, s: str):
        self.parts.append(s)
        return self

    def rect(self, x, y, w, h, fill="none", stroke=LINE, sw=1.0, rx=0, extra=""):
        return self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'
        )

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1.0, dash=None, cap="round"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}"{d}/>'
        )

    def path(self, d, fill="none", stroke=LINE, sw=1.0, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        return self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"{da}/>'
        )

    def circle(self, cx, cy, r, fill="none", stroke=LINE, sw=1.0):
        return self.add(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def text(self, x, y, s, size=12, fill=INK, family=SANS, anchor="start",
             weight="400", style="normal", ls=0):
        s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        return self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
            f'font-style="{style}" letter-spacing="{ls}">{s}</text>'
        )

    def label(self, x, y, s, size=11.5, fill=INK, anchor="start", weight="500"):
        return self.text(x, y, s, size=size, fill=fill, anchor=anchor, weight=weight)

    def caption(self, num, s, x=None, y=None):
        """Figure caption, book style: bold number then roman text."""
        x = 24 if x is None else x
        y = self.h - 18 if y is None else y
        self.text(x, y, f"Figure {num}", size=13, family=SERIF, weight="700")
        self.text(x + 62, y, s, size=13, family=SERIF)
        return self

    def arrow(self, x1, y1, x2, y2, stroke=INK, sw=1.4, head=7, dash=None):
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
        self.line(x1, y1, bx, by, stroke=stroke, sw=sw, dash=dash)
        p1 = (x2, y2)
        p2 = (x2 - head * math.cos(ang - 0.42), y2 - head * math.sin(ang - 0.42))
        p3 = (x2 - head * math.cos(ang + 0.42), y2 - head * math.sin(ang + 0.42))
        return self.add(
            f'<polygon points="{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} '
            f'{p3[0]:.1f},{p3[1]:.1f}" fill="{stroke}"/>'
        )

    def brace(self, x, y1, y2, label="", side="left", stroke=LINE):
        """A curly-ish brace, drawn as a simple bracket."""
        t = 8 if side == "left" else -8
        self.path(f"M {x+t} {y1} L {x} {y1} L {x} {y2} L {x+t} {y2}", stroke=stroke, sw=1.2)
        if label:
            self.text(x - 6 if side == "left" else x + 6, (y1 + y2) / 2 + 4, label,
                      size=11, anchor="end" if side == "left" else "start", fill=GREY)
        return self

    def svg(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}">'
            f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>'
            + "".join(self.parts) + "</svg>"
        )

    def save(self, path):
        import pathlib
        pathlib.Path(path).write_text(self.svg())
        return path


def crown(f: Fig, cx, cy, s, fill):
    """Small crown glyph marking a general."""
    d = (f"M {cx-s} {cy+s*0.55} L {cx-s} {cy-s*0.35} L {cx-s*0.42} {cy+s*0.1} "
         f"L {cx} {cy-s*0.78} L {cx+s*0.42} {cy+s*0.1} L {cx+s} {cy-s*0.35} "
         f"L {cx+s} {cy+s*0.55} Z")
    f.path(d, fill=fill, stroke="none")


def mountain(f: Fig, cx, cy, s, stroke):
    f.path(f"M {cx-s} {cy+s*0.6} L {cx} {cy-s*0.7} L {cx+s} {cy+s*0.6} Z",
           fill="none", stroke=stroke, sw=1.1)
