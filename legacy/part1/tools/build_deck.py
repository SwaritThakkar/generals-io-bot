"""Build the .pptx from writeup.txt content. Bullets only; figures from media/.

Typography: Georgia for headings (scholarly, not a UI font), Arial for bullets
(available everywhere), Consolas for code and data. Equations are rendered by
matplotlib's mathtext and embedded as transparent PNGs.
"""
from __future__ import annotations
import pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
EQ = ROOT / "media" / "eq"; EQ.mkdir(parents=True, exist_ok=True)

INK   = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x6B, 0x6B, 0x6B)
FAINT = RGBColor(0xA8, 0xA8, 0xA8)
ACCENT= RGBColor(0x8F, 0x64, 0x10)
RED   = RGBColor(0xB4, 0x34, 0x2C)
BLUE  = RGBColor(0x2A, 0x5F, 0x95)
GREEN = RGBColor(0x2F, 0x6B, 0x45)
RULE  = RGBColor(0xC8, 0xC8, 0xC8)

SERIF, SANS, MONO = "Georgia", "Arial", "Consolas"
W, H = Inches(13.333), Inches(7.5)

def eq(tex, name, size=22, color="#1A1A1A"):
    """Render a LaTeX expression to a transparent PNG."""
    out = EQ / f"{name}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, f"${tex}$", fontsize=size, color=color)
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out

class Deck:
    def __init__(self):
        self.p = Presentation()
        self.p.slide_width, self.p.slide_height = W, H
        self.blank = self.p.slide_layouts[6]
        self.n = 0

    # ---------- primitives ----------
    def _tb(self, s, l, t, w, h, anchor=MSO_ANCHOR.TOP):
        tb = s.shapes.add_textbox(l, t, w, h); tf = tb.text_frame
        tf.word_wrap = True; tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        return tf

    def _line(self, s, l, t, w, color=RULE, pt=0.75):
        from pptx.enum.shapes import MSO_SHAPE
        sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, Pt(pt))
        sh.fill.solid(); sh.fill.fore_color.rgb = color
        sh.line.fill.background(); sh.shadow.inherit = False
        return sh

    def slide(self, eyebrow=None, title=None, rule=True):
        s = self.p.slides.add_slide(self.blank); self.n += 1
        y = Inches(0.52)
        if eyebrow:
            tf = self._tb(s, Inches(0.62), y, Inches(11.5), Inches(0.26))
            r = tf.paragraphs[0].add_run(); r.text = eyebrow.upper()
            f = r.font; f.name = SANS; f.size = Pt(11); f.bold = True; f.color.rgb = ACCENT
            y = y + Inches(0.30)
        if title:
            tf = self._tb(s, Inches(0.62), y, Inches(12.1), Inches(0.72))
            r = tf.paragraphs[0].add_run(); r.text = title
            f = r.font; f.name = SERIF; f.size = Pt(30); f.bold = True; f.color.rgb = INK
            y = y + Inches(0.74)
        if rule:
            self._line(s, Inches(0.62), y, Inches(12.1))
        # footer
        tf = self._tb(s, Inches(0.62), Inches(7.02), Inches(11.0), Inches(0.24))
        r = tf.paragraphs[0].add_run(); r.text = "Search-only AI for Generals.io   ·   DAC-203"
        fo = r.font; fo.name = SANS; fo.size = Pt(9); fo.color.rgb = FAINT
        tf = self._tb(s, Inches(12.0), Inches(7.02), Inches(0.72), Inches(0.24))
        tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
        r = tf.paragraphs[0].add_run(); r.text = str(self.n)
        fo = r.font; fo.name = MONO; fo.size = Pt(9); fo.color.rgb = FAINT
        s._body_top = y + Inches(0.22)
        return s

    def bullets(self, s, items, left=Inches(0.62), top=None, width=Inches(6.0),
                size=15, gap=9):
        """items: str, or (str, level), or (str, level, colour)."""
        tf = self._tb(s, left, top or s._body_top, width, Inches(5.4))
        first = True
        for it in items:
            txt, lvl, col = (it, 0, INK) if isinstance(it, str) else (
                it + (INK,) if len(it) == 2 else it)
            para = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            para.space_after = Pt(gap)
            para.space_before = Pt(0 if lvl else 3)
            if txt == "":
                para.add_run().text = " "; continue
            bullet = "-  " if lvl else "·  "
            r0 = para.add_run(); r0.text = bullet
            f0 = r0.font; f0.name = SANS; f0.size = Pt(size - lvl*1.5)
            f0.color.rgb = FAINT if lvl else ACCENT
            r = para.add_run(); r.text = txt
            f = r.font; f.name = SANS; f.size = Pt(size - lvl*1.5)
            f.color.rgb = MUTED if lvl else col
            para.level = lvl
        return tf

    def figure(self, s, name, left, top, max_w, max_h, caption=None):
        path = MEDIA / f"{name}.png"
        iw, ih = Image.open(path).size
        scale = min(max_w / iw, max_h / ih)
        w, h = int(iw*scale), int(ih*scale)
        l = left + int((max_w - w)/2)
        s.shapes.add_picture(str(path), l, top, width=w, height=h)
        if caption:
            tf = self._tb(s, left, top + h + Inches(0.08), max_w, Inches(0.3))
            tf.paragraphs[0].alignment = PP_ALIGN.CENTER
            r = tf.paragraphs[0].add_run(); r.text = caption
            f = r.font; f.name = SANS; f.size = Pt(10); f.color.rgb = FAINT
        return h

    def equation(self, s, tex, name, left, top, height=Inches(0.42), size=22, color="#1A1A1A"):
        path = eq(tex, name, size=size, color=color)
        iw, ih = Image.open(path).size
        w = int(height * iw / ih)
        s.shapes.add_picture(str(path), left, top, width=w, height=height)
        return w

    def note(self, s, text, left, top, width, color=ACCENT, size=12.5, bold=True):
        tf = self._tb(s, left, top - Inches(0.08), width, Inches(0.9))
        r = tf.paragraphs[0].add_run(); r.text = text
        f = r.font; f.name = SANS; f.size = Pt(size); f.bold = bold; f.color.rgb = color
        return tf

    def save(self, path):
        self.p.save(path); return path
