"""Tile several rendered figures into one image so a batch can be checked at once."""
import pathlib, sys
from PIL import Image, ImageDraw

def sheet(paths, out, cols=2, cell=(900, 620), pad=14, label=True):
    n = len(paths); rows = (n + cols - 1) // cols
    W = cols*cell[0] + pad*(cols+1); H = rows*cell[1] + pad*(rows+1)
    canvas = Image.new("RGB", (W, H), "#9aa0a6")
    d = ImageDraw.Draw(canvas)
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        x = pad + c*(cell[0]+pad); y = pad + r*(cell[1]+pad)
        im = Image.open(p).convert("RGB")
        im.thumbnail(cell, Image.LANCZOS)
        box = Image.new("RGB", cell, "white")
        box.paste(im, ((cell[0]-im.width)//2, (cell[1]-im.height)//2))
        canvas.paste(box, (x, y))
        d.rectangle([x, y, x+cell[0]-1, y+cell[1]-1], outline="#3c4043", width=2)
        if label:
            d.rectangle([x, y, x+300, y+22], fill="#3c4043")
            d.text((x+6, y+5), pathlib.Path(p).stem, fill="white")
    canvas.save(out)
    print(f"{out}  {W}x{H}  ({n} figures)")

if __name__ == "__main__":
    out = sys.argv[1]; cols = int(sys.argv[2]); paths = sys.argv[3:]
    sheet(paths, out, cols=cols)
