"""Compose 2x2 manifold-evidence composite for V62a SHARED1000.

Pulls four PNG panels from
``reports/manifold_analysis/20260515_V62a_shared1000_full_semantic/figures/``
and arranges them on a uniform white canvas with letter labels (a)-(d).
The output is consumed by chapter 5 of the thesis as an auxiliary V62a
single-expert evidence panel under the section-level scope caveat.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = (REPO_ROOT
           / "reports" / "manifold_analysis"
           / "20260515_V62a_shared1000_full_semantic" / "figures")
OUT_PATH = Path(__file__).with_name("manifold_evidence_composite.png")

PANELS = [
    ("a", "Hubness Lorenz curve", SRC_DIR / "figure_hubness_lorenz.png"),
    ("b", "RSA pairwise similarity scatter", SRC_DIR / "figure_rsa_similarity_scatter.png"),
    ("c", "Retrieval Cosine vs CSLS", SRC_DIR / "figure_retrieval_comparison.png"),
    ("d", "Semantic axis preservation", SRC_DIR / "figure_semantic_axes_preservation_bar.png"),
]

PANEL_W = 1300
PANEL_H = 720
MARGIN = 60
LABEL_HEIGHT = 50
SPACING = 50
BG = (255, 255, 255, 255)


def fit_to_panel(im, w, h):
    """Resize PIL image to fit (w, h) preserving aspect ratio, centered on white."""
    im = im.convert("RGBA")
    src_w, src_h = im.size
    scale = min(w / src_w, h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    im = im.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), BG)
    canvas.paste(im, ((w - new_w) // 2, (h - new_h) // 2), im)
    return canvas


def main() -> None:
    rows = 2
    cols = 2
    total_w = MARGIN * 2 + cols * PANEL_W + (cols - 1) * SPACING
    total_h = MARGIN * 2 + rows * (PANEL_H + LABEL_HEIGHT) + (rows - 1) * SPACING
    canvas = Image.new("RGBA", (total_w, total_h), BG)

    font_label = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=34)
    font_caption = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=26)

    for i, (letter, caption, path) in enumerate(PANELS):
        row = i // cols
        col = i % cols
        x0 = MARGIN + col * (PANEL_W + SPACING)
        y0 = MARGIN + row * (PANEL_H + LABEL_HEIGHT + SPACING)

        # Letter label
        draw = ImageDraw.Draw(canvas)
        draw.text((x0, y0), f"({letter})", fill=(0, 0, 0, 255), font=font_label)
        # Caption next to letter label
        draw.text((x0 + 90, y0 + 4), caption,
                  fill=(40, 40, 40, 255), font=font_caption)

        # Panel image
        if not path.exists():
            raise FileNotFoundError(f"missing panel image: {path}")
        panel = Image.open(path)
        fitted = fit_to_panel(panel, PANEL_W, PANEL_H)
        canvas.paste(fitted, (x0, y0 + LABEL_HEIGHT), fitted)

    canvas.convert("RGB").save(OUT_PATH, "PNG", optimize=True)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
