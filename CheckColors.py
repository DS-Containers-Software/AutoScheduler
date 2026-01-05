"""
Display the PDF color categories as visual swatches.

Requires:
  pip install matplotlib
"""

from __future__ import annotations

import math
import matplotlib.pyplot as plt


# Keep these in sync with your main script
COLOR_RANK = {
    "WHITE": 0, "IVORY": 1, "CREAM": 1, "YELLOW": 2, "GOLD": 3, "TAN": 4, "BEIGE": 4,
    "PINK": 5, "ORANGE": 6, "RED": 7, "MAROON": 8, "BURGUNDY": 8, "PURPLE": 9, "VIOLET": 9,
    "LAVENDER": 9, "BROWN": 10, "GREEN": 11,"Lime": 11, "TEAL": 12, "TURQUOISE": 12, "CYAN": 13,
    "SKY": 14, "BLUE": 15, "NAVY": 16, "SILVER": 17, "GREY": 18,
    "CHARCOAL": 19, "BLACK": 20,
}

BASE_RGB = {
    "WHITE": (255, 255, 255),
    "IVORY": (255, 255, 240),
    "CREAM": (255, 253, 208),
    "YELLOW": (255, 255, 0),
    "GOLD": (255, 215, 64),
    "TAN": (210, 180, 140),
    "BEIGE": (245, 245, 220),
    "PINK": (255, 192, 203),
    "ORANGE": (255, 165, 0),
    "RED": (255, 64, 64),
    "MAROON": (128, 32, 32),
    "BURGUNDY": (128, 32, 64),
    "PURPLE": (128, 32, 128),
    "VIOLET": (138, 43, 226),
    "LAVENDER": (230, 230, 250),
    "BROWN": (139, 69, 19),
    "GREEN": (32, 128, 32),
    "Lime": (64, 255, 64),
    "TEAL": (32, 128, 128),
    "TURQUOISE": (64, 224, 208),
    "CYAN": (0, 255, 255),
    "SKY": (135, 206, 235),
    "BLUE": (0, 64, 255),
    "NAVY": (0, 64, 128),
    "SILVER": (192, 192, 192),
    "GREY": (128, 128, 128),
    "CHARCOAL": (64, 64, 64),
    "BLACK": (0, 0, 0),
}


def _text_color_for_bg(rgb: tuple[int, int, int]) -> str:
    """Pick black/white text based on perceived luminance for readability."""
    r, g, b = rgb
    # sRGB-ish luminance
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum > 145 else "white"


def main() -> None:
    # Use the same ordering concept as your classifier
    names = list(COLOR_RANK.keys())

    n = len(names)
    cols = 4  # change if you want wider/narrower
    rows = math.ceil(n / cols)

    fig_w = max(10, cols * 3.2)
    fig_h = max(6, rows * 1.6)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        rgb = BASE_RGB[name]
        color = tuple(c / 255 for c in rgb)

        ax.set_facecolor(color)
        ax.set_xticks([])
        ax.set_yticks([])

        # Label with name, rank, and RGB
        label = f"{name}\nrank={COLOR_RANK[name]}\nRGB={rgb}"
        ax.text(
            0.5, 0.5, label,
            ha="center", va="center",
            fontsize=12,
            color=_text_color_for_bg(rgb),
            family="DejaVu Sans",
            weight="bold",
        )

        # Frame
        for spine in ax.spines.values():
            spine.set_visible(True)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("PDF Color Categories (BASE_RGB)", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
