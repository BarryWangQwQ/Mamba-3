"""Draw assets/headline.png — the four numbers at the top of the README.

Every figure is a reading of one of the plots further down the README, kept
here as literals so this script stays a drawing step:

    72x       assets/scan.png          speedup at L=512, R=1
    0.51 ms   assets/decode.png        CUDA-graph decode, batch 64
    247x      assets/decode_scaling.png  state vs a 16k-token KV cache
    1.1e-6    assets/alignment.png     step() against one full forward

    python stats_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

MUTED = "#5C6370"
TEAL = "#0F7B6C"
CORAL = "#C45C26"
NAVY = "#1E3A5F"
PLUM = "#7C3AED"

FILL = {
    NAVY: "#EDF1F7",
    TEAL: "#E7F2F0",
    CORAL: "#FBEEE6",
    PLUM: "#F1EBFB",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

CARDS = [
    (TEAL, "72\u00d7", "SISO scan at L=512, vs the recurrence"),
    (NAVY, "0.51 ms", "CUDA-graph decode, batch 64"),
    (PLUM, "247\u00d7", "smaller than a 16k-token KV cache"),
    (CORAL, "\u2264 1.1\u00d710\u207b\u2076", "max |\u0394| vs a full forward"),
]

# Both axes are in inches at 1:1, so corner radii stay circular.
AX_W = 6.0
GAP = 0.16
RADIUS = 0.07
NUM_SIZE, CAP_SIZE = 25.0, 9.6
PAD = 0.30


def lh(size: float) -> float:
    """Height of one line of text, in inches."""
    return size * 1.34 / 72.0


def main() -> None:
    card_w = (AX_W - GAP) / 2
    card_h = lh(NUM_SIZE) + lh(CAP_SIZE) + PAD
    ax_h = 2 * card_h + GAP

    fig, ax = plt.subplots(figsize=(AX_W, ax_h))
    ax.set_xlim(0, AX_W)
    ax.set_ylim(0, ax_h)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for i, (color, number, caption) in enumerate(CARDS):
        x0 = (i % 2) * (card_w + GAP)
        y1 = ax_h - (i // 2) * (card_h + GAP)
        ax.add_patch(FancyBboxPatch(
            (x0, y1 - card_h), card_w, card_h,
            boxstyle=f"round,pad=0,rounding_size={RADIUS}",
            linewidth=1.4, edgecolor=color, facecolor=FILL[color],
        ))
        cx = x0 + card_w / 2
        y = y1 - PAD / 2 - lh(NUM_SIZE) / 2
        ax.text(cx, y, number, ha="center", va="center", fontsize=NUM_SIZE,
                fontweight="semibold", color=color)
        y -= lh(NUM_SIZE) / 2 + lh(CAP_SIZE) / 2
        ax.text(cx, y, caption, ha="center", va="center", fontsize=CAP_SIZE,
                color=MUTED)

    fig.savefig(ASSETS / "headline.png")
    plt.close(fig)
    print(f"wrote headline.png  ({AX_W:.2f} x {ax_h:.2f} in)")


if __name__ == "__main__":
    main()
