"""Draw assets/headline.png — the six numbers at the top of the README.

Each one is a reading of a plot further down the README, kept here as a literal
so this script stays a drawing step. The first row is speed, the second is what
the speed did not cost:

    70x       assets/scan.png            chunked vs recurrence at L=512, R=1
    200x      assets/long_context.png    forward + backward at L=4096
    0.075 ms  assets/compile.png         compiled graph decode, batch 1
    247x      assets/decode_scaling.png  state vs a 16k-token KV cache
    1.1e-6    assets/alignment.png       step() against one full forward
    1 file    mamba3.py                  no custom kernels, no build step

    python scripts/stats_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from figstyle import (BLUE, CORAL, MUTED, NAVY, PLUM, SLATE, TEAL, card, lh,
                      use_style)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

# Each caption reads on from its number, so all of them stay lowercase, and all
# of them stay short: six cards across a phone screen leaves little room. The
# exponent is written 1e-6 because the figure font has no superscript minus.
CARDS = [
    (TEAL, "70\u00d7", "faster prefill scan"),
    (CORAL, "200\u00d7", "faster training at 4k"),
    (BLUE, "0.075 ms", "per decoded token"),
    (PLUM, "247\u00d7", "smaller than a KV cache"),
    (NAVY, "\u2264 1.1e-6", "vs a full forward"),
    (SLATE, "1 file", "no custom kernels"),
]

COLS = 3
GAP = 0.16
RADIUS = 0.10
NUM_SIZE, CAP_SIZE = 24.0, 10.5
PAD = 0.30
SIDE = 0.26     # gap between the longest line and the card edge
BLEED = 0.04    # the outer cards would otherwise be clipped to half a stroke


def _line_width(fig, text: str, size: float, weight: str = "normal") -> float:
    """Rendered width of one line, in inches."""
    handle = fig.text(0, 0, text, fontsize=size, fontweight=weight)
    width = handle.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    handle.remove()
    return width


def _card_width() -> float:
    """Width that fits every card's longest line, measured rather than guessed.

    All six cards share it, so the widest string sets the figure width. Wording
    is what changes here most often, and a card sized by hand goes on looking
    right until the day a caption grows past it and the text runs off the edge.
    """
    probe = plt.figure()
    probe.canvas.draw()
    widest = max(max(_line_width(probe, number, NUM_SIZE, "bold"),
                     _line_width(probe, caption, CAP_SIZE))
                 for _, number, caption in CARDS)
    plt.close(probe)
    return widest + 2 * SIDE


def main() -> None:
    use_style()
    card_w = _card_width()
    card_h = lh(NUM_SIZE) + lh(CAP_SIZE) + PAD
    rows = -(-len(CARDS) // COLS)
    ax_w = COLS * card_w + (COLS - 1) * GAP
    ax_h = rows * card_h + (rows - 1) * GAP

    fig, ax = plt.subplots(figsize=(ax_w + 2 * BLEED, ax_h + 2 * BLEED))
    ax.set_xlim(-BLEED, ax_w + BLEED)
    ax.set_ylim(-BLEED, ax_h + BLEED)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for i, (color, number, caption) in enumerate(CARDS):
        x0 = (i % COLS) * (card_w + GAP)
        y1 = ax_h - (i // COLS) * (card_h + GAP)
        card(ax, x0, y1 - card_h, x0 + card_w, y1, color, RADIUS)
        cx = x0 + card_w / 2
        y = y1 - PAD / 2 - lh(NUM_SIZE) / 2
        ax.text(cx, y, number, ha="center", va="center", fontsize=NUM_SIZE,
                fontweight="bold", color=color, zorder=3)
        y -= lh(NUM_SIZE) / 2 + lh(CAP_SIZE) / 2
        ax.text(cx, y, caption, ha="center", va="center", fontsize=CAP_SIZE,
                color=MUTED, zorder=3)

    fig.savefig(ASSETS / "headline.png")
    plt.close(fig)
    print(f"wrote headline.png  ({ax_w:.2f} x {ax_h:.2f} in)")


if __name__ == "__main__":
    main()
