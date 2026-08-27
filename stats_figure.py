"""Draw assets/headline.png — the four numbers at the top of the README.

Every figure is a reading of one of the plots further down the README, kept
here as literals so this script stays a drawing step:

    72x       assets/scan.png            speedup at L=512, R=1
    0.51 ms   assets/decode.png          CUDA-graph decode, batch 64
    247x      assets/decode_scaling.png  state vs a 16k-token KV cache
    1.1e-6    assets/alignment.png       step() against one full forward

    python stats_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from figstyle import CORAL, MUTED, NAVY, PLUM, TEAL, card, lh, use_style

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

CARDS = [
    (TEAL, "72\u00d7", "SISO scan at L=512, vs the recurrence"),
    (NAVY, "0.51 ms", "CUDA-graph decode, batch 64"),
    (PLUM, "247\u00d7", "smaller than a 16k-token KV cache"),
    (CORAL, "\u2264 1.1\u00d710\u207b\u2076", "max |\u0394| vs a full forward"),
]

AX_W = 6.0
GAP = 0.16
RADIUS = 0.10
NUM_SIZE, CAP_SIZE = 25.0, 9.6
PAD = 0.30
BLEED = 0.04    # the outer cards would otherwise be clipped to half a stroke


def main() -> None:
    use_style()
    card_w = (AX_W - GAP) / 2
    card_h = lh(NUM_SIZE) + lh(CAP_SIZE) + PAD
    ax_h = 2 * card_h + GAP

    fig, ax = plt.subplots(figsize=(AX_W + 2 * BLEED, ax_h + 2 * BLEED))
    ax.set_xlim(-BLEED, AX_W + BLEED)
    ax.set_ylim(-BLEED, ax_h + BLEED)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for i, (color, number, caption) in enumerate(CARDS):
        x0 = (i % 2) * (card_w + GAP)
        y1 = ax_h - (i // 2) * (card_h + GAP)
        card(ax, x0, y1 - card_h, x0 + card_w, y1, color, RADIUS)
        cx = x0 + card_w / 2
        y = y1 - PAD / 2 - lh(NUM_SIZE) / 2
        ax.text(cx, y, number, ha="center", va="center", fontsize=NUM_SIZE,
                fontweight="semibold", color=color, zorder=3)
        y -= lh(NUM_SIZE) / 2 + lh(CAP_SIZE) / 2
        ax.text(cx, y, caption, ha="center", va="center", fontsize=CAP_SIZE,
                color=MUTED, zorder=3)

    fig.savefig(ASSETS / "headline.png")
    plt.close(fig)
    print(f"wrote headline.png  ({AX_W:.2f} x {ax_h:.2f} in)")


if __name__ == "__main__":
    main()
