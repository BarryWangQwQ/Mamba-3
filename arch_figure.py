"""Draw assets/architecture.png — the data path of one Mamba3 layer.

The order follows Mamba3.forward in mamba3.py. It is a map, not a spec: the
formulas live in the README and in the code. Only matplotlib is needed.

The flow wraps onto a second row to keep the figure close to square, so the
labels stay legible when a phone scales the image down to its column width.

    python arch_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from figstyle import (BLUE, CORAL, INK, MUTED, NAVY, PLUM, TEAL, WIRE, card, lh,
                      use_style)

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

AX_W = 6.0
HEAD_H, FOOT_H = 0.56, 0.40
RADIUS = 0.08
MARGIN = 0.16

# Columns, in inches from the left edge of the axes.
X_IN = (0.50, 1.78)
X_CHIP = (2.10, 3.50)
X_SCAN = (3.82, 5.28)
X_BYPASS = 5.66                                     # where z runs down
X_GATE = (4.00, 5.10)                               # row 2, under the scan
X_OUT = (2.31, 3.29)                                # row 2, under the branches

CHIPS = [
    (TEAL, "x  \u2192  V", "the written values"),
    (PLUM, "B, C  \u2192  K, Q", "norm, bias, RoPE"),
    (CORAL, "dt, A, trap", "step size, decay"),
]
GAP_CHIP, GAP_Z, GAP_ROW = 0.09, 0.16, 0.42


def box_h(title_size: float, n_lines: int, line_size: float, pad: float) -> float:
    return lh(title_size) + n_lines * lh(line_size) + pad


def box(ax, x0, x1, y0, y1, color, title, *lines, title_size=12.5, line_size=9.4):
    card(ax, x0, y0, x1, y1, color, RADIUS)
    block = lh(title_size) + len(lines) * lh(line_size)
    y = y1 - (y1 - y0 - block) / 2 - lh(title_size) / 2
    ax.text((x0 + x1) / 2, y, title, ha="center", va="center", fontsize=title_size,
            fontweight="semibold", color=color, zorder=3)
    prev = title_size
    for text in lines:
        y -= lh(prev) / 2 + lh(line_size) / 2
        prev = line_size
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=line_size,
                color=MUTED, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=WIRE):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, linewidth=1.3,
        color=color, shrinkA=0, shrinkB=0, zorder=1, clip_on=False,
    ))


def elbow(ax, pts, color=WIRE):
    """Axis-aligned polyline, arrow head on the final segment."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.3, zorder=1,
                solid_capstyle="round")
    arrow(ax, *pts[-2], *pts[-1], color=color)


def main() -> None:
    use_style()
    z_h = box_h(12.0, 0, 9.4, 0.12)
    chip_h = box_h(12.0, 1, 9.4, 0.13)
    gate_h = box_h(12.5, 2, 9.4, 0.15)
    out_h = box_h(12.5, 0, 9.4, 0.15)

    row1_h = z_h + GAP_Z + 3 * chip_h + 2 * GAP_CHIP
    ax_h = 2 * MARGIN + row1_h + GAP_ROW + gate_h
    fig_h = ax_h + HEAD_H + FOOT_H

    fig, ax = plt.subplots(figsize=(AX_W, fig_h))
    ax.set_xlim(0, AX_W)
    ax.set_ylim(0, ax_h)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1.0 - HEAD_H / fig_h,
                        bottom=FOOT_H / fig_h)

    z_top = ax_h - MARGIN
    chips_top = z_top - z_h - GAP_Z
    chips_bot = chips_top - 3 * chip_h - 2 * GAP_CHIP
    spine = (chips_top + chips_bot) / 2                 # main flow through row 1
    row2 = chips_bot - GAP_ROW - gate_h / 2             # centre line of row 2

    ax.text(0.14, spine, "u", ha="right", va="center", fontsize=13.0,
            fontweight="semibold", color=INK)
    arrow(ax, 0.22, spine, X_IN[0] - 0.04, spine)
    box(ax, *X_IN, chips_bot, z_top, NAVY, "in_proj", "one fused Linear", "no bias")

    # z is the first slice of the projection, and the only one that skips the scan.
    z_cy = z_top - z_h / 2
    box(ax, *X_CHIP, z_top - z_h, z_top, BLUE, "z", title_size=12.0)
    arrow(ax, X_IN[1], z_cy, X_CHIP[0] - 0.04, z_cy, color=BLUE)

    for i, (color, title, sub) in enumerate(CHIPS):
        y1 = chips_top - i * (chip_h + GAP_CHIP)
        cy = y1 - chip_h / 2
        arrow(ax, X_IN[1], cy, X_CHIP[0] - 0.04, cy)
        box(ax, *X_CHIP, y1 - chip_h, y1, color, title, sub, title_size=12.0)
        arrow(ax, X_CHIP[1], cy, X_SCAN[0] - 0.04, cy)

    box(ax, *X_SCAN, chips_bot, chips_top, NAVY, "selective scan",
        "chunked for prefill", "one step for decode")

    # Row 1 turns down into row 2, which reads right to left.
    scan_cx = (X_SCAN[0] + X_SCAN[1]) / 2
    arrow(ax, scan_cx, chips_bot, scan_cx, row2 + gate_h / 2 + 0.04)
    box(ax, *X_GATE, row2 - gate_h / 2, row2 + gate_h / 2, NAVY, "gate by z",
        "optional", "RMSNorm first")
    elbow(ax, [(X_CHIP[1], z_cy), (X_BYPASS, z_cy), (X_BYPASS, row2),
               (X_GATE[1] + 0.04, row2)], color=BLUE)

    arrow(ax, X_GATE[0], row2, X_OUT[1] + 0.04, row2)
    box(ax, *X_OUT, row2 - out_h / 2, row2 + out_h / 2, NAVY, "out_proj")
    arrow(ax, X_OUT[0], row2, X_OUT[0] - 0.32, row2)
    ax.text(X_OUT[0] - 0.40, row2, "out", ha="right", va="center", fontsize=13.0,
            fontweight="semibold", color=INK)

    fig.text(0.0, 1.0 - 0.05 / fig_h, "One Mamba-3 layer", fontsize=15.0,
             fontweight="semibold", color=INK, va="top")
    fig.text(0.0, 1.0 - 0.32 / fig_h, "The data path, in the order mamba3.py runs it.",
             fontsize=10.2, color=MUTED, va="top")
    fig.text(0.0, (FOOT_H - 0.06) / fig_h, "Decode keeps one fixed-size state per "
             "layer: ssm (B, H, P, N), k, v, RoPE angle.", fontsize=9.3, color=MUTED,
             va="top")
    fig.text(0.0, (FOOT_H - 0.24) / fig_h, "u, out are (B, L, D)  \u00b7  B batch  "
             "\u00b7  H heads  \u00b7  P headdim  \u00b7  N d_state",
             fontsize=9.3, color=MUTED, va="top")

    fig.savefig(ASSETS / "architecture.png")
    plt.close(fig)
    print(f"wrote architecture.png  ({AX_W:.2f} x {fig_h:.2f} in)")


if __name__ == "__main__":
    main()
