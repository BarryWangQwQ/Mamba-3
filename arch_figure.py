"""Draw assets/architecture.png — the data path of one Mamba3 layer, left to right.

The order follows Mamba3.forward in mamba3.py. It is a map, not a spec: the
formulas live in the README and in the code. Only matplotlib is needed.

    python arch_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

INK = "#1A1D23"
MUTED = "#5C6370"
TEAL = "#0F7B6C"
CORAL = "#C45C26"
NAVY = "#1E3A5F"
BLUE = "#1D4ED8"
PLUM = "#7C3AED"
SLATE = "#94A3B8"

FILL = {
    NAVY: "#EDF1F7",
    TEAL: "#E7F2F0",
    CORAL: "#FBEEE6",
    PLUM: "#F1EBFB",
    BLUE: "#E9EEFC",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.24,
})

FIG_W, FIG_H = 9.2, 4.7
TOP, BOTTOM = 0.865, 0.105
# The canvas is 0..100 in both directions, so a font size has to be converted
# into vertical units before text can be stacked.
U_PER_PT = 100.0 / (FIG_H * (TOP - BOTTOM) * 72.0)

SPINE = 60.0          # vertical centre of the main flow
LANE = 12.0           # the z bypass runs along here


def line_h(size: float) -> float:
    return size * 1.34 * U_PER_PT


def box(ax, x0, x1, height, color, title, *lines, center=SPINE,
        title_size=13.0, line_size=9.8):
    y0, y1 = center - height / 2, center + height / 2
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, height,
        boxstyle="round,pad=0,rounding_size=2.0",
        linewidth=1.6, edgecolor=color, facecolor=FILL.get(color, "white"), zorder=2,
    ))
    block = line_h(title_size) + len(lines) * line_h(line_size)
    y = y1 - (height - block) / 2 - line_h(title_size) / 2
    ax.text((x0 + x1) / 2, y, title, ha="center", va="center", fontsize=title_size,
            fontweight="semibold", color=color, zorder=3)
    prev = title_size
    for text in lines:
        y -= line_h(prev) / 2 + line_h(line_size) / 2
        prev = line_size
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=line_size,
                color=MUTED, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=SLATE):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, linewidth=1.5,
        color=color, shrinkA=0, shrinkB=0, zorder=1,
    ))


def elbow(ax, pts, color=SLATE):
    """Polyline with an arrow head at the end."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.5, zorder=1,
                solid_capstyle="round")
    arrow(ax, *pts[-2], *pts[-1], color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=TOP, bottom=BOTTOM)

    ax.text(2.5, SPINE, "u", ha="right", va="center", fontsize=13.0,
            fontweight="semibold", color=INK)
    ax.text(2.5, SPINE - 9, "(B, L, D)", ha="right", va="center", fontsize=9.4,
            color=MUTED)
    arrow(ax, 4, SPINE, 8.6, SPINE)

    box(ax, 9, 21, 30, NAVY, "in_proj", "one fused", "Linear, no bias")

    chips = [
        (TEAL, "x  →  V", "the written values", 88.5),
        (PLUM, "B, C  →  K, Q", "norm, bias, RoPE", 60.0),
        (CORAL, "dt, A, trap", "step size, decay", 31.5),
    ]
    for color, title, sub, cy in chips:
        arrow(ax, 21, SPINE, 26.6, cy)
        box(ax, 27, 43, 23, color, title, sub, center=cy, title_size=12.0,
            line_size=9.4)
        arrow(ax, 43, cy, 48.6, SPINE)

    box(ax, 49, 66, 40, NAVY, "selective scan",
        "chunked for prefill", "one step for decode")
    arrow(ax, 66, SPINE, 70.6, SPINE)

    box(ax, 71, 83, 30, NAVY, "gate by z", "optional", "RMSNorm first")
    arrow(ax, 83, SPINE, 87.6, SPINE)

    box(ax, 88, 99, 30, NAVY, "out_proj")
    arrow(ax, 99, SPINE, 103.6, SPINE)
    ax.text(105, SPINE, "out", ha="left", va="center", fontsize=13.0,
            fontweight="semibold", color=INK)
    ax.text(105, SPINE - 9, "(B, L, D)", ha="left", va="center", fontsize=9.4,
            color=MUTED)

    # z leaves in_proj, bypasses the scan and meets the flow again at the gate.
    box(ax, 27, 43, 17, BLUE, "z", center=LANE, title_size=12.0)
    elbow(ax, [(15, 45), (15, LANE), (26.6, LANE)], color=BLUE)
    elbow(ax, [(43, LANE), (77, LANE), (77, 45)], color=BLUE)

    fig.text(0.0, 0.99, "One Mamba-3 layer", fontsize=15.5, fontweight="semibold",
             color=INK, va="top")
    fig.text(0.0, 0.945, "Exponential-trapezoidal discretization and a rotary state "
                         "space, in the order mamba3.py runs them.", fontsize=10.8,
             color=MUTED, va="top")
    fig.text(0.0, 0.055, "Decode carries one fixed-size state per layer: "
                         "ssm (B, H, P, N), plus k, v and the accumulated RoPE angle.",
             fontsize=9.6, color=MUTED, va="top")
    fig.text(0.0, 0.008, "B batch  ·  H heads  ·  P headdim  ·  N d_state",
             fontsize=9.6, color=MUTED, va="top")

    fig.savefig(ASSETS / "architecture.png")
    plt.close(fig)
    print("wrote architecture.png")


if __name__ == "__main__":
    main()
