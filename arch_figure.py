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
WIRE = "#98A2B3"

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

FIG_W, FIG_H = 9.2, 4.2
TOP, BOTTOM = 0.872, 0.104

# x runs 0..100. y is scaled so that one unit is the same physical length on
# both axes — otherwise rounded corners come out as stretched ellipses.
AX_W, AX_H = FIG_W, FIG_H * (TOP - BOTTOM)
YMAX = 100.0 * AX_H / AX_W
PT = 100.0 / (AX_W * 72.0)          # one typographic point, in canvas units
RADIUS = 1.05                       # corner radius, same units


def line_h(size: float) -> float:
    return size * 1.34 * PT


def block_h(title_size: float, n_lines: int, line_size: float, pad: float) -> float:
    return line_h(title_size) + n_lines * line_h(line_size) + pad


def box(ax, x0, x1, y0, y1, color, title, *lines, title_size=12.5, line_size=9.4):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={RADIUS}",
        linewidth=1.5, edgecolor=color, facecolor=FILL.get(color, "white"), zorder=2,
    ))
    block = line_h(title_size) + len(lines) * line_h(line_size)
    y = y1 - (y1 - y0 - block) / 2 - line_h(title_size) / 2
    ax.text((x0 + x1) / 2, y, title, ha="center", va="center", fontsize=title_size,
            fontweight="semibold", color=color, zorder=3)
    prev = title_size
    for text in lines:
        y -= line_h(prev) / 2 + line_h(line_size) / 2
        prev = line_size
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=line_size,
                color=MUTED, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=WIRE):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),         arrowstyle="-|>", mutation_scale=11, linewidth=1.4,
        color=color, shrinkA=0, shrinkB=0, zorder=1, clip_on=False,
    ))


def elbow(ax, pts, color=WIRE):
    """Axis-aligned polyline, arrow head on the final segment."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.4, zorder=1,
                solid_capstyle="round")
    arrow(ax, *pts[-2], *pts[-1], color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=TOP, bottom=BOTTOM)

    chips = [
        (TEAL, "x  →  V", "the written values"),
        (PLUM, "B, C  →  K, Q", "norm, bias, RoPE"),
        (CORAL, "dt, A, trap", "step size, decay"),
    ]
    chip_h = block_h(12.0, 1, 9.4, 2.0)
    chip_gap = 1.35
    z_h = block_h(12.0, 0, 9.4, 2.0)
    z_gap = 2.5

    # Centre the whole drawing vertically.
    content = 3 * chip_h + 2 * chip_gap + z_gap + z_h
    top = (YMAX + content) / 2
    chips_bot = top - 3 * chip_h - 2 * chip_gap
    z_top = chips_bot - z_gap
    z_bot = z_top - z_h
    spine = (top + chips_bot) / 2

    # Columns
    X_IN = (9.0, 21.0)
    X_CHIP = (26.0, 44.0)
    X_SCAN = (49.0, 66.0)
    X_GATE = (70.5, 81.5)
    X_OUT = (86.0, 97.0)

    ax.text(2.4, spine + 1.1, "u", ha="right", va="center", fontsize=13.0,
            fontweight="semibold", color=INK)
    ax.text(2.4, spine - 1.9, "(B, L, D)", ha="right", va="center", fontsize=9.2,
            color=MUTED)
    arrow(ax, 3.8, spine, X_IN[0] - 0.6, spine)

    box(ax, *X_IN, z_bot, top, NAVY, "in_proj", "one fused Linear", "no bias")

    for i, (color, title, sub) in enumerate(chips):
        y1 = top - i * (chip_h + chip_gap)
        y0 = y1 - chip_h
        cy = (y0 + y1) / 2
        arrow(ax, X_IN[1], cy, X_CHIP[0] - 0.6, cy)
        box(ax, *X_CHIP, y0, y1, color, title, sub, title_size=12.0)
        arrow(ax, X_CHIP[1], cy, X_SCAN[0] - 0.6, cy)

    box(ax, *X_SCAN, chips_bot, top, NAVY, "selective scan",
        "chunked for prefill", "one step for decode")
    arrow(ax, X_SCAN[1], spine, X_GATE[0] - 0.6, spine)

    gate_h = block_h(12.5, 2, 9.4, 2.2)
    box(ax, *X_GATE, spine - gate_h / 2, spine + gate_h / 2, NAVY, "gate by z",
        "optional", "RMSNorm first")
    arrow(ax, X_GATE[1], spine, X_OUT[0] - 0.6, spine)

    out_h = block_h(12.5, 0, 9.4, 2.2)
    box(ax, *X_OUT, spine - out_h / 2, spine + out_h / 2, NAVY, "out_proj")
    arrow(ax, X_OUT[1], spine, X_OUT[1] + 3.8, spine)
    ax.text(X_OUT[1] + 5.2, spine + 1.1, "out", ha="left", va="center", fontsize=13.0,
            fontweight="semibold", color=INK)
    ax.text(X_OUT[1] + 5.2, spine - 1.9, "(B, L, D)", ha="left", va="center",
            fontsize=9.2, color=MUTED)

    # z leaves in_proj, bypasses the scan, and meets the flow again at the gate.
    z_cy = (z_top + z_bot) / 2
    gate_cx = (X_GATE[0] + X_GATE[1]) / 2
    box(ax, *X_CHIP, z_bot, z_top, BLUE, "z", title_size=12.0)
    arrow(ax, X_IN[1], z_cy, X_CHIP[0] - 0.6, z_cy, color=BLUE)
    elbow(ax, [(X_CHIP[1], z_cy), (gate_cx, z_cy), (gate_cx, spine - gate_h / 2 - 0.6)],
          color=BLUE)

    fig.text(0.0, 0.99, "One Mamba-3 layer", fontsize=15.5, fontweight="semibold",
             color=INK, va="top")
    fig.text(0.0, 0.945, "Exponential-trapezoidal discretization and a rotary state "
                         "space, in the order mamba3.py runs them.", fontsize=10.6,
             color=MUTED, va="top")
    fig.text(0.0, 0.072, "Decode carries one fixed-size state per layer: "
                         "ssm (B, H, P, N), plus k, v and the accumulated RoPE angle.",
             fontsize=9.5, color=MUTED, va="top")
    fig.text(0.0, 0.018, "B batch  ·  H heads  ·  P headdim  ·  N d_state",
             fontsize=9.5, color=MUTED, va="top")

    fig.savefig(ASSETS / "architecture.png")
    plt.close(fig)
    print("wrote architecture.png")


if __name__ == "__main__":
    main()
