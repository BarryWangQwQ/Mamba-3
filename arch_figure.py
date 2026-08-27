"""Draw assets/architecture.png — the data path of one Mamba3 layer.

The diagram follows Mamba3.forward and _recurrent_scan in mamba3.py, in order.
Only matplotlib is needed; run it after changing the layer so the two stay in sync.

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
SLATE = "#64748B"

FILL = {
    NAVY: "#EDF1F7",
    TEAL: "#E7F2F0",
    CORAL: "#FBEEE6",
    PLUM: "#F1EBFB",
    BLUE: "#E9EEFC",
}

MONO = ["Consolas", "DejaVu Sans Mono"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.22,
})

# Canvas: y runs 0..100 over the axes. Filled in by main() so that text can be
# laid out in font points rather than guessed fractions.
FIG_H, TOP, BOTTOM = 9.0, 0.912, 0.07
U_PER_PT = 100.0 / (FIG_H * (TOP - BOTTOM) * 72.0)


def line_h(size: float) -> float:
    """Baseline-to-baseline distance for a given font size, in canvas units."""
    return size * 1.32 * U_PER_PT


def text_height(title_size: float, n_lines: int, line_size: float) -> float:
    return line_h(title_size) + n_lines * line_h(line_size)


def box(ax, x0, x1, y_top, height, color, title, *lines,
        title_size=12.5, line_size=10.0, mono=False, note=None, note_size=9.4):
    """A rounded box whose text block is centred vertically."""
    y0 = y_top - height
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, height,
        boxstyle="round,pad=0,rounding_size=1.5",
        linewidth=1.5, edgecolor=color, facecolor=FILL.get(color, "white"), zorder=2,
    ))
    cx = (x0 + x1) / 2
    block = text_height(title_size, len(lines), line_size)
    if note:
        block += line_h(note_size)
    y = y_top - (height - block) / 2 - line_h(title_size) / 2
    ax.text(cx, y, title, ha="center", va="center", fontsize=title_size,
            fontweight="semibold", color=color, zorder=3)
    prev = title_size
    for i, text in enumerate(lines):
        y -= line_h(prev) / 2 + line_h(line_size) / 2
        prev = line_size
        ax.text(cx, y, text, ha="center", va="center", fontsize=line_size,
                color=INK if mono else MUTED, family=MONO if mono else None, zorder=3)
    if note:
        y -= line_h(prev) / 2 + line_h(note_size) / 2
        ax.text(cx, y, note, ha="center", va="center", fontsize=note_size,
                color=MUTED, zorder=3)
    return y0


def pill(ax, cx, y_top, height, w, text, color=INK):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y_top - height), w, height,
        boxstyle="round,pad=0,rounding_size=2.7",
        linewidth=1.5, edgecolor=color, facecolor="white", zorder=2,
    ))
    ax.text(cx, y_top - height / 2, text, ha="center", va="center", fontsize=12.0,
            fontweight="semibold", color=color, zorder=3)
    return y_top - height


def arrow(ax, x0, y0, x1, y1, color=SLATE):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, linewidth=1.3,
        color=color, shrinkA=0, shrinkB=0, zorder=1,
    ))


def elbow(ax, pts, color=SLATE):
    """Polyline, arrow head on the final segment."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.3, zorder=1,
                solid_capstyle="round")
    arrow(ax, *pts[-2], *pts[-1], color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.8, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=TOP, bottom=BOTTOM)

    L, R = 4, 82          # main column
    LANE = 92             # right-hand lane for the gate path
    CX = (L + R) / 2
    GAP = 5.2             # vertical space taken by a connecting arrow

    def down(y, gap=GAP, x=CX, color=SLATE):
        arrow(ax, x, y, x, y - gap, color=color)
        return y - gap

    y = pill(ax, CX, 100, 6.0, 34, "u    (B, L, D)")
    y = down(y, 4.4)

    y_inproj_top = y
    y = box(ax, L, R, y, text_height(12.5, 1, 9.6) + 4.0, NAVY, "in_proj",
            "one fused Linear, no bias  →  z | x | B | C | dt | A | trap | angles",
            line_size=9.6)
    y_inproj_bot = y

    chips = [
        (TEAL, "x  →  V",
         ["the values written", "into the state",
          "MIMO: × mimo_x over R"]),
        (PLUM, "B, C  →  K, Q",
         ["RMSNorm, then + bias", "RoPE with angles scaled",
          "by dt, summed over t"]),
        (CORAL, "dt, A, trap",
         ["dt = softplus(· + bias)", "A = −heavy_tail(·)",
          "α = dt(1−tr/2), β = dt·tr/2"]),
    ]
    chip_h = text_height(11.5, 3, 9.0) + 3.4
    w = (R - L - 2 * 2.4) / 3
    y_chips_top = y - GAP
    for i, (color, title, lines) in enumerate(chips):
        x0 = L + i * (w + 2.4)
        cx = x0 + w / 2
        arrow(ax, cx, y_inproj_bot, cx, y_chips_top)
        box(ax, x0, x0 + w, y_chips_top, chip_h, color, title, *lines,
            title_size=11.5, line_size=9.0)
        arrow(ax, cx, y_chips_top - chip_h, cx, y_chips_top - chip_h - GAP)
    y = y_chips_top - chip_h - GAP

    scan_h = text_height(12.5, 3, 10.2) + line_h(9.4) + 4.4
    y = box(ax, L, R, y, scan_h, NAVY, "selective scan",
            "h[t] = exp(A·dt[t]) · h[t−1]",
            "         + α[t]·V[t]K[t]ᵀ + β[t]·V[t−1]K[t−1]ᵀ",
            "Y[t] = Q[t] · h[t]  +  D · V[t]",
            line_size=10.2, mono=True,
            note="chunked GEMM for prefill   ·   one step for decode")

    y = down(y)
    y_gate_top = y
    y = box(ax, L, R, y, text_height(12.5, 2, 9.6) + 4.0, NAVY, "gate by z",
            "RMSNorm first when is_outproj_norm is set",
            "MIMO: fold the rank axis back with mimo_o", line_size=9.6)
    y_gate_mid = (y_gate_top + y) / 2

    y = down(y)
    y = box(ax, L, R, y, line_h(12.5) + 4.0, NAVY, "out_proj")
    y = down(y, 4.4)
    pill(ax, CX, y, 6.0, 34, "out    (B, L, D)")

    # z leaves in_proj, skips the scan entirely and arrives at the gate.
    z_top = y_chips_top
    z_h = line_h(11.5) + 3.6
    box(ax, R + 3, 100, z_top, z_h, BLUE, "z", title_size=11.5)
    y_mid_inproj = (y_inproj_top + y_inproj_bot) / 2
    elbow(ax, [(R, y_mid_inproj), (LANE, y_mid_inproj), (LANE, z_top)])
    elbow(ax, [(LANE, z_top - z_h), (LANE, y_gate_mid), (R, y_gate_mid)], color=BLUE)

    fig.text(0.0, 0.995, "One Mamba-3 layer", fontsize=15.0, fontweight="semibold",
             color=INK, va="top")
    fig.text(0.0, 0.958, "Exponential-trapezoidal discretization, a rotary state space "
                         "and an optional rank-R", fontsize=10.6, color=MUTED, va="top")
    fig.text(0.0, 0.936, "read/write — in the order mamba3.py executes them.",
             fontsize=10.6, color=MUTED, va="top")
    fig.text(0.0, 0.045, "Decode carries one fixed-size state per layer: ssm (B, H, P, N) "
                         "plus k, v and the accumulated", fontsize=9.5, color=MUTED,
             va="top")
    fig.text(0.0, 0.025, "RoPE angle.   Shapes: H heads, P headdim, N d_state, R mimo_rank.",
             fontsize=9.5, color=MUTED, va="top")

    fig.savefig(ASSETS / "architecture.png")
    plt.close(fig)
    print("wrote architecture.png")


if __name__ == "__main__":
    main()
