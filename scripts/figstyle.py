"""Shared drawing style for the diagram scripts (arch_figure, stats_figure).

Both lay out in inches on axes set 1:1, so text metrics and corner radii share
one unit and nothing gets stretched.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

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

KAPPA = 0.5522847498307936      # cubic Bezier control arm for a quarter circle


def use_style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


def lh(size: float) -> float:
    """Height of one line of text, in inches."""
    return size * 1.34 / 72.0


def rounded_rect(x0: float, y0: float, x1: float, y1: float, r: float) -> Path:
    """Rectangle whose corners are circular arcs.

    matplotlib's "round" box style puts a quadratic control point on the corner
    itself. That curve passes 0.354r from the corner where a circle passes
    0.293r, so it bulges out and kinks visibly where it meets the straight edge.
    """
    k = KAPPA * r
    verts = [
        (x0 + r, y0),
        (x1 - r, y0),
        (x1 - r + k, y0), (x1, y0 + r - k), (x1, y0 + r),
        (x1, y1 - r),
        (x1, y1 - r + k), (x1 - r + k, y1), (x1 - r, y1),
        (x0 + r, y1),
        (x0 + r - k, y1), (x0, y1 - r + k), (x0, y1 - r),
        (x0, y0 + r),
        (x0, y0 + r - k), (x0 + r - k, y0), (x0 + r, y0),
    ]
    arc = [Path.CURVE4] * 3
    codes = ([Path.MOVETO, Path.LINETO] + arc + [Path.LINETO] + arc
             + [Path.LINETO] + arc + [Path.LINETO] + arc)
    return Path(verts, codes)


def card(ax, x0, y0, x1, y1, color, radius, lw=1.4, zorder=2) -> None:
    ax.add_patch(PathPatch(
        rounded_rect(x0, y0, x1, y1, radius), linewidth=lw, edgecolor=color,
        facecolor=FILL.get(color, "white"), zorder=zorder,
    ))
