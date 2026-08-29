"""Shared drawing style: palette, type scale, fonts, rounded cards.

One place for all three figure scripts, so the plots and the diagrams cannot
drift apart.

The figures are read on GitHub, usually on a phone that scales the image down
to the column width. That is what the type scale is for: a narrow canvas with
large type survives the scaling, a wide one with small type does not. Sizes are
generous on purpose — a figure has to be legible at a glance, or the reader
skips it and reads the table instead.

Text is set in HarmonyOS Sans SC, in the three weights under fonts/. See
make_fonts.py for where those come from.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

INK = "#1A1D23"
MUTED = "#5C6370"
FAINT = "#B0B6C0"      # setup lines: present if looked for, invisible if not
HAIR = "#D5D8DE"
GRID = "#EEF0F3"
TEAL = "#0F7B6C"
CORAL = "#C45C26"
NAVY = "#1E3A5F"
BLUE = "#1D4ED8"
PLUM = "#7C3AED"
SLATE = "#64748B"
WIRE = "#98A2B3"

FILL = {
    NAVY: "#EDF1F7",
    TEAL: "#E7F2F0",
    CORAL: "#FBEEE6",
    PLUM: "#F1EBFB",
    BLUE: "#E9EEFC",
}

# One scale for every figure. Raising a number here raises it everywhere, which
# is the only way a set of figures stays consistent as they get edited.
TITLE = 17.5      # figure heading
SUB = 12.8        # one line under the heading
PANEL = 14.5      # what a single panel shows
TICK = 12.8
LABEL = 13.2
LEGEND = 12.6
NOTE = 12.8       # callouts inside the axes
FOOT = 11.0       # small type that still has to be read: units, thresholds
META = 8.6        # the setup line under the figure, for whoever goes looking

FAMILY = "HarmonyOS Sans SC"
FONTS = Path(__file__).resolve().parent / "fonts"

KAPPA = 0.5522847498307936      # cubic Bezier control arm for a quarter circle


def _font_stack() -> tuple[list[str], str]:
    """Register the bundled weights; return the families and the medium weight.

    Two things are decided together here because they depend on each other.

    The bundled family covers Latin, Greek and the punctuation the labels use
    in three weights, so when it is present it is the whole list. Adding a
    fallback behind it would cost a weight: at draw time matplotlib resolves
    every family in the list, and a system sans has no semibold, so each one
    warns even though the text is being set correctly in the first family.

    Without the files, the system sans is used instead — and then semibold is
    genuinely unavailable, so medium emphasis has to be bold.
    """
    files = sorted(FONTS.glob("HarmonyOS_Sans_SC-*.ttf"))
    for path in files:
        font_manager.fontManager.addfont(str(path))
    if files:
        return [FAMILY], "semibold"
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    system = [n for n in ("Segoe UI", "Helvetica Neue", "Inter") if n in installed]
    return system + ["DejaVu Sans"], "bold"


FAMILIES, SEMI = _font_stack()


def use_style() -> None:
    plt.rcParams.update({
        "font.family": FAMILIES,
        "font.size": LABEL,
        "axes.titlesize": PANEL,
        "axes.titleweight": SEMI,
        "axes.titlepad": 11.0,
        "axes.labelsize": LABEL,
        "axes.labelpad": 7.0,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": HAIR,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 7.0,
        "ytick.major.pad": 6.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "legend.fontsize": LEGEND,
        "legend.borderaxespad": 0.2,
        "legend.handletextpad": 0.6,
        "lines.linewidth": 2.3,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.26,
    })


def lh(size: float) -> float:
    """Height of one line of text, in inches."""
    return size * 1.34 / 72.0


def rounded_rect(x0: float, y0: float, x1: float, y1: float, r: float) -> MplPath:
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
    arc = [MplPath.CURVE4] * 3
    codes = ([MplPath.MOVETO, MplPath.LINETO] + arc + [MplPath.LINETO] + arc
             + [MplPath.LINETO] + arc + [MplPath.LINETO] + arc)
    return MplPath(verts, codes)


def card(ax, x0, y0, x1, y1, color, radius, lw=1.4, zorder=2) -> None:
    ax.add_patch(PathPatch(
        rounded_rect(x0, y0, x1, y1, radius), linewidth=lw, edgecolor=color,
        facecolor=FILL.get(color, "white"), zorder=zorder,
    ))
