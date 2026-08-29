"""Build the small font files in scripts/fonts/ from HarmonyOS Sans SC.

Run once; the output is committed, so drawing a figure needs neither fontTools
nor the original file.

    pip install fonttools
    python scripts/make_fonts.py path/to/HarmonyOS_Sans_SC.ttf

The source is a variable font (wght 40-900) shipped as a single default
instance, which matplotlib reads as Regular only — ask it for bold and it warns
and hands back 400. So the weights are instanced out here instead, and the CJK
glyphs are dropped: the figures are all Latin, and that turns 20 MB into a few
tens of KB, small enough to keep in the repo.

HarmonyOS Sans is Huawei's, free to use and redistribute:
https://developer.huawei.com/consumer/cn/design/resource/
"""

from __future__ import annotations

import sys
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

FONTS = Path(__file__).resolve().parent / "fonts"

WEIGHTS = {400: "Regular", 600: "SemiBold", 700: "Bold"}

# Whole ranges rather than the characters currently in use, because a glyph
# that turns out to be missing does not fail loudly: matplotlib falls back to
# DejaVu for that one string, which has no semibold, so the figure silently
# loses a weight and prints a warning about the weight instead of the cause.
# ASCII, Latin-1 (µ for microseconds lives there, and is not Greek mu), Greek,
# and the punctuation and superscripts the labels are written with. Still only
# a few hundred glyphs; the size is all in the CJK being dropped.
CHARS = (
    "".join(chr(c) for c in range(0x20, 0x7F))          # ASCII
    + "".join(chr(c) for c in range(0xA0, 0x100))       # Latin-1 supplement
    + "".join(chr(c) for c in range(0x391, 0x3CA))      # Greek
    + "\u2212\u2013\u2014\u2026\u2192\u2190\u2191\u2193\u2194"
    + "\u2248\u2264\u2265\u2260\u221e\u2022\u2713\u2717\u2032\u2033"
    + "\u2018\u2019\u201c\u201d\u2039\u203a"
    + "\u2070\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b"
    + "".join(chr(c) for c in range(0x2080, 0x208A))    # subscript digits
)


def build(src: Path) -> None:
    FONTS.mkdir(exist_ok=True)
    for wght, style in WEIGHTS.items():
        font = instancer.instantiateVariableFont(
            TTFont(src, fontNumber=0), {"wght": wght}, inplace=False,
        )
        # Named after the fact, not by updateFontNames: that reads the source's
        # STAT table and refuses a weight with no named instance there, which
        # rules out 600. Naming is also the one thing that has to be right —
        # matplotlib matches on family first and weight second, so the three
        # files have to agree on family to be selectable by weight at all.
        name, os2 = font["name"], font["OS/2"]
        for rec in list(name.names):
            if rec.nameID in (1, 16):
                name.setName("HarmonyOS Sans SC", rec.nameID, rec.platformID,
                             rec.platEncID, rec.langID)
            elif rec.nameID in (2, 17):
                name.setName(style, rec.nameID, rec.platformID, rec.platEncID,
                             rec.langID)
            elif rec.nameID == 4:
                name.setName(f"HarmonyOS Sans SC {style}", rec.nameID,
                             rec.platformID, rec.platEncID, rec.langID)
            elif rec.nameID == 6:
                name.setName(f"HarmonyOSSansSC-{style}", rec.nameID,
                             rec.platformID, rec.platEncID, rec.langID)
        os2.usWeightClass = wght
        # Style bits, which the source set for its own default instance: the
        # regular bit excludes the bold one and has to travel with macStyle.
        bold, regular = wght >= 700, style == "Regular"
        os2.fsSelection &= ~0x61
        os2.fsSelection |= (0x20 if bold else 0) | (0x40 if regular else 0)
        font["head"].macStyle = (font["head"].macStyle & ~0x01) | (0x01 if bold else 0)

        opts = subset.Options(layout_features=["kern", "liga", "calt"],
                              name_IDs=[1, 2, 3, 4, 5, 6, 16, 17], notdef_outline=True,
                              recalc_bounds=True, drop_tables=["DSIG"])
        subsetter = subset.Subsetter(options=opts)
        subsetter.populate(text=CHARS)
        subsetter.subset(font)

        out = FONTS / f"HarmonyOS_Sans_SC-{style}.ttf"
        font.save(out)
        print(f"wrote {out.name}  {out.stat().st_size / 1024:.0f} KB  "
              f"({len(font.getBestCmap())} glyphs, weight {wght})")

        # Asking for a character the source does not have is silent otherwise,
        # and shows up much later as a box in a figure. The source has no
        # superscript minus, for one, so exponents are written 1e-6 rather than
        # 10⁻⁶.
        cmap = font.getBestCmap()
        missing = sorted({c for c in CHARS if ord(c) not in cmap})
        if missing:
            print("  not in the source font: "
                  + " ".join(f"U+{ord(c):04X}" for c in missing))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(Path(sys.argv[1]))
