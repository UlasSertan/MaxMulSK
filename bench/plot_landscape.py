#!/usr/bin/env python3
"""The wide view: what every FP32 GEMM path on this machine does across sizes.

UNLIKE plot_headline.py, THIS CHART IS NOT ONE MEASUREMENT. Its series come from
runs on different dates, and NumPy's stops at 2048 because that is where its
sweep ended. It is here to show tiers -- three orders of magnitude between a
scalar loop and SME -- not to rank neighbours. Anything within about 2x on this
chart should be read off the headline benchmark instead, which measures its rows
together, in one process, with the call order rotated.

Numbers are transcribed rather than parsed, because they come from four
different files. Each row states where it came from; keep that up to date.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps

W, H = 860, 460
PAD_L, PAD_R, PAD_T, PAD_B = 62, 178, 66, 62

SIZES = [256, 512, 1024, 2048, 4096]

# (label, values per size (None = not measured), colour, dash, date, weight)
SERIES = [
    ("MaxMulSK",            [1395.7, 1657.4, 1767.8, 1613.2, 1638.2], ps.BLUE,   None,  "09-09", 2.4),
    ("Apple Accelerate",    [1681.2, 1776.2, 1687.7, 1667.5, 1598.3], ps.SLATE,  None,  "09-09", 1.7),
    ("OpenBLAS, SME build", [1326.7, 1641.4, 1546.4, 1311.2, 1246.3], ps.ORANGE, None,  "09-09", 1.7),
    ("OpenBLAS, Homebrew",  [1624.7, 1667.4, 1574.0,  634.3,  113.9], ps.ORANGE, "5 4", "09-05", 1.7),
    ("Our NEON 8x12",       [ 113.2,  119.9,  122.3,  122.5,  123.1], ps.DIM,    None,  "09-05", 1.5),
    ("NumPy, vecLib",       [  88.6,  108.1,  106.5,  112.4,   None], ps.DIM,    "5 4", "09-05", 1.5),
    ("llama.cpp FP32 CPU",  [  42.0,   44.2,   42.5,   36.6,   35.8], ps.DIM,    "1 3", "09-08", 1.5),
]

LO, HI = 30.0, 2600.0
TICKS = [50, 100, 200, 500, 1000, 2000]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/img/landscape.svg"
    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    lg = math.log10

    inset = 26.0

    def X(i):
        return PAD_L + inset + (plot_w - 2 * inset) * i / (len(SIZES) - 1)

    def Y(v):
        return PAD_T + plot_h - (lg(v) - lg(LO)) / (lg(HI) - lg(LO)) * plot_h

    o = ps.head(W, H)
    o += ps.title(20, 24, "Single-thread FP32 GEMM on Apple M4 — the whole range",
                  "GFLOP/s, log scale. Series come from different dates: read this "
                  "for tiers, not for close calls.")

    for v in TICKS:
        o.append(ps.grid_line(PAD_L, W - PAD_R, Y(v)))
        o.append(f'<text x="{PAD_L-9}" y="{Y(v)+3.5:.1f}" text-anchor="end" '
                 f'font-size="10" fill="{ps.INK}">{v}</text>')

    for name, vals, colour, dash, date, width in SERIES:
        pts = [(X(i), Y(v)) for i, v in enumerate(vals) if v is not None]
        o.append(ps.series_path(pts, colour, width, dash))
        for x, y in pts:
            o.append(ps.marker(x, y, colour, 3.8 if name == "MaxMulSK" else 3.0))

    for i, s in enumerate(SIZES):
        o.append(f'<text x="{X(i):.1f}" y="{PAD_T+plot_h+20:.1f}" text-anchor="middle" '
                 f'font-size="10.5" fill="{ps.INK}">{s}&#179;</text>')
    o.append(ps.axis_line(PAD_L, W - PAD_R, PAD_T + plot_h))

    # Direct labels at the right edge. Two pairs nearly touch there -- MaxMulSK
    # against Accelerate at the top, NEON against NumPy in the lower tier -- so
    # they are stacked apart rather than left to overlap.
    ends = []
    for name, vals, colour, dash, date, _ in SERIES:
        last_i = max(i for i, v in enumerate(vals) if v is not None)
        ends.append((Y(vals[last_i]), (name, colour, date, X(last_i), Y(vals[last_i]))))
    for ly, (name, colour, date, mx, my) in ps.stack(ends, min_gap=26.0, offset=0.0):
        lx = W - PAD_R + 14
        # a hairline from the marker to its label, so a pushed label stays attached
        o.append(f'<path d="M {mx+6:.1f} {my:.1f} L {lx-5:.1f} {ly:.1f}" fill="none" '
                 f'stroke="{colour}" stroke-width="0.8" opacity="0.45"/>')
        o.append(f'<text x="{lx}" y="{ly-1:.1f}" font-size="10.5" '
                 f'font-weight="{600 if name == "MaxMulSK" else 400}" '
                 f'fill="{colour}">{ps.esc(name)}</text>')
        o.append(f'<text x="{lx}" y="{ly+11:.1f}" font-size="8.5" '
                 f'fill="{ps.FAINT}">measured {date}</text>')

    o.append(f'<text x="20" y="{H-14}" font-size="9.5" fill="{ps.FAINT}">'
             f'NumPy stops at 2048&#179;, where its sweep ended. The two OpenBLAS '
             f'lines are the same library built two ways: dashed is what '
             f'brew install gives you, solid is what it can do.</text>')
    o.append("</svg>")
    open(out, "w").write("\n".join(o) + "\n")
    print(f"  {out}")
    ps.verify(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
