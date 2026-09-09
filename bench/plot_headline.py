#!/usr/bin/env python3
"""Render the headline comparison as two SVGs, straight from the benchmark CSVs.

SVG on purpose: it is text, so it diffs, it renders on GitHub in both themes,
and regenerating it needs no plotting library installed.

Usage: plot_headline.py <out_dir> <run1.csv> [run2.csv ...]
       Median across runs per (shape, implementation).
"""
import collections
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps

W, H = 820, 400
PAD_L, PAD_R, PAD_T, PAD_B = 62, 132, 62, 52

SERIES = [
    ("MaxMulSK",   ps.BLUE,   2.2),
    ("Accelerate", ps.SLATE,  1.7),
    ("OpenBLAS",   ps.ORANGE, 1.7),
]


def chart(path, heading, sub, labels, data, ymin, ymax):
    """Points joined by thin lines, one series each, labelled at the right edge.

    Points rather than bars because the y axis does not start at zero: a bar's
    length encodes its value, so cutting the bottom off inflates every visible
    difference, while a point encodes its value by position and can be zoomed
    honestly. The floor is stated in the subtitle and every marker carries its
    number, so nothing is hidden by the crop.
    """
    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    span = ymax - ymin

    # Inset the first and last point from the axis. Value labels are centred on
    # their marker, so a point sitting exactly on the left edge puts its label on
    # top of the y-axis numbers.
    inset = 34.0

    def X(i):
        return PAD_L + inset + (plot_w - 2 * inset) * i / (len(labels) - 1)

    def Y(v):
        return PAD_T + plot_h - (v - ymin) / span * plot_h

    o = ps.head(W, H)
    o += ps.title(20, 24, heading, sub)

    grid = 100 if span <= 700 else (200 if span <= 1500 else 250)
    v = ymin
    while v <= ymax + 1:
        o.append(ps.grid_line(PAD_L, W - PAD_R, Y(v)))
        o.append(f'<text x="{PAD_L-9}" y="{Y(v)+3.5:.1f}" text-anchor="end" '
                 f'font-size="10" fill="{ps.INK}">{int(v)}</text>')
        v += grid

    # value labels, one stack per x so three numbers never sit on top of each other
    for i in range(len(labels)):
        col = [(Y(data[n][i]), (n, c, data[n][i])) for n, c, _ in SERIES]
        for ly, (n, c, val) in ps.stack(col, min_gap=12.5, offset=-9.0):
            o.append(f'<text x="{X(i):.1f}" y="{ly:.1f}" text-anchor="middle" '
                     f'font-size="9.5" fill="{c}">{val:.0f}</text>')

    for name, colour, width in SERIES:
        pts = [(X(i), Y(v)) for i, v in enumerate(data[name])]
        o.append(ps.series_path(pts, colour, width))
        for x, y in pts:
            o.append(ps.marker(x, y, colour, 4.2 if name == "MaxMulSK" else 3.6))

    # direct labels at the right edge, also de-collided
    ends = [(Y(data[n][-1]), (n, c)) for n, c, _ in SERIES]
    for ly, (name, colour) in ps.stack(ends, min_gap=15.0, offset=3.5):
        o.append(f'<text x="{W-PAD_R+12}" y="{ly:.1f}" font-size="11" '
                 f'font-weight="{600 if name == "MaxMulSK" else 400}" '
                 f'fill="{colour}">{ps.esc(name)}</text>')

    for i, label in enumerate(labels):
        o.append(f'<text x="{X(i):.1f}" y="{PAD_T+plot_h+20:.1f}" text-anchor="middle" '
                 f'font-size="10.5" fill="{ps.INK}">{ps.esc(label)}</text>')
    o.append(ps.axis_line(PAD_L, W - PAD_R, PAD_T + plot_h))
    o.append("</svg>")
    open(path, "w").write("\n".join(o) + "\n")
    print(f"  {path}")
    ps.verify(path)


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    out_dir, csvs = sys.argv[1], sys.argv[2:]

    acc, order = collections.defaultdict(list), []
    for path in csvs:
        for r in csv.DictReader(open(path)):
            shape = f"{r['M']}x{r['K']}x{r['N']}"
            if (r["group"], shape) not in order:
                order.append((r["group"], shape))
            try:
                acc[(r["group"], shape, r["implementation"])].append(float(r["gflops"]))
            except ValueError:
                pass

    for group, heading, fname, fmt in [
        ("square", "Square GEMM", "headline_square.svg",
         lambda s: s.split("x")[0] + "³"),
        ("llm", "LLM-shaped GEMM (K is the long axis, N = 512)", "headline_llm.svg",
         lambda s: "M={}\nK={}".format(*s.split("x")[:2])),
    ]:
        labels, data = [], collections.defaultdict(list)
        for g, shape in order:
            if g != group:
                continue
            vals = {n: acc.get((g, shape, n), []) for n, _, _ in SERIES}
            if not all(vals.values()):
                continue
            labels.append(fmt(shape).replace("\n", " "))
            for n, _, _ in SERIES:
                data[n].append(statistics.median(vals[n]))
        if not labels:
            continue
        lo = min(min(v) for v in data.values())
        hi = max(max(v) for v in data.values())
        # Zoom to the data. A fixed floor leaves the series crowded into a band
        # too narrow to read, which defeats the point of drawing a chart at all.
        ymin = int(lo / 100) * 100
        ymax = (int((hi + (hi - lo) * 0.12) / 100) + 1) * 100
        sub = (f"GFLOP/s, single thread, Apple M4 · median of three runs "
               f"· y axis starts at {ymin}, not 0")
        chart(f"{out_dir}/{fname}", heading, sub, labels, data, ymin, ymax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
