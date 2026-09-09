"""Shared chart style. Kept in one place so the charts cannot drift apart.

Design constraints this file encodes:
  - readable on GitHub's light AND dark themes, so every colour is a mid-tone
    and nothing relies on the page background
  - no gradients, no shadows, no icons; colour carries data and nothing else
  - labels never overlap, which is handled by stack() rather than by hoping
"""

FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif")

INK    = "#8b949e"   # axis text, category labels
FAINT  = "#9aa4ae"   # subtitles, footnotes
GRID   = "#d0d7de"

BLUE   = "#0969da"   # MaxMulSK
SLATE  = "#57606a"   # Apple Accelerate
ORANGE = "#bc4c00"   # OpenBLAS
DIM    = "#8c959f"   # everything in the lower tier


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def head(w, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}" font-size="11">']


def title(x, y, text, sub=None):
    o = [f'<text x="{x}" y="{y}" font-size="13.5" font-weight="600" '
         f'fill="{INK}">{esc(text)}</text>']
    if sub:
        o.append(f'<text x="{x}" y="{y+16}" font-size="10.5" '
                 f'fill="{FAINT}">{esc(sub)}</text>')
    return o


def stack(items, min_gap=12.5, offset=-9.0):
    """Place one label per point so that none of them collide.

    items: [(y, payload)] at their marker positions. Returns [(label_y, payload)]
    in the same order as the input. Labels want to sit `offset` above their
    marker; when that would put two of them closer than `min_gap`, the lower one
    is pushed down instead. Greedy top-to-bottom, which is enough for the three
    to seven series these charts carry and is stable frame to frame.
    """
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    placed = [0.0] * len(items)
    last = -1e9
    for i in order:
        y = items[i][0] + offset
        if y - last < min_gap:
            y = last + min_gap
        placed[i] = y
        last = y
    return [(placed[i], items[i][1]) for i in range(len(items))]


def grid_line(x1, x2, y):
    return (f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1" opacity="0.38"/>')


def axis_line(x1, x2, y):
    return (f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
            f'stroke="{INK}" stroke-width="1" opacity="0.55"/>')


def series_path(points, colour, width=1.8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" stroke-linejoin="round" '
            f'stroke-linecap="round"{d}/>')


def marker(x, y, colour, r=4.0):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{colour}"/>'


def verify(path):
    """Re-read a written SVG and complain if any two text labels overlap.

    Charts are regenerated whenever the numbers change, and label collisions
    come back silently when a value moves. Estimating each <text> box from its
    anchor, font size and length is crude, but it catches the case that actually
    happens: two labels landing on the same spot.
    """
    import itertools
    import re

    src = open(path).read()
    boxes = []
    for m in re.finditer(r"<text([^>]*)>(.*?)</text>", src, re.S):
        attrs, body = m.group(1), re.sub(r"&[a-z]+;|&#\d+;", "X", m.group(2))
        def attr(name, dflt=None):
            hit = re.search(name + r'="([^"]*)"', attrs)
            return hit.group(1) if hit else dflt
        x, y = float(attr("x", 0)), float(attr("y", 0))
        fs = float(attr("font-size", 11))
        w = len(body.strip()) * fs * 0.56
        anchor = attr("text-anchor", "start")
        x0 = x - w / 2 if anchor == "middle" else (x - w if anchor == "end" else x)
        boxes.append((x0, y - fs * 0.8, x0 + w, y + fs * 0.25, body.strip()[:24]))

    clashes = [(a, b) for a, b in itertools.combinations(boxes, 2)
               if not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])]
    if clashes:
        print(f"    WARNING: {len(clashes)} overlapping labels in {path}")
        for a, b in clashes[:6]:
            print(f"      {a[4]!r} <-> {b[4]!r}")
    return not clashes
