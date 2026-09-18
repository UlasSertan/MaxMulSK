#!/usr/bin/env python3
"""Grouped horizontal bars, MaxMulSK vs Apple Accelerate vs MpGEMM, split by
workload regime. Reads bench/results/2026-09-10/vs_mpgemm.csv.

Only the 1x4 kernel is drawn for MaxMulSK: it is the best of our three on 33 of
the 35 shapes, so plotting all three would triple the ink for almost no signal.
The per-shape numbers for 2x2 and 4x1 are in the CSV.
"""
import csv, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "results", "2026-09-10", "vs_mpgemm.csv")
OUT  = os.path.join(HERE, "results", "2026-09-10", "vs_mpgemm.svg")

SERIES = [("MaxMulSK 1x4", "MaxMulSK-1x4", ps.BLUE),
          ("Apple Accelerate", "Accelerate", ps.SLATE),
          ("MpGEMM", "MpGEMM-sgemm", ps.ORANGE)]

def regime(tag):
    if tag == "square": return 0
    if tag == "llm":    return 1
    if tag.startswith("llama"): return 4
    i = int(tag.split("id")[1])
    return 2 if i <= 12 else 3

PANELS = [
    ("Square", "M = N = K"),
    ("LLM", "K is the long axis, N = 512"),
    ("DeepSeek, small M", "M = 64 or 128 — tall-thin activations, IDs 1–12"),
    ("DeepSeek, large M", "M = 4096, IDs 13–18"),
    ("LLaMA", "N = 256 — narrow output, IDs 19–24"),
]

rows = list(csv.DictReader(open(SRC)))
shapes = collections.OrderedDict()
for r in rows:
    key = (int(r["shape"]), r["tag"], r["M"], r["K"], r["N"])
    shapes.setdefault(key, {})[r["impl"]] = float(r["GFLOPs"])

groups = collections.defaultdict(list)
for key, d in shapes.items():
    groups[regime(key[1])].append((key, d))

# One x-scale for every panel, so bar lengths mean the same thing throughout.
XMAX = 2000.0
X0, XW = 250, 520
BAR, BGAP, ROW = 8, 4.5, 48.0   # >=12.5px between bar rows so value labels cannot collide

def x(v): return X0 + XW * min(v, XMAX) / XMAX

o = []
H = 130 + sum(78 + ROW * len(groups[g]) for g in range(5))
o += ps.head(900, int(H))
o += ps.title(40, 52, "MaxMulSK vs MpGEMM vs Apple Accelerate",
                  "Single-thread FP32 GEMM, Apple M4. Higher is better. "
                  "Measured 2026-09-10; every implementation returns the same result on all 35 shapes.")

# legend
lx = 40
for name, _, col in SERIES:
    o.append(f'<rect x="{lx}" y="82" width="10" height="10" fill="{col}"/>')
    o.append(f'<text x="{lx+15}" y="91" font-family="{ps.FONT}" font-size="11.5" fill="{ps.INK}">{ps.esc(name)}</text>')
    lx += 22 + 7.2 * len(name)

y = 122.0
for g in range(5):
    name, sub = PANELS[g]
    o.append(f'<text x="40" y="{y}" font-family="{ps.FONT}" font-size="13.5" font-weight="600" fill="#24292f">{ps.esc(name)}</text>')
    o.append(f'<text x="{40+9.0*len(name)}" y="{y}" font-family="{ps.FONT}" font-size="11" fill="{ps.FAINT}">{ps.esc(sub)}</text>')
    y += 20
    for gv in (500, 1000, 1500, 2000):
        o.append(f'<line x1="{x(gv)}" y1="{y-4}" x2="{x(gv)}" y2="{y + ROW*len(groups[g]) - 6}" stroke="{ps.GRID}" stroke-width="1"/>')
    for (key, d) in groups[g]:
        _, tag, M, K, N = key
        label = f"{M}×{K}×{N}"
        idtxt = tag.split("-")[1].replace("id", "ID ") if "-id" in tag else ""
        o.append(f'<text x="{X0-10}" y="{y+13}" text-anchor="end" font-family="{ps.FONT}" font-size="11.5" fill="#24292f">{ps.esc(label)}</text>')
        if idtxt:
            o.append(f'<text x="{X0-10}" y="{y+25}" text-anchor="end" font-family="{ps.FONT}" font-size="9.5" fill="{ps.FAINT}">{ps.esc(idtxt)}</text>')
        best = max(d.get(k, 0) for _, k, _ in SERIES)
        by = y + 3
        for _, k, col in SERIES:
            v = d.get(k, 0.0)
            w = max(x(v) - X0, 1)
            op = "1" if v >= best - 1e-9 else "0.72"
            o.append(f'<rect x="{X0}" y="{by}" width="{w - X0}" height="{BAR}" fill="{col}" opacity="{op}"/>')
            o.append(f'<text x="{x(v)+6}" y="{by+BAR-0.5}" font-family="{ps.FONT}" font-size="10" fill="{ps.INK}">{v:.0f}</text>')
            by += BAR + BGAP
        y += ROW
    y += 58

o.append(f'<text x="40" y="{H-26}" font-family="{ps.FONT}" font-size="10.5" fill="{ps.FAINT}">'
         'Shapes labelled M×K×N. DeepSeek and LLaMA IDs follow Table III of the MpGEMM paper, which lists them as M, N, K. '
         'Reps calibrated on the slowest entrant; call order rotated as a cyclic Latin square; median of per-call timings.</text>')
o.append(f'<text x="40" y="{H-12}" font-family="{ps.FONT}" font-size="10.5" fill="{ps.FAINT}">'
         'MaxMulSK 2x2 and 4x1 are omitted: 1x4 is the fastest of the three on 33 of 35 shapes. Full data in vs_mpgemm.csv.</text>')
o.append("</svg>")
open(OUT, "w").write("\n".join(o))
ps.verify(OUT)
print("yazildi:", OUT)
for g in range(5):
    ws = sum(1 for k, d in groups[g] if d["MaxMulSK-1x4"] > d["MpGEMM-sgemm"] * 1.05)
    ls = sum(1 for k, d in groups[g] if d["MaxMulSK-1x4"] < d["MpGEMM-sgemm"] * 0.95)
    print(f"  {PANELS[g][0]:20} n={len(groups[g]):2}  kazanc={ws} kayip={ls} berabere={len(groups[g])-ws-ls}")
