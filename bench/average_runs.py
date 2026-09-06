#!/usr/bin/env python3
"""Combine repeated benchmark runs into one CSV.

Between-run variance on this machine is materially larger than the within-run
sample spread (no thread pinning on macOS; see docs/BENCHMARKS.md), so a single
run is not a reliable number. This takes the MEDIAN across runs per row and adds
a `spread_pct` column, (max-min)/median over the runs, so any figure quoted from
the result carries its own reliability estimate.

Handles both CSV schemas produced in bench/: the GEMM comparison
(benchmark_maxmul_vs_kleidiai) and the energy profile (energy_bench.sh). The
schema is detected from the header, so the call site is the same for both.

For energy rows the median is taken over each column independently, including
the derived cpu_j and j_per_gflop. That is the median of the observed ratios,
not a ratio recomputed from medians -- the former is what was actually measured.

Usage: average_runs.py out.csv in1.csv in2.csv [in3.csv ...]
"""
import csv
import statistics
import sys

GEMM_KEY = ("experiment", "mode", "M", "N", "K", "implementation", "variant")
GEMM_NUM = ("median_ms", "mean_ms", "min_ms", "stddev_ms", "gflops",
            "ns_per_invocation")

ENERGY_KEY = ("impl", "M", "K", "N")
ENERGY_NUM = ("iters", "wall_s", "gflop", "gflops", "cpu_w", "pcluster_w",
              "ecluster_w", "cpu_j", "j_per_gflop", "samples")

INTEGRAL = {"iters", "samples"}


def schema(fields):
    """Pick (key columns, numeric columns) from the CSV header."""
    if "j_per_gflop" in fields:
        return ENERGY_KEY, ENERGY_NUM
    return GEMM_KEY, GEMM_NUM


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    out_path, in_paths = sys.argv[1], sys.argv[2:]

    with open(in_paths[0]) as f:
        key_cols, num_cols = schema(csv.DictReader(f).fieldnames or [])

    runs = []
    for p in in_paths:
        with open(p) as f:
            runs.append({tuple(r[k] for k in key_cols): r
                         for r in csv.DictReader(f)})

    base = runs[0]
    fields = list(next(iter(base.values())).keys())
    for extra in ("spread_pct", "n_runs"):
        if extra not in fields:
            fields.append(extra)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, row in base.items():
            present = [r[key] for r in runs if key in r]
            merged = dict(row)
            for col in num_cols:
                vals = []
                for r in present:
                    try:
                        vals.append(float(r.get(col, "")))
                    except ValueError:
                        pass          # "NA" and friends stay as-is
                if vals:
                    m = statistics.median(vals)
                    merged[col] = f"{m:.0f}" if col in INTEGRAL else f"{m:.6f}"
            g = []
            for r in present:
                try:
                    g.append(float(r["gflops"]))
                except ValueError:
                    pass
            if g and statistics.median(g) > 0:
                merged["spread_pct"] = f"{100 * (max(g) - min(g)) / statistics.median(g):.2f}"
            else:
                merged["spread_pct"] = "NA"
            merged["n_runs"] = str(len(present))
            w.writerow(merged)

    print(f"{out_path}: merged {len(in_paths)} runs, {len(base)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
