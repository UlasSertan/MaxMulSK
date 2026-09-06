#!/usr/bin/env python3
"""Parse one powermetrics capture into average power figures.

Usage: energy_parse.py <powermetrics.raw> <workload_wall_seconds>

powermetrics is started just before the workload and killed just after, so the
first sample can straddle the idle gap before the workload really gets going;
it is dropped. Energy is reported as avg_power * workload_wall_seconds, using
the workload's own measured wall time rather than powermetrics' capture length,
so start-up and tear-down of the sampler do not leak into the number.

Prints "key=value" lines; unknown/absent counters are simply omitted.
"""
import re
import sys
from collections import defaultdict

# "CPU Power: 1234 mW" -> group(1) is the bare name, without " Power".
RE = re.compile(r"^([A-Za-z][\w\- ]*?)\s+Power:\s+([\d.]+)\s*mW", re.M)
# The combined line carries a parenthesised list before the colon, so it needs
# its own pattern.
RE_COMBINED = re.compile(r"^Combined Power \([^)]*\):\s+([\d.]+)\s*mW", re.M)

# Only these are reported; anything else powermetrics prints is noise here.
KEEP = {"CPU": "cpu_w",
        "P-Cluster": "pcluster_w",
        "E-Cluster": "ecluster_w",
        "Package": "package_w"}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: energy_parse.py <raw> <wall_seconds>", file=sys.stderr)
        return 2
    raw_path, wall = sys.argv[1], float(sys.argv[2])
    try:
        text = open(raw_path, errors="ignore").read()
    except OSError as exc:
        print(f"error={exc}")
        return 1

    series = defaultdict(list)
    for name, val in RE.findall(text):
        name = name.strip()
        if name in KEEP:
            series[KEEP[name]].append(float(val))
    for val in RE_COMBINED.findall(text):
        series["combined_w"].append(float(val))

    if not series:
        print("error=no_power_samples")
        return 1

    for key, vals in sorted(series.items()):
        if len(vals) > 1:
            vals = vals[1:]          # drop the straddling first sample
        avg_w = sum(vals) / len(vals) / 1000.0
        print(f"{key}={avg_w:.4f}")
        print(f"{key.replace('_w', '')}_j={avg_w * wall:.4f}")
    print(f"samples={max(len(v) for v in series.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
