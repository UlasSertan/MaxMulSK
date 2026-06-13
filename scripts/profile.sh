#!/usr/bin/env bash
# Run the GemmTemplate Instruments trace and produce a compact summary.
#
# Usage:
#   ./scripts/profile.sh                    # output saved as traces/<timestamp>.txt
#   ./scripts/profile.sh 4x1                # output saved as traces/4x1.txt
#   ./scripts/profile.sh --no-build 4x1     # skip rebuild
#   ./scripts/profile.sh --keep-trace 4x1   # also keep the full .trace bundle
#
# Produces:
#   traces/NAME.txt     — summary (GFLOPS from stdout + counter totals)
#   traces/NAME.log     — raw binary stdout
#   traces/NAME.trace   — full Instruments bundle (only with --keep-trace)
#
# Edit main.cpp to leave only the kernel you want to measure active.

set -eo pipefail

if ! command -v cmake &>/dev/null; then
    export PATH="/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin:$PATH"
fi

# Anchor to repo root so relative paths resolve regardless of caller's cwd
cd "$(dirname "$0")/.."

BUILD_DIR="cmake-build-release"
BINARY="$BUILD_DIR/MatrixLibrary"
TEMPLATE="GemmTemplate.tracetemplate"
OUT_DIR="traces"
JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)

DO_BUILD=1
KEEP_TRACE=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)   DO_BUILD=0; shift ;;
        --keep-trace) KEEP_TRACE=1; shift ;;
        --binary)     BINARY="$2"; DO_BUILD=0; shift 2 ;;
        --args)       IFS=' ' read -r -a EXTRA_ARGS <<< "$2"; shift 2 ;;
        -*)           echo "unknown flag: $1" >&2; exit 1 ;;
        *)            break ;;
    esac
done

TARGET_PROCESS=$(basename "$BINARY")

NAME="${1:-$(date +%Y%m%d_%H%M%S)}"
TRACE="$OUT_DIR/$NAME.trace"
LOG="$OUT_DIR/$NAME.log"
SUMMARY="$OUT_DIR/$NAME.txt"

[[ -f "$TEMPLATE" ]] || { echo "ERROR: $TEMPLATE not found" >&2; exit 1; }

mkdir -p "$OUT_DIR"
rm -rf "$TRACE" "$LOG" "$SUMMARY"

if [[ "$DO_BUILD" == "1" ]]; then
    echo "==> Configuring..."
    cmake -B "$BUILD_DIR" \
          -DCMAKE_BUILD_TYPE=Release \
          --no-warn-unused-cli -Wno-dev \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
          2>&1 | grep -v "^--" || true

    echo "==> Building ($JOBS jobs)..."
    cmake --build "$BUILD_DIR" -j"$JOBS"
fi

[[ -x "$BINARY" ]] || { echo "ERROR: missing $BINARY" >&2; exit 1; }

echo "==> Profiling ($TEMPLATE) [target=$TARGET_PROCESS] ..."
set +e
xcrun xctrace record \
    --template "$TEMPLATE" \
    --output "$TRACE" \
    --target-stdout - \
    --launch -- "$BINARY" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
[[ $RC -eq 0 ]] || { echo "xctrace failed (rc=$RC)"; exit $RC; }

# ---- Extract counter totals + GFLOPS into a compact .txt ----
echo "==> Extracting counters -> $SUMMARY"
TARGET_PROCESS="$TARGET_PROCESS" python3 - "$TRACE" "$LOG" "$SUMMARY" <<'PY'
import sys, subprocess
from xml.etree import ElementTree as ET

trace, log, out = sys.argv[1], sys.argv[2], sys.argv[3]

import os
# GemmTemplate counter order — must match the template's counter list.
# If you edit the template's counters, update this list in the same order.
COUNTER_LABELS = ["L1D_CACHE_MISS_LD", "L1D_CACHE_MISS_ST", "INST_ALL"]
TARGET_PROCESS = os.environ.get("TARGET_PROCESS", "MatrixLibrary")

def xctrace_export(*args):
    return subprocess.run(
        ["xcrun", "xctrace", "export", "--input", trace, *args],
        capture_output=True, text=True, check=True).stdout

lines = []

# --- Binary stdout: keep GFLOPS / iters / size lines ---
lines.append("=== Run info ===")
with open(log) as f:
    for ln in f:
        s = ln.rstrip()
        if any(k in s for k in ("GFLOPS", "profile runs", "Threads:")):
            if s.strip() and "==>" not in s:
                lines.append(s)

lines.append("")
lines.append("=== Counter totals ===")

try:
    xml = xctrace_export("--xpath",
        '/trace-toc/run[@number="1"]/data/table[@schema="counters-profile"]')
    root = ET.fromstring(xml)
except Exception as e:
    lines.append(f"[counters-profile read failed: {e}]")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0)

# Build process-id → name map (for filtering to MatrixLibrary only)
proc_name = {}
for p in root.iter("process"):
    pid = p.attrib.get("id")
    fmt = p.attrib.get("fmt", "")
    if pid and fmt:
        proc_name[pid] = fmt

# Iterate rows, filter by target process, sum pmc-events columns
n = len(COUNTER_LABELS)
totals = [0] * n
samples = 0
skipped = 0

for row in root.iter("row"):
    # find this row's process (either <process id=".." fmt="..">  or <process ref="..">)
    row_proc = None
    for p in row.iter("process"):
        pid = p.attrib.get("id") or p.attrib.get("ref")
        if pid:
            row_proc = proc_name.get(pid, "")
            break
    if not row_proc or TARGET_PROCESS not in row_proc:
        skipped += 1
        continue

    # Find this row's pmc-events (a single element with whitespace-separated values)
    for ev in row.iter("pmc-events"):
        txt = (ev.text or "").strip()
        if not txt:
            continue  # ref= with no own text — shouldn't happen for per-row values
        parts = txt.split()
        if len(parts) >= n:
            for i in range(n):
                try: totals[i] += int(parts[i])
                except ValueError: pass
            samples += 1
        break

lines.append(f"  process           : {TARGET_PROCESS}")
lines.append(f"  samples (kept)    : {samples}")
lines.append(f"  samples (skipped) : {skipped}    # dyld/other-process setup")
lines.append("")
w = max(len(k) for k in COUNTER_LABELS)
for i, name in enumerate(COUNTER_LABELS):
    lines.append(f"  {name:<{w}}  total = {totals[i]:>18,}")

# Derived metrics
if samples > 0:
    inst_idx = COUNTER_LABELS.index("INST_ALL") if "INST_ALL" in COUNTER_LABELS else None
    miss_total = sum(totals[i] for i, n_ in enumerate(COUNTER_LABELS) if "MISS" in n_)
    if inst_idx is not None and totals[inst_idx] > 0:
        lines.append("")
        lines.append("=== Derived ===")
        lines.append(f"  L1D misses / 1M instructions : {miss_total * 1_000_000 / totals[inst_idx]:>8.1f}")

open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

# ---- Cleanup ----
if [[ "$KEEP_TRACE" == "0" ]]; then
    rm -rf "$TRACE"
    echo "==> Removed $TRACE (pass --keep-trace to keep it)"
else
    echo "==> Kept $TRACE"
fi

echo "==> Summary: $SUMMARY"
echo "==> Log:     $LOG"
